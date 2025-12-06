import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits, l1_loss, mse_loss
from torch.distributed import all_reduce
from torchvision.ops.boxes import nms, batched_nms
import math
from scipy.optimize import linear_sum_assignment

from utils.box_utils import box_cxcywh_to_xyxy,  box_xyxy_to_cxcywh, generalized_box_iou, box_iou
from utils.distributed_utils import is_dist_avail_and_initialized, get_world_size
from collections import defaultdict
def inverse_sigmoid(x):
    eps = 1e-6
    x = torch.clamp(x, eps, 1-eps)
    return torch.log(x/(1-x))

def batched_diag_iou(boxes1, boxes2, eps=1e-7):
    """
    计算 boxes1[i] 和 boxes2[i] 之间的 IoU（支持任意 dim >= 2，且输入为 xywh 格式）
    
    Args:
        boxes1 (Tensor): (..., 4), xywh 格式
        boxes2 (Tensor): (..., 4), xywh 格式
        eps (float): 防止除以零的小值
    
    Returns:
        Tensor: (...,) IoU 值
    """
    # 转换 xywh -> xyxy
    boxes1_xyxy = torch.empty_like(boxes1)
    boxes1_xyxy[..., 0] = boxes1[..., 0] - boxes1[..., 2] / 2  # x1 = cx - w/2
    boxes1_xyxy[..., 1] = boxes1[..., 1] - boxes1[..., 3] / 2  # y1 = cy - h/2
    boxes1_xyxy[..., 2] = boxes1[..., 0] + boxes1[..., 2] / 2  # x2 = cx + w/2
    boxes1_xyxy[..., 3] = boxes1[..., 1] + boxes1[..., 3] / 2  # y2 = cy + h/2

    boxes2_xyxy = torch.empty_like(boxes2)
    boxes2_xyxy[..., 0] = boxes2[..., 0] - boxes2[..., 2] / 2
    boxes2_xyxy[..., 1] = boxes2[..., 1] - boxes2[..., 3] / 2
    boxes2_xyxy[..., 2] = boxes2[..., 0] + boxes2[..., 2] / 2
    boxes2_xyxy[..., 3] = boxes2[..., 1] + boxes2[..., 3] / 2

    # 计算交集区域 (broadcast 支持任意 dim)
    x1 = torch.max(boxes1_xyxy[..., 0], boxes2_xyxy[..., 0])
    y1 = torch.max(boxes1_xyxy[..., 1], boxes2_xyxy[..., 1])
    x2 = torch.min(boxes1_xyxy[..., 2], boxes2_xyxy[..., 2])
    y2 = torch.min(boxes1_xyxy[..., 3], boxes2_xyxy[..., 3])

    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # 计算并集区域
    area1 = boxes1[..., 2] * boxes1[..., 3]  # w * h
    area2 = boxes2[..., 2] * boxes2[..., 3]
    union_area = area1 + area2 - inter_area

    return inter_area / (union_area + eps)  # IoU

class HungarianMatcher(nn.Module):

    def __init__(self,
                 coef_class: float = 2,
                 coef_bbox: float = 5,
                 coef_giou: float = 2,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 iou_order_alpha: float = 4.0,
                 high_quality_matches: bool = False):
        """Creates the matcher

        Params:
            coef_class: This is the relative weight of the classification error in the matching cost
            coef_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            coef_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.coef_class = coef_class
        self.coef_bbox = coef_bbox
        self.coef_giou = coef_giou
        self.alpha = alpha
        self.gamma = gamma
        self.iou_order_alpha = iou_order_alpha
        self.high_quality_matches = high_quality_matches
        assert coef_class != 0 or coef_bbox != 0 or coef_giou != 0, "all costs cant be 0"

    def forward(self, pred_logits, pred_boxes, annotations):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            annotations: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = pred_logits.shape[:2]

            # We flatten to compute the cost matrices in a batch
            pred_logits = pred_logits.flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
            pred_boxes = pred_boxes.flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            gt_class = torch.cat([anno["labels"] for anno in annotations]).to(pred_logits.device)
            gt_boxes = torch.cat([anno["boxes"] for anno in annotations]).to(pred_logits.device)

            if self.high_quality_matches:
                class_score = pred_logits[:, gt_class]  # shape = [batch_size * num_queries, gt num within a batch]

                # # Compute iou
                bbox_iou, _ = box_iou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(gt_boxes))  # shape = [batch_size * num_queries, gt num within a batch]

                # Final cost matrix
                C = (-1) * (class_score * torch.pow(bbox_iou, self.iou_order_alpha))
            else:  # Default matching
                # Compute the classification cost.
                neg_cost_class = (1 - self.alpha) * (pred_logits ** self.gamma) * (-(1 - pred_logits + 1e-8).log())
                pos_cost_class = self.alpha * ((1 - pred_logits) ** self.gamma) * (-(pred_logits + 1e-8).log())
                cost_class = pos_cost_class[:, gt_class] - neg_cost_class[:, gt_class]

                # Compute the L1 cost between boxes
                cost_boxes = torch.cdist(pred_boxes, gt_boxes, p=1)

                # Compute the giou cost between boxes
                cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(gt_boxes))

                # Final cost matrix
                C = self.coef_bbox * cost_boxes + self.coef_class * cost_class + self.coef_giou * cost_giou

            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(anno["boxes"]) for anno in annotations]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

import torch.nn.functional as  F

class SetCriterion(nn.Module):

    def __init__(self,
                 num_classes=9,
                 coef_class=2,
                 coef_boxes=5,
                 coef_giou=2,
                 alpha_focal=0.25,
                 alpha_dt=0.5,
                 gamma_dt=0.9,
                 max_dt=0.45,
                 device='cuda',
                 high_quality_matches=False):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.matcher = HungarianMatcher(high_quality_matches=high_quality_matches)
        self.coef_class = coef_class
        self.coef_boxes = coef_boxes
        self.coef_giou = coef_giou
        self.alpha_focal = alpha_focal
        self.logits_sum = [torch.zeros(1, dtype=torch.float, device=device) for _ in range(num_classes)]
        self.logits_count = [torch.zeros(1, dtype=torch.int, device=device) for _ in range(num_classes)]
        self.alpha_dt = alpha_dt
        self.gamma_dt = gamma_dt
        self.max_dt = max_dt

    @staticmethod
    def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        prob = inputs.sigmoid()
        ce_loss = binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean(1).sum() / num_boxes

    @staticmethod
    def sigmoid_quality_focal_loss(inputs, targets, scores, num_boxes, alpha: float = 0.25, gamma: float = 2):
        """
        Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
        Qualified and Distributed Bounding Boxes for Dense Object Detection
         <https://arxiv.org/abs/2006.04388>`_.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            scores: A float tensor with the same shape as targets: targets weighted by scores
                    (0 for the negative class and _score (0<_score<=1) for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor to
                balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        prob = inputs.sigmoid()
        ce_loss = binary_cross_entropy_with_logits(inputs, scores, reduction="none")
        # p_t = prob * targets + (1 - prob) * (1 - targets)
        p_t = (scores - prob) * targets + prob * (1 - targets)
        loss = ce_loss * (abs(p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean(1).sum() / num_boxes

    def loss_class(self, pred_logits, annotations, indices, num_boxes, use_pseudo_label_weights=False):
        """Classification loss (NLL)
        annotations dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        idx = self._get_src_permutation_idx(indices)
        gt_classes_o = torch.cat([anno["labels"][j] for anno, (_, j) in zip(annotations, indices)])
        gt_classes = torch.full(pred_logits.shape[:2], self.num_classes, dtype=torch.int64, device=pred_logits.device)
        gt_classes[idx] = gt_classes_o

        gt_classes_onehot = torch.zeros([pred_logits.shape[0], pred_logits.shape[1], pred_logits.shape[2] + 1],
                                        dtype=pred_logits.dtype, layout=pred_logits.layout, device=pred_logits.device)
        gt_classes_onehot.scatter_(2, gt_classes.unsqueeze(-1), 1)
        gt_classes_onehot = gt_classes_onehot[:, :, :-1]

        if use_pseudo_label_weights:
            gt_scores_o = torch.cat([anno["scores"][j] for anno, (_, j) in zip(annotations, indices)])
            gt_scores = torch.full(pred_logits.shape[:2], 0.0, dtype=torch.float, device=pred_logits.device)
            gt_scores[idx] = gt_scores_o
            gt_scores_weight = gt_classes_onehot * gt_scores.unsqueeze(-1)
            loss_ce = self.sigmoid_quality_focal_loss(pred_logits, gt_classes_onehot, gt_scores_weight, num_boxes, alpha=self.alpha_focal, gamma=2) * pred_logits.shape[1]
        else:
            loss_ce = self.sigmoid_focal_loss(pred_logits, gt_classes_onehot, num_boxes, alpha=self.alpha_focal, gamma=2) * pred_logits.shape[1]

        return loss_ce

    def loss_boxes(self, pred_boxes, annotations, indices, num_boxes, use_pseudo_label_weights=False):
        """Compute the losses related to the bounding boxes: the L1 regression loss
           annotations dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The annotations boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = pred_boxes[idx]
        gt_boxes = torch.cat([anno['boxes'][i] for anno, (_, i) in zip(annotations, indices)], dim=0)
        if use_pseudo_label_weights:
            gt_weights = torch.cat([anno['scores'][i] for anno, (_, i) in zip(annotations, indices)], dim=0)
            loss_bbox = l1_loss(src_boxes, gt_boxes, reduction='none') * gt_weights.unsqueeze(-1)
        else:
            loss_bbox = l1_loss(src_boxes, gt_boxes, reduction='none')
        return loss_bbox.sum() / num_boxes

    def loss_giou(self, pred_boxes, annotations, indices, num_boxes, use_pseudo_label_weights=False):
        """Compute the losses related to the bounding boxes: the gIoU loss
           annotations dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The annotations boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = pred_boxes[idx]
        gt_boxes = torch.cat([anno['boxes'][i] for anno, (_, i) in zip(annotations, indices)], dim=0)
        if use_pseudo_label_weights:
            gt_weights = torch.cat([anno['scores'][i] for anno, (_, i) in zip(annotations, indices)], dim=0)
            loss_giou = 1 - torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(gt_boxes)))
            loss_giou = loss_giou * gt_weights
        else:
            loss_giou = 1 - torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(gt_boxes)))
        return loss_giou.sum() / num_boxes

    def record_positive_logits(self, logits, indices):
        idx = self._get_src_permutation_idx(indices)
        labels = logits[idx].argmax(dim=1)
        pos_logits = logits[idx].max(dim=1).values
        for label, logit in zip(labels, pos_logits):
            self.logits_sum[label] += logit
            self.logits_count[label] += 1

    def dynamic_threshold(self, thresholds):
        for s in self.logits_sum:
            all_reduce(s)
        for n in self.logits_count:
            all_reduce(n)
        logits_means = [s.item() / n.item() if n > 0 else 0.0 for s, n in zip(self.logits_sum, self.logits_count)]
        assert len(logits_means) == len(thresholds)
        new_thresholds = [self.gamma_dt * threshold + (1 - self.gamma_dt) * self.alpha_dt * math.sqrt(mean)
                          for threshold, mean in zip(thresholds, logits_means)]
        new_thresholds = [max(min(threshold, self.max_dt), 0.25) for threshold in new_thresholds]
        print('New Dynamic Thresholds: ', new_thresholds)
        return new_thresholds

    def clear_positive_logits(self):
        self.logits_sum = [torch.zeros(1, dtype=torch.float, device=self.device) for _ in range(self.num_classes)]
        self.logits_count = [torch.zeros(1, dtype=torch.int, device=self.device) for _ in range(self.num_classes)]

    @staticmethod
    def _get_src_permutation_idx(indices):
        # Permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        # Permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    @staticmethod
    def _discard_empty_labels(out, annotations):
        reserve_index = []
        for anno_idx in range(len(annotations)):
            if torch.numel(annotations[anno_idx]["boxes"]) != 0:
                reserve_index.append(anno_idx)
        for key, value in out.items():
            if key in ['logit_all', 'boxes_all']:
                out[key] = value[:, reserve_index, ...]
            elif key in ['features']:
                continue
            else:
                out[key] = value[reserve_index, ...]
        annotations = [annotations[idx] for idx in reserve_index]
        return out, annotations

    def forward(self, out, annotations=None, use_pseudo_label_weights=False):
        logit_all = out['logit_all']
        boxes_all = out['boxes_all']
        # import pdb;pdb.set_trace()
        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(anno["labels"]) for anno in annotations) if annotations is not None else 0
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=logit_all.device)
        if is_dist_avail_and_initialized():
            all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        # Compute all the requested losses
        loss = torch.zeros(1).to(logit_all.device)
        loss_dict = defaultdict(float)
        num_decoder_layers = logit_all.shape[0]
        for i in range(num_decoder_layers):
            # Compute DETR losses
            if annotations is not None:
                # Retrieve the matching between the outputs of the last layer and the targets
                indices = self.matcher(logit_all[i], boxes_all[i], annotations)

                # Compute the DETR losses
                loss_class = self.loss_class(logit_all[i], annotations, indices, num_boxes, use_pseudo_label_weights)
                loss_boxes = self.loss_boxes(boxes_all[i], annotations, indices, num_boxes, use_pseudo_label_weights)
                loss_giou = self.loss_giou(boxes_all[i], annotations, indices, num_boxes, use_pseudo_label_weights)
                loss_dict["loss_class"] += loss_class 
                loss_dict["loss_boxes"] += loss_boxes
                loss_dict["loss_giou"] += loss_giou 
                loss += self.coef_class * loss_class + self.coef_boxes * loss_boxes + self.coef_giou * loss_giou

        # Calculate average for all decoder layers
        loss /= num_decoder_layers
        if 'dis_cnn_loss' in out:
            loss = loss + out['dis_cnn_loss'] * 0.1 + out['dis_dino_loss'] * 0.1# + out['construct_loss'] * 0.1
            loss_dict['dis_cnn_loss'] = out['dis_cnn_loss']
            loss_dict['dis_dino_loss'] = out['dis_dino_loss']
        enable_sim_score_loss=  True
        if 'sim_score_0' in out and enable_sim_score_loss:
            #compute centerness with fg box in annotations and generate a heat map 
            #compute mse loss between sim_score and fg centerness mask
            sim_score_loss = 0
            anchor_area = [0.01, 0.1, 0.2]
            temperature = 0.2
            for layer in range(3):
                sim_score = out[f'sim_score_{layer}']
                targets = torch.zeros_like(sim_score) 
                h, w = sim_score.shape
                annots_boxes = annotations[0]['boxes'] * torch.tensor([w,h,w,h]).to(targets.device)
                y_grid, x_grid = torch.meshgrid(
                    torch.arange(h, device=sim_score.device, dtype=torch.float32),
                    torch.arange(w, device=sim_score.device, dtype=torch.float32),
                    indexing='ij'
                )
                #H,W,2
                # scale = 2**(3-layer)
                scale = 1
                with torch.no_grad():
                    anchor_grids = torch.stack([x_grid, y_grid], dim=-1)
                    for cx,cy,bw,bh in annots_boxes:
                        #compute relative scale and scale sensitive weight
                        area = torch.sqrt((bw/w) * (bh/h))
                        ac_area = anchor_area[layer]
                        weight = torch.exp(-(area - ac_area)**2 / temperature**2)
                        
                        #compute distance
                        dist = (anchor_grids - torch.tensor([cx, cy], device=sim_score.device, dtype=torch.float32))**2
                        sigma_w, sigma_h = bw/scale , bh/scale

                        #compute gaussian
                        dist = -0.5 * dist / torch.tensor([sigma_w**2, sigma_h**2], device=sim_score.device, dtype=torch.float32)
                        gaussian = torch.exp(dist.sum(dim=-1)) *  1 / math.sqrt(2*math.pi) 
                        targets = torch.maximum(targets,gaussian)

                #visualize
                import cv2
                # plt.imshow(targets.cpu())
                # vis_sim = F.interpolate(targets.cpu().unsqueeze(0).unsqueeze(0), (h*4,w*4), mode='bilinear')
                # cv2.imwrite(f'test_sim_target_{layer}.jpg', vis_sim[0][0].numpy()*255)

                bce_loss = F.binary_cross_entropy_with_logits(inverse_sigmoid(sim_score), targets, reduction='none').mean()
                # bce_loss = bce_loss[targets!=0].sum()/(targets.sum()+1e-5)
                sim_score_loss += bce_loss
            loss += sim_score_loss 
            loss_dict['sim_score_loss'] = sim_score_loss / 3 / 2
            # loss_dict['construct_loss'] = out['construct_loss']
        # for k, v in loss_dict.items():
        #     loss_dict[k] /= num_decoder_layers
        # feature_alignment_loss = F.mse_loss(out['features'][-1], out['dino_features'][-1], reduction='mean') + F.mse_loss(out['features'][-2], out['dino_features'][-2], reduction='mean') + F.mse_loss(out['features'][-3], out['dino_features'][-3], reduction='mean')
        # loss_dict['feature_alignment_loss'] = feature_alignment_loss.detach()
        # loss += feature_alignment_loss 
        enable_logit_alignment = False
        distill_type = 'logit_mse'
        if enable_logit_alignment and 'teacher_logit_all' in  out:
            if distill_type == 'logit_mse':
                logit_mse_loss = F.mse_loss(out['teacher_logit_all'], out['logit_all'])
                loss = loss + logit_mse_loss
                loss_dict['logit_mse_loss'] = logit_mse_loss
            elif distill_type == 'logit+box':
                logit_mse_loss = F.mse_loss(out['teacher_logit_all'], out['logit_all'])
                box_l1_loss = F.smooth_l1_loss(out['teacher_boxes_all'], out['boxes_all'])
                loss = loss + logit_mse_loss + box_l1_loss 
                loss_dict['logit_mse_loss'] = logit_mse_loss + box_l1_loss
            elif distill_type == 'box_only':
                logit_mse_loss = F.mse_loss(out['teacher_logit_all'], out['logit_all'])
                box_l1_loss = F.smooth_l1_loss(out['teacher_boxes_all'], out['boxes_all'],reduction='none').sum() / num_decoder_layers /num_boxes
                loss = loss +  box_l1_loss 
                loss_dict['logit_mse_loss'] =   box_l1_loss
                
            elif distill_type == 'sigmoid_mse':
                logit_mse_loss = F.mse_loss(out['teacher_logit_all'].sigmoid(), out['logit_all'].sigmoid())
                loss = loss + logit_mse_loss * 10
                loss_dict['logit_mse_loss'] = logit_mse_loss
            elif distill_type == 'kl':
                logit_mse_loss = F.mse_loss(out['teacher_logit_all'].sigmoid(), out['logit_all'].sigmoid())
                loss = loss + logit_mse_loss
                loss_dict['logit_mse_loss'] = logit_mse_loss
            elif distill_type == 'weighted_mse':
                iou_weight = batched_diag_iou(box_cxcywh_to_xyxy(out['teacher_boxes_all']) , box_cxcywh_to_xyxy(out['boxes_all'])).detach()
                logit_mse_loss = iou_weight.unsqueeze(-1) * F.mse_loss(out['teacher_logit_all'], out['logit_all'], reduction='none')
                loss = loss + logit_mse_loss.mean()
                loss_dict['logit_mse_loss'] = logit_mse_loss.mean()
            elif distill_type == 'weighted_kl':
                logit_mse_loss = F.mse_loss(out['teacher_logit_all'].sigmoid(), out['logit_all'].sigmoid())
                loss = loss + logit_mse_loss
                loss_dict['logit_mse_loss'] = logit_mse_loss


        enable_feature_alignment = False
        if enable_feature_alignment and 'teacher_features' in out:
            feature_alignment_loss = F.mse_loss(out['teacher_features'][-1], out['features'][-1]) #+ F.mse_loss(out['teacher_features'][-2], out['features'][-2]) +  F.mse_loss(out['teacher_features'][-3], out['features'][-3])
            loss += feature_alignment_loss
            loss_dict['feature_alignment_loss'] = feature_alignment_loss
        enable_dino_alignment = False
        if enable_dino_alignment and 'dino_features' in out:
            query_embed = out['query_embed'][0]
            #Given boxes_all[-1] as box input (x,y,w,h) mode, 0-1 normalzied  . Size [BS,Len_Query, 4]
            boxes = box_cxcywh_to_xyxy(boxes_all[0])[0] # take bs = 1 as default
            b,c,h,w = out['dino_features'].shape
            rois = boxes * torch.tensor([[w,h,w,h]]).cuda()
            from torchvision.ops import roi_align
            rois = torch.cat([torch.zeros(len(rois), 1).cuda(), rois],  dim=1)
            roi_features = roi_align(out['dino_features'], rois, output_size=(1, 1))
            #Compute ROI features from DINO feature Size [BS, Channel, H, W] 
            loss_dict['query_alignment_loss'] = F.mse_loss(query_embed[0], roi_features.flatten(1), reduction='mean') 
            loss += loss_dict['query_alignment_loss'] * 1
        if 'dino_features_proj' in out:
            loss_dict['dino_feats_proj_norm'] = out['dino_features_proj'][-1].norm(dim=1,p=2).mean()
        
        #update metric
        if 'dino_factor' in out:
            loss_dict['dino_factor'] = out['dino_factor']
        loss_dict.update({k:v for k,v in out.items() if 'metric' in k})
        return loss, loss_dict



@torch.no_grad()
def post_process(pred_logits, pred_boxes, target_sizes, topk=100):
    """ Perform the computation
        Parameters:
            outputs -> pred_logits, pred_boxes: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
    """
    assert len(pred_logits) == len(target_sizes)
    assert target_sizes.shape[1] == 2

    prob = pred_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(prob.view(pred_logits.shape[0], -1), topk, dim=1)
    scores = topk_values
    topk_boxes = torch.div(topk_indexes, pred_logits.shape[2], rounding_mode='trunc')
    labels = topk_indexes % pred_logits.shape[2]
    boxes = box_cxcywh_to_xyxy(pred_boxes)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

    # From relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]
    results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
    return results

def get_topk_outputs(pred_logits, pred_boxes, topk=50):
    """
    Get top_k outputs from pred_logits and pred_boxes
    """
    prob = pred_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(prob.view(pred_logits.shape[0], -1), topk, dim=1)
    topk_boxes = torch.div(topk_indexes, pred_logits.shape[2], rounding_mode='trunc')
    labels = topk_indexes % pred_logits.shape[2]

    # took_pred_boxes :[batch_size, topk, 4]
    topk_pred_boxes = torch.gather(pred_boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

    # topk_pred_logits : [batch_size, topk, num_classes]
    topk_pred_logits = torch.gather(pred_logits, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, pred_logits.shape[-1]))

    topk_outputs = {'labels_topk': labels, 'boxes_topk': topk_pred_boxes, 'logits_topk': topk_pred_logits}
    return topk_outputs

def get_pseudo_labels(pred_logits, pred_boxes, thresholds, is_nms=False, nms_threshold=0.7):
    probs = pred_logits.sigmoid()
    scores_batch, labels_batch = torch.max(probs, dim=-1)
    pseudo_labels = []
    thresholds_tensor = torch.tensor(thresholds, device=pred_logits.device)
    for scores, labels, pred_box in zip(scores_batch, labels_batch, pred_boxes):
        larger_idx = torch.gt(scores, thresholds_tensor[labels]).nonzero()[:, 0]
        scores, labels, boxes = scores[larger_idx], labels[larger_idx], pred_box[larger_idx, :]
        if is_nms:
            nms_idx = nms(box_cxcywh_to_xyxy(boxes), scores, iou_threshold=nms_threshold)
            scores, labels, boxes = scores[nms_idx], labels[nms_idx], boxes[nms_idx, :]
        pseudo_labels.append({'scores': scores, 'labels': labels, 'boxes': boxes})
    return pseudo_labels


def merge_pseudo_labels(pseudo_labels_0, pseudo_labels_1, iou_threshold=0.7, weights=[1,1], fuse_type='nms'):
    """
    Merge two sets of pseudo labels using Non-Maximum Suppression (NMS)
    
    Args:
        pseudo_labels_0: First set of pseudo labels (list of dicts)
        pseudo_labels_1: Second set of pseudo labels (list of dicts)
        iou_threshold: IoU threshold for NMS
    
    Returns:
        Merged pseudo labels (list of dicts)
    """
    from .wbf  import weighted_boxes_fusion,nms
    merged_labels = []
    skip_box_thr = 0.001
    #iter through batch
    for labels_0, labels_1 in zip(pseudo_labels_0, pseudo_labels_1):
        # Combine detections from both sets
        if fuse_type == 'nms':
            boxes_list = box_cxcywh_to_xyxy(torch.cat([labels_0['boxes'], labels_1['boxes']], dim=0)).cpu()
            scores_list = torch.cat([labels_0['scores'], labels_1['scores']], dim=0).cpu()
            labels_list = torch.cat([labels_0['labels'], labels_1['labels']], dim=0).cpu()
            # boxes, scores, labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_threshold)
            from torchvision.ops import batched_nms
            keep_idx = batched_nms(boxes_list, scores_list, labels_list, iou_threshold=iou_threshold)
            boxes = boxes_list[keep_idx]
            scores = scores_list[keep_idx]
            labels = labels_list[keep_idx]
        elif fuse_type == 'wbf':
            boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        merged_labels.append({
            'scores': (scores).cuda().float(),
            'labels': (labels).cuda().long(),
            'boxes': box_xyxy_to_cxcywh((boxes).cuda().float())
        })
    # print(merged_labels)
    return merged_labels