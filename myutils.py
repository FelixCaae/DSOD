import logging
import os
import copy
import torch.optim as optim
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch, SimpleTrainer
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
    ClipartDetectionEvaluator,
    WatercolorDetectionEvaluator,
    CityscapeDetectionEvaluator,
    FoggyDetectionEvaluator,
    CityscapeCarDetectionEvaluator,
)

from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage

import pdb
import cv2
from pynvml import *
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.data.detection_utils import convert_image_to_rgb

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import json
import torch
import cv2
import os
import numpy as np
from itertools import groupby
# from segment_anything import SamPredictor, sam_model_registry
# from segment_anything.utils.amg import batched_mask_to_box
import torch.nn.functional as F
import tqdm

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.modeling.roi_heads import build_roi_heads, StandardROIHeads
from detectron2.engine import HookBase
from detectron2.layers import  ShapeSpec
from detectron2.structures import Boxes, pairwise_iou
from detectron2.utils.events import TensorboardXWriter, EventWriter, get_event_storage
from detectron2.config import CfgNode as CN
import torch.distributed as dist
from collections import deque
def get_gpu_memory_usage():
    """获取当前GPU显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # 峰值显存
        
        print("="*50)
        print("GPU显存使用情况:")
        print(f"当前已分配: {allocated:.3f} GB")
        print(f"当前已保留: {reserved:.3f} GB") 
        print(f"峰值显存: {max_allocated:.3f} GB")
        print("="*50)
        
        return allocated, reserved, max_allocated
    else:
        print("CUDA不可用")
        return 0, 0, 0

def inverse_sigmoid(x):
    x = x.clamp(1e-6, 1-1e-6)
    return torch.log(x/(1-x))

def box_jitter(boxes, scale):
    assert isinstance(boxes, torch.Tensor)
    n = boxes.shape[0]
    device = boxes.device
    
    # Calculate width and height for each box
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    
    # Generate random shifts in x and y directions
    shift_x = w * torch.empty(n, device=device).uniform_(-scale, scale)
    shift_y = h * torch.empty(n, device=device).uniform_(-scale, scale)
    
    # Clone the boxes to avoid modifying the original
    jittered_boxes = boxes.clone()
    
    # Apply shifts to x-coordinates
    jittered_boxes[:, 0] += shift_x
    jittered_boxes[:, 2] += shift_x
    
    # Apply shifts to y-coordinates
    jittered_boxes[:, 1] += shift_y
    jittered_boxes[:, 3] += shift_y
    
    return jittered_boxes
# class MLP(nn.Module):
#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dim: int,
#         output_dim: int,
#         num_layers: int,
#         sigmoid_output: bool = False,
#     ) -> None:
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(
#             nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
#         )
#         self.sigmoid_output = sigmoid_output

#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
#         if self.sigmoid_output:
#             x = F.sigmoid(x)
#         return x
class Counter():
    def __init__(self, maxlen=50):
        self.count_1 = deque(maxlen=maxlen)
        self.count_2 = deque(maxlen=maxlen)
        
    def update(self, num1, num2):
        assert (num1 >= 0) and (num2>=0)
        self.count_1.append(float(num1))
        self.count_2.append(float(num2))
    def reset(self):
        pass
        # self.count_1 = torch.tensor(0).cuda()
        # self.count_2 = torch.tensor(0).cuda()
    def get(self):
        # 首先计算本地结果
        # 如果启用了分布式环境，进行all_reduce操作
        if dist.is_initialized():
            # 创建用于reduce的张量
            sum_1 = sum(self.count_1)
            sum_1 = torch.tensor(sum_1).cuda()
            sum_2 = sum(self.count_2)
            sum_2 = torch.tensor(sum_2).cuda()
            # 使用all_reduce求和
            dist.all_reduce(sum_1, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_2, op=dist.ReduceOp.SUM)            
            # 计算全局均值
            mean_result = float(sum_1) / max(float(sum_2),1)
            return mean_result
        else:
            return  sum(self.count_1)/max(sum(self.count_2), 1)
class EvalPseudoLabelHook(HookBase):
    def __init__(self, trainer, period):
        self._period = period
        self.trainer = trainer
        self.frcnn_acc = Counter()
        self.frcnn_rec = Counter()
        self.dino_acc = Counter()
        self.dino_rec = Counter()
        self.counter_list = [self.frcnn_acc, self.frcnn_rec, self.dino_acc, self.dino_rec]
    def after_step(self):
        # eval frcnn
        if (self.trainer.iter + 1) % self.trainer.len_epoch ==0:
            for counter in self.counter_list:
                counter.reset()
        if (self.trainer.iter + 1) % self._period == 0 or (
            self.trainer.iter == self.trainer.max_iter - 1
        ):
            storage = get_event_storage()
            if not hasattr(self.trainer, 'data'):
                return
            num_gt = max(len(self.trainer.data[0]['instances']), 1)
            if hasattr(self.trainer, "teacher_pseudo_results_dino"):
                num_pred_dino = len(self.trainer.teacher_pseudo_results_dino[0])
                error, correct = count_cls_errors(self.trainer.teacher_pseudo_results_dino[0], self.trainer.data[0]['instances'].to(torch.device('cuda')))
                self.dino_acc.update(correct, num_pred_dino)
                self.dino_rec.update(correct, num_gt)
                dino_acc = self.dino_acc.get()
                dino_rec = self.dino_rec.get()
                if comm.is_main_process():
                    storage.put_scalar("dino_pl_accuracy", dino_acc)
                    storage.put_scalar("dino_pl_recall", dino_rec)
            if hasattr(self.trainer, "teacher_pseudo_results"):
                num_pred_frcnn = max(len(self.trainer.teacher_pseudo_results[0]), 1)
                error, correct = count_cls_errors(self.trainer.teacher_pseudo_results[0], self.trainer.data[0]['instances'].to(torch.device('cuda')))
                self.frcnn_acc.update(correct, num_pred_frcnn)
                self.frcnn_rec.update(correct, num_gt)
                frcnn_acc = self.frcnn_acc.get()
                frcnn_rec = self.frcnn_rec.get()
                if hasattr(self.trainer.teacher_pseudo_results[0], 'source'):
                    source_sam = (1-self.trainer.teacher_pseudo_results[0].source).sum()
                    source_frcnn = (self.trainer.teacher_pseudo_results[0].source).sum()
                    if dist.is_initialized():
                        dist.all_reduce(source_sam, op=dist.ReduceOp.SUM)
                        dist.all_reduce(source_frcnn, op=dist.ReduceOp.SUM)
                else:
                    source_sam = 0
                    source_frcnn = torch.tensor(len(self.trainer.teacher_pseudo_results[0]))
                    source_frcnn = source_frcnn.cuda()
                    if dist.is_initialized():
                        dist.all_reduce(source_frcnn, op=dist.ReduceOp.SUM)
                if comm.is_main_process():
                    storage.put_scalar("frcnn_pl_accuracy", frcnn_acc)
                    storage.put_scalar("frcnn_pl_recall", frcnn_rec)
                    storage.put_scalar('frcnn_pred_num', num_pred_frcnn)
                    storage.put_scalar('mixed_label_frcnn_num', source_frcnn)
                    storage.put_scalar('mixed_label_sam_num', source_sam)

            if comm.is_main_process():
                storage.put_scalar('low_thres', self.trainer.low_thres)
                storage.put_scalar('high_thres', self.trainer.high_thres)
                storage.put_scalar('num_gt', num_gt)
                if hasattr(self, 'valid_mask'):
                    storage.put_scalar('filtered_box_num', (~self.valid_mask).sum())
                    
        
class SINEFactorHook(HookBase):
    """
    Write events to EventStorage (by calling ``writer.write()``) periodically.

    It is executed every ``period`` iterations and after the last iteration.
    Note that ``period`` does not affect how data is smoothed by each writer.
    """

    def __init__(self, model_student, model_teacher, period, max_epoch=10, beta=0.75, schedule_type='sine'):
        """
        Args:
            writers (list[EventWriter]): a list of EventWriter objects
            period (int):
        """
        self.model_student  = model_student
        self.model_teacher = model_teacher
        self._period = period
        self._beta = beta
        self.max_epoch = max_epoch
        self.schedule_type = schedule_type
    def get_value(self, epoch, saturate_epoch=10, schedule_type='sine'):
        assert epoch >=0 
        import math
        epoch = min(epoch, saturate_epoch)
        if schedule_type == 'sine':
            return math.sin(epoch/saturate_epoch * math.pi / 2)
        elif schedule_type == 'cosine':
            return math.cos(epoch/saturate_epoch * math.pi / 2)
        elif schedule_type == 'sine+1':
            return math.sin(epoch/saturate_epoch * math.pi / 2) + 1
        elif schedule_type == 'none':
            return 1
        else:
            logger.warning(f"{schedule_type} not supported!")
            raise NotImplementedError
    @torch.no_grad()
    def after_step(self):
        import math
        if (self.trainer.iter + 1) % self._period == 0 or (
            self.trainer.iter == self.trainer.max_iter - 1
        ) or (self.trainer.iter==0):
            epoch = (self.trainer.iter+1) // self._period
            factor1 = self.get_value(epoch, self.max_epoch, self.schedule_type)
            factor2 = self.get_value(epoch+1, self.max_epoch, self.schedule_type)
            #Execute only if epoch ends 
            if comm.get_world_size()>1:
                model_teacher = self.model_teacher.module
                model_student = self.model_student.module
            else:
                model_teacher = self.model_teacher
                model_student = self.model_student

            model_student.dino_factor = factor2
            model_teacher.dino_factor = factor1
            print(factor1, factor2)
            if comm.is_main_process():
                storage = get_event_storage()
                storage.put_scalar('dino_factor/student', factor2)
                storage.put_scalar('dino_factor/teacher', factor1)

class EMAHook(HookBase):
    """
    Write events to EventStorage (by calling ``writer.write()``) periodically.

    It is executed every ``period`` iterations and after the last iteration.
    Note that ``period`` does not affect how data is smoothed by each writer.
    """

    def __init__(self, model_student, model_teacher, period, beta=0.75):
        """
        Args:
            writers (list[EventWriter]): a list of EventWriter objects
            period (int):
        """
        self.model_student  = model_student
        self.model_teacher = model_teacher
        self._period = period
        self._beta = beta
    @torch.no_grad()
    def after_step(self):
        if (self.trainer.iter + 1) % self._period == 0 or (
            self.trainer.iter == self.trainer.max_iter - 1
        ):
            #Execute only if epoch ends 
            if comm.get_world_size()>1:
                model_teacher = self.model_teacher.module
                model_student = self.model_student.module
            else:
                model_teacher = self.model_teacher
                model_student = self.model_student

            student_model_dict = model_student.state_dict()
            new_teacher_dict = OrderedDict()
            for key, value in model_teacher.state_dict().items():
                if key in student_model_dict.keys():
                    new_teacher_dict[key] = (
                        student_model_dict[key] *
                        (1 - self._beta) + value * self._beta
                    )
                # elif 'fuse_attn' in key or 'fuse_mlp' in key:
                #     new_teacher_dict[key] = (
                #         student_model_dict[key] *
                #         (1 - 0.) + value * 0.
                #     )
                else:
                    raise Exception("{} is not found in student model".format(key))
            model_teacher.load_state_dict(new_teacher_dict)
                
class WandbWriter(EventWriter):
    """
    Write all scalars to a wandb file.
    """

    def __init__(self, cfg, window_size: int = 20, **kwargs):
        """
        Args:
            log_dir (str): the directory to save the output events
            window_size (int): the scalars will be median-smoothed by this window size

            kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        self._window_size = window_size

        self._writer = wandb.init(
        project=cfg.WANDB.PROJECT,  # 建议在cfg中自定义WANDB字段
        name= cfg.WANDB.NAME,
        tags=cfg.WANDB.TAGS if hasattr(cfg, "WANDB") else ["detectron2"],
        config={
            "model": dict(cfg.MODEL),
            "optimizer": {
                "lr": cfg.SOLVER.BASE_LR,
                "batch_size": cfg.SOLVER.IMS_PER_BATCH
            },
            "data": cfg.DATASETS.TRAIN
        },
        mode="online"  # 离线模式开关
    )
    
    # # 可选：记录代码快照
    #     if cfg.WANDB.LOG_CODE and not cfg.WANDB.OFFLINE:
    #         wandb.run.log_code(".", include_fn=lambda p: p.endswith(".py"))
    #         self._last_write = -1
    
    def write(self):
        storage = get_event_storage()
        new_last_write = self._last_write
        for k, (v, iter) in storage.latest_with_smoothing_hint(self._window_size).items():
            if iter > self._last_write:
                self._writer.log({k: v}, step=iter)
                new_last_write = max(new_last_write, iter)
        self._last_write = new_last_write

        # visualize training samples
        if len(storage._vis_data) >= 1:
            for img_name, img, step_num in storage._vis_data:
                log_img = Image.fromarray(img.transpose(1, 2, 0))  # convert to (h, w, 3) PIL.Image
                log_img = wandb.Image(log_img, caption=img_name)
                self._writer.log({img_name: [log_img]})
            # Storage stores all image data and rely on this writer to clear them.
            # As a result it assumes only one writer will use its image data.
            # An alternative design is to let storage store limited recent
            # data (e.g. only the most recent image) that all writers can access.
            # In that case a writer may not see all image data if its period is long.
            storage.clear_images()
    
    def close(self):
        if hasattr(self, "_writer"):
            self._writer.finish()


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # 在合并前动态添加WANDB配置组
    cfg.MT = CN()
    cfg.MT.LOW_THRES = 0.7
    cfg.MT.HIGH_THRES = 1.0
    cfg.WANDB = CN()
    cfg.WANDB.PROJECT = "default_project"
    cfg.WANDB.NAME = ""
    cfg.WANDB.TAGS = []
    cfg.WANDB.OFFLINE = False
    cfg.WANDB.LOG_CODE = False
    cfg.MODEL.LPL_KL = CN()
    cfg.MODEL.LPL_KL.ENABLED = False
    cfg.DINOHEAD = CN()
    cfg.DINOHEAD.ENABLED = False
    cfg.DINOHEAD.PL_REJECT = False
    cfg.DINOHEAD.DINO_ONLY = False
    cfg.DYNAMIC_DINO = CN()
    cfg.DYNAMIC_DINO.ENABLED = False
    cfg.DYNAMIC_DINO.SCHEDULE = 'sine+1'
    cfg.DYNAMIC_DINO.FUSE_TYPE = 'add'
    cfg.DYNAMIC_DINO.DINO_ARCH = 'dinov2_vitb14_reg'
    cfg.DYNAMIC_DINO.DINO_WEIGHTS = 'weights/dinov2_vitb14_reg4_pretrain.pth'
    cfg.BOX_FILTERING = CN()
    cfg.BOX_FILTERING.TYPE = 'hard'
    cfg.BOX_FILTERING.ENABLED = False
    cfg.BOX_FILTERING.JITTER_SCALE = 0.2
    cfg.BOX_FILTERING.JITTER_NUM = 5
    cfg.BOX_FILTERING.CONF_THRESH = 0.055
    cfg.WEIGHT_LABELS = CN()
    cfg.WEIGHT_LABELS.ENABLED = False
    cfg.WEIGHT_LABELS.CLS_WEIGHT = ""
    cfg.WEIGHT_LABELS.REG_WEIGHT = ""
    cfg.EXTRA_PL = CN()
    cfg.EXTRA_PL.AUX_PROPOSALS = False
    cfg.EXTRA_PL.AUX_PSEUDO_LABELS = False
    cfg.EXTRA_PL.UPDATE = False
    cfg.EXTRA_PL.ENABLED = False
    cfg.EXTRA_PL.PATH = 'sam_pred.json'
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def count_cls_errors(d1, gt, iou_thresh = 0.5, ignore_thresh=0.95):
    #compute fp and tp
    num_fp = 0
    num_tp = 0
    d1_box = d1.gt_boxes
    d1_score = d1.scores
    d1_class = d1.gt_classes
    d2_box = gt.gt_boxes
    d2_class = gt.gt_classes
    iou_matrix = pairwise_iou(d1_box, d2_box)
    for i in range(len(d1_box)):
        d2_iou, d2_ind = (iou_matrix[i]).max(dim=0)
        # no matched dino prediction
        if d2_iou <=  0.5:
            num_fp += 1
            continue
        #consistency logit
        if (d1_class[i] != d2_class[d2_ind]):
            num_fp += 1
        else:
            num_tp += 1
    return num_fp, num_tp

# 1. 初始化 SAM 模型
def load_sam(model_type="vit_l", checkpoint_path="../CrowdSAM/weights/sam_vit_l_0b3195.pth"):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).cuda()
    # sam.load_state_dict(torch.load('../CrowdSAM/adapter_weights/cityscape_adapter.pth'))
    predictor = SamPredictor(sam)
    return predictor

def load_dinov2(dino_repo ='dinov2',model_type="dinov2_vitb14_reg", checkpoint_path="./weights/dinov2_vitb14_reg4_pretrain.pth"):
    dino = torch.hub.load(dino_repo, model_type,source='local',pretrained=False).cuda()
    dino.load_state_dict(torch.load(checkpoint_path))
    from torchvision import transforms
    transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正则化
    ])
    return dino, transform

def build_standard_roi_head(num_classes, hidden_dim=1024):
    # 1. 创建默认配置
    cfg = get_cfg()
    
    # 2. 设置 ROI_HEADS 相关参数
    cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # COCO 数据集类别数
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # 每张图像的 RoI 数量
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25  # 正样本比例
    
    # 3. RoI Pooling 设置（例如 ROIAlign）
    cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7  # 输出特征图大小
    cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 2
    
    # 4. Box Head 设置（MLP 或 CNN）
    cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
    cfg.MODEL.ROI_BOX_HEAD.NUM_CONV = 2  # 卷积层数量
    cfg.MODEL.ROI_BOX_HEAD.CONV_DIM = 1024  # 卷积层通道数
    cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 1  # 全连接层数量
    cfg.MODEL.ROI_BOX_HEAD.FC_DIM = 1024  # 隐藏层维度
    
    # 5. 构建模型（ROI Heads 会自动构建）
    roi_heads = build_roi_heads(cfg, {'res4':ShapeSpec(768,7,7,stride=16)})
    # roi_heads = model.roi_heads  # 获取 StandardROIHeads 实例
    
    return roi_heads.cuda()

def build_res5_roi_head(num_classes, backbone_out_channels=1024):
    cfg = get_cfg()
    
    # ---------- 核心配置 ----------
    cfg.MODEL.ROI_HEADS.NAME = "Res5ROIHeads"  # 关键修改点
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    
    # ROI Pooling设置
    cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7  # 输入res5的尺寸（比StandardROIHeads大）
    cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 2
    
    # Res5配置
    cfg.MODEL.RESNETS.NUM_GROUPS = 1
    cfg.MODEL.RESNETS.WIDTH_PER_GROUP = 64
    cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS = backbone_out_channels  # 通常为1024或2048
    
    # Head输出设置
    cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 1  # res5后仍可接FC层（可选）
    cfg.MODEL.ROI_BOX_HEAD.FC_DIM = 1024
    
    # ---------- 构建模型 ----------
    # 输入特征需匹配backbone的res4输出
    input_shape =  {'res4':ShapeSpec(768,7,7,stride=16)}
    
    # 显式构建Res5ROIHeads（需传入backbone的部分参数）
    roi_heads = Res5ROIHeads(
        cfg,
        input_shape,
        box_in_features=["res4"],
        box_pooler=None,  # 会自动创建
    )
    
    return roi_heads
# 3. 处理单个检测结果
def refine_detection(predictor, boxes):
    # 读取图像
    # image = cv2.imread(image_path)
    # 设置图像（SAM需要先编码）

    # 提取原始边界框（格式：[x1, y1, x2, y2]）
    # boxes = [d['bbox'] for d in detection]
    assert len(boxes) > 0
    input_boxes = np.array(boxes)  # SAM需要的格式
    input_boxes = predictor.transform.apply_boxes(input_boxes, predictor.original_size)
    input_boxes = torch.tensor(input_boxes).cuda()
    # SAM预测分割掩码
    batch_size = 16
    masks_total = []
    start_idx = 0
    while start_idx < len(input_boxes):
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=input_boxes[start_idx:start_idx+batch_size],  # 添加batch维度
            multimask_output=False,    # 只返回一个最佳掩码
        )
        masks_total.append(masks)
        start_idx = start_idx + batch_size
    masks = torch.cat(masks_total, dim=0)[:, 0]  # 取第一个掩码

    # 从掩码计算优化后的边界框
    # refined_bbox = [mask_to_bbox(mask) for mask in masks ]
    refined_bbox = batched_mask_to_box(masks)
    # 更新检测结果
    return refined_bbox

def filter_LPL(model_teacher, t_features, t_proposals, t_results):

    t_box_features = model_teacher.roi_heads._shared_roi_transform([t_features['res4']], [t_proposals[0].proposal_boxes])
    t_box_features_mean = t_box_features.mean(dim=[2, 3])
    t_box_features_norm = F.normalize(t_box_features_mean, dim=1)
    t_roih_logits = model_teacher.roi_heads.box_predictor(t_box_features_mean)

    # Compute the cosine similarity between the student and teacher features
    # c_similarity = F.cosine_similarity(s_box_features_norm.detach(), t_box_features_norm.detach(), dim=1)
    t_roih_classes = t_roih_logits[0].argmax(dim=1)

    # Compute the LPL loss
    t_proposal_boxes = torch.cat([p.proposal_boxes.tensor for p in t_proposals], dim=0)
    t_boxes = model_teacher.roi_heads.box_predictor.box2box_transform.apply_deltas(t_roih_logits[1], t_proposal_boxes)

    #t_box is a set of candidates
    t_box  = torch.zeros(t_proposal_boxes.shape[0], 4).cuda()

    for index, cl in enumerate(t_roih_classes):
        #background class
        if cl == 8:
            t_box[index] = t_proposal_boxes[index]
        else:

            t_box[index] =  t_boxes[index, 4*cl:4*cl+4]
    #对于关联前景类别的样例，使用roi box作为备选框，否则使用rpn box作为备选框
    t_box_boxes = Boxes(t_box)
    t_result_boxes = Boxes(t_results[0].gt_boxes.tensor)
    iou_matrix = pairwise_iou(t_box_boxes, t_result_boxes)

    if iou_matrix.shape[1] == 0:
        return None

    #过滤掉与前景相关联程度较大的
    t_indices = torch.nonzero(torch.max(iou_matrix, dim=1).values <= 0.4).flatten()
    if t_indices.nelement() == 0:
        return None
    t_softmax_w_bg = F.softmax(t_roih_logits[0][t_indices], dim=1)
    t_softmax_wo_bg = F.softmax(t_roih_logits[0][t_indices][:, :-1], dim=1)
    wo_bg_max = torch.max(t_softmax_wo_bg, dim=1)[0]
    w_bg_prob = t_softmax_w_bg[:, -1]
    # 背景分数小于某一阈值 且去背景后分数大于某一阈值， 即低置信度前景结果
    t_indices_filtered = t_indices[(wo_bg_max >= 0.9) & (w_bg_prob <= 0.99)]

    if t_indices_filtered.nelement() == 0:
        return None
    return  t_indices_filtered
logger = logging.getLogger("detectron2")
def set_all_seeds(seed):
    # 设置Python随机种子
    import random
    random.seed(seed)
    
    # 设置NumPy随机种子
    np.random.seed(seed)
    
    # 设置PyTorch随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 设置CUDA卷积运算的确定性（可能影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if evaluator_type == "clipart":
        return ClipartDetectionEvaluator(dataset_name)
    if evaluator_type == "watercolor":
        return WatercolorDetectionEvaluator(dataset_name)
    if evaluator_type == "cityscape":
        return CityscapeDetectionEvaluator(dataset_name)
    if evaluator_type == "foggy":
        return FoggyDetectionEvaluator(dataset_name, output_folder)
    if evaluator_type == "cityscape_car":
        return CityscapeCarDetectionEvaluator(dataset_name)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)
# =====================================================
# ================== DINO　 ===========================
# =====================================================
@torch.no_grad()
def forward_dino(model_dino, transforms, data, output_shape=None, keyword='image_weak', max_length=560, patch_size=14,):
    assert max_length % patch_size == 0
    output = []
    for item in data:
        img_weak = item[keyword]
        _,h,w = img_weak.shape
        x = transforms(img_weak/255).unsqueeze(0)
        if h <= w:
            s_w = max_length
            s_h = s_w * (h/w)
            s_h = (s_h // patch_size) * patch_size
        else:
            s_h = max_length
            s_w = s_h * (w/h)
            s_w = (s_w // patch_size) * patch_size
        s_h, s_w = int(s_h), int(s_w)
        x = F.interpolate(x, (s_h, s_w), mode='bilinear')
        feat_weak = model_dino.forward_features(x.cuda())
        feat_weak = feat_weak['x_norm_patchtokens'].reshape(1, s_h//patch_size, s_w//patch_size,-1).permute(0,3,1,2)
        if output_shape is None:
            output_shape = s_h//patch_size, s_w//patch_size
        feat_weak = F.interpolate(feat_weak, output_shape, mode='bilinear')
        output.append(feat_weak)
    return torch.cat(output, dim=0)

# =====================================================
# ================== Pseduo-labeling ==================
# =====================================================
def threshold_bbox(proposal_bbox_inst, low_thres=0.7, high_thres=1.0, proposal_type="roih"):
    if proposal_type == "rpn":
        valid_map = (proposal_bbox_inst.objectness_logits > low_thres) & (proposal_bbox_inst.objectness_logits < high_thres)

        # create instances containing boxes and gt_classes
        image_shape = proposal_bbox_inst.image_size
        new_proposal_inst = Instances(image_shape)

        # create box
        new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
        new_boxes = Boxes(new_bbox_loc)

        # add boxes to instances
        new_proposal_inst.gt_boxes = new_boxes
        new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
            valid_map
        ]
    elif proposal_type == "roih":
        valid_map = (proposal_bbox_inst.scores > low_thres) & (proposal_bbox_inst.scores < high_thres) 

        # create instances containing boxes and gt_classes
        image_shape = proposal_bbox_inst.image_size
        new_proposal_inst = Instances(image_shape)

        # create box
        new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
        new_boxes = Boxes(new_bbox_loc)

        # add boxes to instances
        new_proposal_inst.gt_boxes = new_boxes
        new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
        new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

    return new_proposal_inst

def draw_box(image, box, score=None, color=[255,255,0],font_scale = 0.5,thickness = 2):
    assert isinstance(image, np.ndarray)
    assert (image.shape[0] > 0) and (image.shape[1]>0)
    if isinstance(color, np.ndarray):
        color = color.tolist()
    font = cv2.FONT_HERSHEY_SIMPLEX
    if score is not None:
        cv2.putText(image, str(score), (int(box[0]),int(box[1])),color=color,fontScale=font_scale, fontFace=font, thickness=thickness)
        # ax.text(box[0],box[1], str(round(score,3)), color='green')
    cv2.rectangle(image, (int(box[0]),int(box[1])), (int(box[2]), int(box[3])), color=color)
    return image
def re_evaluate(d1, d2, iou_thresh = 0.5, ignore_thresh=0.95):
    flag_ind = []
    d1_box = d1.pred_boxes
    d1_score = d1.scores
    d1_class = d1.pred_classes
    d2_box = d2.pred_boxes
    d2_class = d2.pred_classes
    iou_matrix = pairwise_iou(d1_box, d2_box)
    if len(d2_box)==0:
        return flag_ind
    for i in range(len(d1_box)):
        d2_iou, d2_ind = iou_matrix[i].max(dim=0)
        if d2_iou < iou_thresh:
            continue
        if d1_class[i] != d2_class[d2_ind]:
            flag_ind.append(i)
#original version of dino refining
def re_evaluate_v0(d1, d2, iou_thresh = 0.5, ignore_thresh=0.95):
    flag_ind = []
    d1_box = d1.gt_boxes
    d1_score = d1.scores
    d1_class = d1.gt_classes
    d2_box = d2.pred_boxes
    d2_class = d2.pred_classes
    iou_matrix = pairwise_iou(d1_box, d2_box)
    #If there is no voter then return
    if len(d2_box)==0:
        return flag_ind
    for i in range(len(d1_box)):
        #Find related voters
        d2_ind = (iou_matrix[i] > iou_thresh).nonzero().flatten()        
        #If there is no voter then continue
        if len(d2_ind)==0:
            continue
        # Vote and then decide by major
        t = (d1_class[i] == d2_class[d2_ind])
        if t.sum()/len(t) < 0.5:
            flag_ind.append(i)
#TODO: Maybe Ensemble Two sets of predictions rather than filtering
def merge_gt(gt1, gt2):
    pass

def vis_compare(cfg, batched_inputs, det_1 , det_2, metadata):
    from detectron2.utils.visualizer import Visualizer
    for input, d1,d2 in zip(batched_inputs, det_1, det_2):
        flag_ind = re_evaluate(d1, d2)
        img = input["image_weak"]
        gt_inst = input['instances']
        img = convert_image_to_rgb(img.permute(1, 2, 0), None)
        v_pred = Visualizer(img, metadata)
        # draw_box(img, d1_box[i].tensor.int().flatten(), f"{int(d1_class[i])}:{round(float(d1_score[i]),3)}",color=[255,0,0])
        v_pred = v_pred.draw_instance_predictions(d1[flag_ind])
        vis_img = v_pred.get_image()
        v_pred_gt = Visualizer(img, metadata)
        vis_gt = v_pred_gt.draw_dataset_dict(batched_inputs[0]).get_image()
        vis_img = np.concatenate([vis_img, vis_gt], axis=1)
        save_path = os.path.join(cfg.OUTPUT_DIR, 'compare_vis') 
        save_img_path = os.path.join(cfg.OUTPUT_DIR, 'compare_vis', input['file_name'].split('/')[-1]) 
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(save_img_path, vis_img)

def visualize_result(data, det, class_names, save_path=None, conf_thresh=0.001, special_ind=None,special_color=[255,0,0]):
    from detectron2.utils.visualizer import Visualizer
    assert isinstance(det, Instances)
    #convert image from PIL to numpy 
    image = data['image_weak'].cpu().permute(1,2,0).numpy().copy()
    # det is a d2 Instance structure that contains pred_boxes pred_classes and scores
    assert hasattr(det, 'pred_boxes')
    assert hasattr(det, 'pred_classes')
    assert hasattr(det, 'scores')
    assert isinstance(conf_thresh, float)

    #iterate through all boxes and masks
    for i in range(len(det.pred_boxes)):
        if special_ind is not  None and i in special_ind:
            color = np.array(special_color)
        else:
            color = (np.random.rand(3) * 255).astype('int')

        box = det.pred_boxes.tensor[i]
        score =round(float(det.scores[i]),3)
        class_id = det.pred_classes[i]
        if score  < conf_thresh:
            continue
        if class_names is not None: 
            class_name = class_names[class_id] 
        else:
            class_name = str(class_id)
        image = draw_box(image, box,f"{class_name}:{score}",color=color.tolist())
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = os.path.basename(data['file_name'])
        save_path = os.path.join(save_path, f"{filename}")
        cv2.imwrite(save_path, img)
    return image
    #if FN given, draw FN box in blue
    
def visualize_proposals(cfg, batched_inputs, proposals, proposal_dir, metadata, box_size=300):
        from detectron2.utils.visualizer import Visualizer

        for input, prop in zip(batched_inputs, proposals):
            img = input["image_weak"]
            gt_inst = input['instances']
            img = convert_image_to_rgb(img.permute(1, 2, 0), None)
            #v_gt = Visualizer(img, None)
            #v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            #anno_img = v_gt.get_image()
            v_pred_gt = Visualizer(img, metadata)
            vis_gt = v_pred_gt.draw_dataset_dict(batched_inputs[0]).get_image()
            v_pred = Visualizer(img, metadata)
            if proposal_dir == "rpn":
                v_pred = v_pred.overlay_instances( boxes=prop.gt_boxes[0:int(box_size)].tensor.cpu().numpy())
            else:
                if hasattr(prop,  'gt_boxes'):
                    prop.pred_boxes = prop.gt_boxes
                    prop.pred_classes = prop.gt_classes
                v_pred = v_pred.draw_instance_predictions(prop)
            vis_img = v_pred.get_image()
            vis_img = np.concatenate([vis_img, vis_gt], axis=1)
            save_path = os.path.join(cfg.OUTPUT_DIR, proposal_dir) 
            save_img_path = os.path.join(cfg.OUTPUT_DIR, proposal_dir, input['file_name'].split('/')[-1]) 
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(save_img_path, vis_img)

def convert_pl_to_instances(pl,scale):
    new_pl = {}
    for img_path, content in list(pl.items()):
        if isinstance(content, list):
            #for converting irg predictions
            pl_boxes, pl_classes, pl_scores, img_size_o,gt_classes = content[:5]
        elif isinstance(content, dict):
            #for converting sam predictions
            pl_boxes, pl_classes, pl_scores, img_size_o,gt_classes = content['boxes'], content['categories'], content['scores'], content['img_size'], content['gt_categories']
        else:
            raise Exception(f"{pl[img_path]} is not parsable, only supports list and dicts")
        pl_boxes = Boxes(torch.tensor(pl_boxes) * scale)
        pl_classes = torch.tensor(pl_classes)
        pl_scores = torch.tensor(pl_scores)
        gt_classes = torch.tensor(gt_classes)
        inst = Instances((600, 1200))
        inst.pred_boxes = pl_boxes
        inst.pred_classes = pl_classes
        inst.scores = pl_scores
        new_pl[os.path.basename(img_path)] = inst
    return new_pl

def convert_crowdsam_output(json_content, img_size=(600,1200)):
    """
    转换众包标注输出为结构化字典
    json_content: 众包标注JSON内容列表
    """
    new_json_content = {}
    for item in json_content:
        img_path = item["image_path"]
        
        if img_path not in new_json_content:
            # 初始化图像结构
            new_json_content[img_path] = {
                "boxes": [],
                "categories": [],
                "scores": [],  # 众包标注可能没有分数
                "img_size": img_size,
                'gt_categories':[],
            }
        # 提取标注信息
        category_id = item["category_id"]
        bbox = item["bbox"]
        #XYWH(COCO) -> XYXY
        bbox[2] = bbox[2] + bbox[0]
        bbox[3] = bbox[3] + bbox[1]
        score = item['score']
        # 添加类别和边界框
        new_json_content[img_path]["categories"].append(category_id)
        new_json_content[img_path]["boxes"].append(bbox)
        new_json_content[img_path]["scores"].append(score)
    return convert_pl_to_instances(new_json_content, scale=1.0)

def merge_with_nms(pl_1, pl_2):
    #merge two pseudo label sets
    #note that we should build a union set of keys of each set
    import torchvision.ops as ops
    assert isinstance(pl_1, dict)
    assert isinstance(pl_2, dict)
    keys = set(pl_1.keys( )).union(set(pl_2.keys()))
    new_pl = {}
    for name in keys:
        if name not in pl_1:
            logger.warning(f'{name} not in pl_1')
            continue
        if name not in pl_2:
            logger.warning(f'{name} not in pl_2')
            continue
        inst = Instances.cat([pl_1[name], pl_2[name]])
        #do nms on joined inst
        if len(inst) > 0:
            keep_idx = ops.batched_nms(inst.pred_boxes.tensor, inst.scores, inst.pred_classes, 0.5)
            new_pl[name] = inst[keep_idx]
        else:
            new_pl[name] = inst
    return new_pl  
    
def cache_pl_label(cfg, model, thresh = 0.7, save_path="output.json"):
    #cache initial training pseudo-labels for tuning
    import json
    dataset_name = cfg.DATASETS.TRAIN[0]
    cfg.defrost()
    cfg.SOURCE_FREE.TYPE = False
    cfg.freeze()
    data_loader = build_detection_test_loader(cfg,dataset_name)
    test_metadata = MetadataCatalog.get(dataset_name)
    json_content = {}
    for data in tqdm.tqdm(data_loader):
        data[0]['image_weak'] = data[0]['image']
        with torch.no_grad():
            #Generate Teacher pseudo-labels
            _, teacher_features, teacher_proposals, teacher_results = model(data, mode="train")
        #Process Pseudo labels
        teacher_pseudo_results, num_results_frcnn =  Trainer.process_pseudo_label(teacher_results, thresh, 1.0, "roih", "thresholding")
        if self.sam_pl is not None:
            self.merge_sam_pl(teacher_pseudo_results, data)
        #output_file
        filename = data[0]['file_name']
        pl_boxes = teacher_pseudo_results[0].gt_boxes.tensor.cpu().tolist()
        gt_classes = match_box(pl_boxes, data[0]['instances']).cpu().tolist()
        pl_classes = teacher_pseudo_results[0].gt_classes.cpu().tolist()
        pl_scores  = teacher_pseudo_results[0].scores.cpu().tolist()
        proposal_boxes = teacher_proposals[0].proposal_boxes.tensor.cpu().tolist()
        proposal_scores = teacher_proposals[0].objectness_logits.sigmoid().cpu().tolist()
        #
        assert len(pl_classes) == len(gt_classes) 
        json_content.update({filename:[pl_boxes,
        pl_classes,
        pl_scores,
        data[0]['image'].shape, 
        gt_classes,
        proposal_boxes, 
        proposal_scores]})
    json.dump(json_content, open(save_path, 'w'), ensure_ascii=True)


