import argparse
import random
import copy

from pathlib import Path
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

from engine import *
from build_modules import *
from datasets.augmentations import train_trans, val_trans, strong_trans
from utils import get_rank, init_distributed_mode, resume_and_load, save_ckpt, selective_reinitialize, is_main_process


def get_args_parser(parser):
    # Model Settings
    parser.add_argument('--backbone', default='resnet50', type=str)
    ##
    parser.add_argument('--enable_dino', action='store_true')
    parser.add_argument('--fuse_type', default='add')
    ##
    parser.add_argument('--pos_encoding', default='sine', type=str)
    parser.add_argument('--num_classes', default=9, type=int)
    parser.add_argument('--num_queries', default=300, type=int)
    parser.add_argument('--num_feature_levels', default=4, type=int)
    parser.add_argument('--with_box_refine', action="store_true")
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_encoder_layers', default=6, type=int)
    parser.add_argument('--num_decoder_layers', default=6, type=int)
    parser.add_argument('--feedforward_dim', default=1024, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    # Optimization hyperparameters
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--eval_batch_size', default=1, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj', default=2e-5, type=float)
    parser.add_argument('--sgd', action="store_true")
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--clip_max_norm', default=0.5, type=float, help='gradient clipping max norm')
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--epoch_lr_drop', default=40, type=int)
    # Loss coefficients
    parser.add_argument('--only_class_loss', action="store_true")  # default: False, when define in config: it is True
    parser.add_argument('--high_quality_matches', action="store_true")
    parser.add_argument('--coef_class', default=2.0, type=float)
    parser.add_argument('--coef_boxes', default=5.0, type=float)
    parser.add_argument('--coef_giou', default=2.0, type=float)
    parser.add_argument('--alpha_focal', default=0.25, type=float)
    parser.add_argument('--alpha_ema', default=0.999, type=float)
    # Dataset parameters
    parser.add_argument('--data_root', default='./data', type=str)
    parser.add_argument('--source_dataset', default='cityscapes', type=str)
    parser.add_argument('--target_dataset', default='foggy_cityscapes', type=str)
    # Retraining parameters
    parser.add_argument('--keep_modules', default=["backbone", "encoder"], type=str, nargs="+")  # "decoder"
    # Masking parameters
    parser.add_argument('--block_size', default=64, type=int)
    parser.add_argument('--masked_ratio', default=0.5, type=float)
    parser.add_argument('--coef_masked_img', default=1.0, type=float)
    # Teaching parameters
    parser.add_argument('--dynamic_update', action="store_true")
    parser.add_argument('--fix_update_iter', default=1, type=int)
    parser.add_argument('--max_update_iter', default=5, type=int)
    parser.add_argument('--use_pseudo_label_weights', action="store_true")
    parser.add_argument('--use_loss_student', action="store_true")
    # Dynamic threshold (DT) parameters
    parser.add_argument('--threshold', default=0.3, type=float)
    parser.add_argument('--alpha_dt', default=0.5, type=float)
    parser.add_argument('--gamma_dt', default=0.9, type=float)
    parser.add_argument('--max_dt', default=0.45, type=float)
    # mode settings
    parser.add_argument("--mode", default="single_domain", type=str,
                        help="'single_domain' for single domain training,"
                             "'teaching_standard' for teaching standard process,"
                             "'teaching_mask' for teaching with mask process,"
                             "'eval' for evaluation only.")
    # Other settings
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--random_seed', default=8008, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument('--flush', default=True, type=bool)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--tag", default="", type=str)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_loss(epoch, prefix, total_loss, loss_dict):
    writer.add_scalar(prefix + '/total_loss', total_loss, epoch)
    for k, v in loss_dict.items():
        writer.add_scalar(prefix + '/' + k, v, epoch)


def write_ap50(epoch, prefix, m_ap, ap_per_class, idx_to_class):
    writer.add_scalar(prefix + '/mAP50', m_ap, epoch)
    for idx, num in zip(idx_to_class.keys(), ap_per_class):
        writer.add_scalar(prefix + '/AP50_%s' % (idx_to_class[idx]['name']), num, epoch)


def single_domain_training(model, device):
    # Record the start time
    start_time = time.time()
    # Build dataloaders
    train_loader = build_dataloader(args, args.source_dataset, 'source', 'train', train_trans)
    val_loader = build_dataloader(args, args.target_dataset, 'target', 'val', val_trans)
    idx_to_class = val_loader.dataset.coco.cats
    # Prepare model for optimization
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu])
    criterion = build_criterion(args, device)
    optimizer = build_optimizer(args, model)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epoch_lr_drop)
    # Record the best mAP
    ap50_best = -1.0
    saturate_epoch = 10
    for epoch in range(args.epoch):

        # Set the epoch for the sampler
        if args.distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        # Train for one epoch
        loss_train = train_one_epoch_standard(
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            clip_max_norm=args.clip_max_norm,
            print_freq=args.print_freq,
            flush=args.flush
        )
        # write_loss(epoch, 'single_domain', loss_train)
        lr_scheduler.step()
        # Evaluate
        ap50_per_class, loss_val = evaluate(
            model=model,
            criterion=criterion,
            data_loader_val=val_loader,
            device=device,
            print_freq=args.print_freq,
            flush=args.flush
        )
        # Save the best checkpoint
        map50 = np.asarray([ap for ap in ap50_per_class if ap > -0.001]).mean().tolist()
        if map50 > ap50_best:
            ap50_best = map50
            save_ckpt(model, output_dir/'model_best.pth', args.distributed)
        if epoch == args.epoch - 1:
            save_ckpt(model, output_dir/'model_last.pth', args.distributed)
        # Write the evaluation results to tensorboard
        # write_ap50(epoch, 'single_domain', map50, ap50_per_class, idx_to_class)
    # Record the end time
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Single-domain training finished. Time cost: ' + total_time_str +
          ' . Best mAP50: ' + str(ap50_best), flush=args.flush)
import torch

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
@torch.no_grad()
def infer_model(model, samples):
    # out_list = []
    cls_out_list = []
    box_out_list = []
    for target_teacher_images,target_masks in samples:
        student_out = model(target_teacher_images, target_masks)
        # variance logit
        cls_out = student_out['logit_all'][-1][0, :, 1:].sigmoid()
        box_out = student_out['boxes_all'][-1][0] #xywh
        # out_list.append([cls_out, box_out])
        cls_out_list.append(cls_out)
        box_out_list.append(box_out)
    return torch.stack(cls_out_list, dim=0), torch.stack(box_out_list, dim=0)
def caculate_stability( init_out, new_out):
    from scipy.optimize import linear_sum_assignment
    import torchvision
    from utils.box_utils import box_cxcywh_to_xyxy, generalized_box_iou, box_iou
    cls_out_init, box_out_init = init_out
    box_out_init = [box_cxcywh_to_xyxy(box) for box in box_out_init]
    eps = 1e-6
    cls_new_out, box_new_out = new_out
    box_new_out = [box_cxcywh_to_xyxy(box) for box in box_new_out]

    # if is_main_process():
    #     vis_output(init_out, samples, 'init')
    #     vis_output(new_out, samples, f'test_{model_strip.dino_factor.data}')
    iou_list, cls_list = [], []
    for i in range(len(cls_out_init)):
        if len(box_new_out[i]) == 0 or len(box_out_init[i]) == 0:
            iou_list.append(0.0)
            cls_list.append(0.0)
            continue
        
        # 计算 IoU 和分类相似度
        init_score, init_cls = cls_out_init[i].max(dim=-1)
        new_score, new_cls = cls_new_out[i].max(dim=-1)
        mean_score = (new_score.unsqueeze(1) + init_score.unsqueeze(0)) / 2
        cls_sim = (new_cls.unsqueeze(1) == init_cls.unsqueeze(0)) * mean_score
        bbox_iou = box_iou(box_new_out[i], box_out_init[i])[0] * mean_score
        
        # 匈牙利匹配
        C = - (cls_sim + bbox_iou)
        ind_i, ind_j = linear_sum_assignment(C.cpu())
        z = mean_score[ind_i, ind_j].sum() + 1e-6
        iou_list.append(bbox_iou[ind_i, ind_j].sum()/ z)
        cls_list.append(cls_sim[ind_i, ind_j].sum()/z)

    # 综合一致性
    consist_cls_pred = torch.mean(torch.stack(cls_list))
    consist_iou_pred = torch.mean(torch.stack(iou_list))
    consist_pred = torch.sqrt(consist_iou_pred * consist_cls_pred + eps)
    
    print(f"Consistency: IoU={consist_iou_pred:.4f}, Cls={consist_cls_pred:.4f}")
    return consist_pred
def binary_search(model, samples, init_out, stability_target=0.9, search_range=[0,1], iter_num = 5, eps=1e-6):
    # 提前转换边界框格式
    # 动态调整
    start_pos, end_pos = search_range
    old_factor = model.module.dino_factor.data
    for i in range(iter_num):
        model.module.dino_factor.data = torch.tensor(start_pos + end_pos).cuda() /2
        new_out = infer_model(model, samples)
        consist_pred = caculate_stability(init_out, new_out)
        if consist_pred > stability_target:
            start_pos = (start_pos + end_pos) / 2
        else:
            end_pos = (start_pos + end_pos) / 2
        print(f"Iter {i} Factor={model.module.dino_factor.data:.4f}, Step={model.module.dino_step:.4f}")
    model.module.dino_factor.data = old_factor
    return (start_pos + end_pos) / 2
def self_consistency_update(model, samples, consistency_thresh=0.9, scale=1.1):
    new_out = infer_model(model, samples)
    old_dino_factor = model.module.dino_factor.data
    model.module.dino_factor.data.zero_()
    init_out = infer_model(model, samples)
    stability_score = caculate_stability(init_out, new_out)
    if stability_score > consistency_thresh:
        return  old_dino_factor * scale
    return old_dino_factor
    # else:
    # model.module.dino_factor.data = old_dino_factor / scale
def vis_output(model_out, samples, prefix=""):
    import os
    import torch
    from detectron2.structures import Instances
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog,Metadata

    save_dir = 'vis_output'
    os.makedirs(save_dir, exist_ok=True)    
    metadata = MetadataCatalog.get("cityscape_2007_train_s")  # 注意：Cityscapes的注册名是"cityscapes"（小写）
 # ImageNet归一化参数
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])
    metadata  = Metadata(thing_classes=['person', 'car', 'train', 'rider', 'truck', 'motorcycle', 'bicycle', 'bus'])
    for idx, (target_teacher_images, target_masks) in enumerate(samples):
        with torch.no_grad():
            # 获取图像尺寸（假设输入是 [B, C, H, W]）
            height, width = target_teacher_images.shape[-2:]
            instances = Instances(image_size=(height, width))
            # 提取预测框、类别和置信度
            out_scores, out_boxes = model_out[0][idx], model_out[1][idx]
            pred_boxes = out_boxes.cpu() * torch.tensor([[width,height,width,height]])  # [N, 4]
            #convert from cxcywh to xyxy
            pred_boxes = torch.cat([pred_boxes[:,:2] - pred_boxes[:,2:]/2,  pred_boxes[:,:2] + pred_boxes[:,2:]/2], dim=1) 
            pred_logits = out_scores.cpu()  # [N, num_classes]
            pred_scores, pred_classes = pred_logits.max(dim=-1)  # [N], [N]
            # 填充Instances对象
            instances.pred_boxes = pred_boxes[pred_scores>0.3]
            instances.scores = pred_scores[pred_scores>0.3]
            instances.pred_classes = pred_classes[pred_scores>0.3]
            # 可视化（假设图像是 [0,1] 归一化的）
            image_np = target_teacher_images[0].cpu().numpy()  # [C, H, W]
            image_np = np.transpose(image_np, (1, 2, 0))  # -> [H, W, C]
            image_np = image_np * IMAGENET_STD.numpy() + IMAGENET_MEAN.numpy()  # 反归一化
            image_np = np.clip(image_np * 255, 0, 255).astype("uint8")
            #这块反归一化有点问题，因为图片是根据Image Net pretrain参数归一化的 
            vis = Visualizer(image_np, metadata=metadata, scale=1.0)
            vis_output = vis.draw_instance_predictions(instances.to(torch.device('cpu')))
            # 保存结果
            output_path = os.path.join(save_dir, f"{prefix}_pred_{idx}.png")
            vis_output.save(output_path)

def sample_random_samples(dataloader, device, sample_num=10):
    target_fetcher = DataPreFetcher(dataloader,  device=device)
    samples = []
    i = 0
    sample_idx = np.random.choice(np.arange(len(dataloader)), sample_num, replace=False)
    for i in range(len(dataloader)):
        if i not in sample_idx:
            continue
        images, masks, _  = target_fetcher.next()
        samples.append([images[1], masks])
    return samples

def js_divergence(p, q, eps=1e-10):
    from scipy.stats import entropy
    p = p.cpu().numpy()
    q = q.cpu().numpy()
    p = np.clip(p, eps, 1.0)  # 避免log(0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m, axis=-1) + entropy(q, m, axis=-1)).mean()

# Teaching
def teaching(model_stu, device):
    start_time = time.time()
    # Build dataloaders
    target_loader = build_dataloader_teaching(args, args.target_dataset, 'target', 'train')
    val_loader = build_dataloader(args, args.target_dataset, 'target', 'val', val_trans)
    idx_to_class = val_loader.dataset.coco.cats
    # Build teacher model
    model_tch = build_teacher(args, model_stu, device)
    # Build init student model
    init_model_stu = build_teacher(args, model_stu, device)
    # print(init_model_stu.keys())
    # Prepare model for optimization
    if args.distributed:
        model_stu = DistributedDataParallel(model_stu, device_ids=[args.gpu], find_unused_parameters=False)
        model_tch = DistributedDataParallel(model_tch, device_ids=[args.gpu])
        init_model_stu = DistributedDataParallel(init_model_stu, device_ids=[args.gpu])
    # Build criterion, optimizer and lr_scheduler
    criterion = build_criterion(args, device)
    criterion_pseudo = build_criterion(args, device)
    criterion_pseudo_weak = build_criterion(args, device, only_class_loss=args.only_class_loss)
    optimizer = build_optimizer(args, model_stu)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epoch_lr_drop)
    # Initialize thresholds
    thresholds = [args.threshold] * args.num_classes
    # Record the best mAP
    ap50_best = -1.0

    # Initialize buffers
    stu_buffer_cost = []
    stu_buffer_img = []
    stu_buffer_mask = []
    res_dict = {'stu_ori': [], 'stu_now': [], 'update_iter': []}

    # Initialize masking
    masking = Masking(block_size=args.block_size, masked_ratio=args.masked_ratio)
    test_samples = sample_random_samples(target_loader, device)
    #Initialize dino factors
    # if args.enable_dino:
        # model_stu.module.dino_factor.data = 0.0
        # model_tch.module.dino_factor.data = 0.0
        # model_stu.module.dino_step = 0.05
        # model_tch.module.dino_step = 0.05

    import math
    if args.enable_dino:
        pass
    # init_out = infer_model(model_tch, test_samples)
    # stab_factor = binary_search(model_tch, test_samples, init_out, stability_target=0.7)
    for epoch in range(args.epoch):
        if epoch == 1:
            pass
            #fast updating teacher parameter
            # model_tch.module.load_state_dict(model_stu.module.state_dict())
        print('dynamic updating teacher dino weight')
        if args.enable_dino:
            model_tch.module.dino_factor.data = torch.tensor(math.sin(epoch/10 * math.pi/2) * 0.5).cuda()
            model_stu.module.dino_factor.data = model_tch.module.dino_factor.data

        #adaptive adjusting teacher dino factor
        # dino_factor =self_consistency_update(model_tch, test_samples)
        # model_tch.module.dino_factor.data = dino_factor

        #keep student same with teacher
        if is_main_process() and args.enable_dino:
            print('tch dino factor', model_stu.module.dino_factor.data)
            print('stu dino factor', model_tch.module.dino_factor.data)
        # Set the epoch for the sampler
        if args.distributed and hasattr(target_loader.sampler, 'set_epoch'):
            target_loader.sampler.set_epoch(epoch)
        if args.mode == "teaching_mask":
            loss_train, loss_target_dict = train_one_epoch_teaching_mask(
                student_model=model_stu,
                teacher_model=model_tch,
                init_student_model=init_model_stu,
                criterion_pseudo=criterion_pseudo,
                criterion_pseudo_weak=criterion_pseudo_weak,
                target_loader=target_loader,
                optimizer=optimizer,
                thresholds=thresholds,
                coef_masked_img=args.coef_masked_img,
                alpha_ema=args.alpha_ema,
                device=device,
                epoch=epoch,
                keep_modules=args.keep_modules,
                clip_max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                masking=masking,
                flush=args.flush,
                fix_update_iter=args.fix_update_iter,
                max_update_iter=args.max_update_iter,
                dynamic_update=args.dynamic_update,
                stu_buffer_cost=stu_buffer_cost,
                stu_buffer_img=stu_buffer_img,
                stu_buffer_mask=stu_buffer_mask,
                res_dict=res_dict,
                use_pseudo_label_weights=args.use_pseudo_label_weights,
                use_loss_student=args.use_loss_student
            )
        elif args.mode == "teaching_standard":
            loss_train, loss_target_dict = train_one_epoch_teaching_standard(
                student_model=model_stu,
                teacher_model=model_tch,
                criterion_pseudo=criterion_pseudo,
                target_loader=target_loader,
                optimizer=optimizer,
                thresholds=thresholds,
                alpha_ema=args.alpha_ema,
                device=device,
                epoch=epoch,
                clip_max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                flush=args.flush,
                fix_update_iter=args.fix_update_iter,
                test_samples = test_samples, 
            )
        else:
            raise ValueError('Invalid mode: ' + args.mode)

        # Renew thresholds
        # thresholds = criterion.dynamic_threshold(thresholds)
        # criterion.clear_positive_logits()
        # Write the losses to tensorboard
        if is_main_process():
            write_loss(epoch, 'teaching_target', loss_train, loss_target_dict)
        lr_scheduler.step()

        # Evaluate teacher and student model
        ap50_per_class_teacher, loss_val_teacher = evaluate(
            model=model_tch,
            criterion=criterion,
            data_loader_val=val_loader,
            device=device,
            print_freq=args.print_freq,
            flush=args.flush
        )
        ap50_per_class_student, loss_val_student = evaluate(
            model=model_stu,
            criterion=criterion,
            data_loader_val=val_loader,
            device=device,
            print_freq=args.print_freq,
            flush=args.flush
        )
        if is_main_process():
            # Save the best checkpoint
            map50_tch = np.asarray([ap for ap in ap50_per_class_teacher if ap > -0.001]).mean().tolist()
            map50_stu = np.asarray([ap for ap in ap50_per_class_student if ap > -0.001]).mean().tolist()
            print('eval teacher')
            write_ap50(epoch, 'teaching_teacher', map50_tch, ap50_per_class_teacher, idx_to_class)
            print('eval stdent')
            write_ap50(epoch, 'teaching_student', map50_stu, ap50_per_class_student, idx_to_class)
            # if max(map50_tch, map50_stu) > ap50_best:
            #     ap50_best = max(map50_tch, map50_stu)
            #     save_ckpt(model_tch if map50_tch > map50_stu else model_stu, output_dir/'model_best.pth', args.distributed)
            if map50_tch > ap50_best:
                ap50_best = map50_tch
                save_ckpt(model_tch, output_dir/'model_best.pth', args.distributed)
            if epoch == args.epoch - 1:
                save_ckpt(model_tch, output_dir/'model_last_tch.pth', args.distributed)
                save_ckpt(model_stu, output_dir/'model_last_stu.pth', args.distributed)
        # if (epoch+1) % 5 == 0:
        #     save_ckpt(model_tch, output_dir/f'tch_epoch{epoch:02}.pth', args.distributed)
        #     save_ckpt(model_stu, output_dir/f'stu_epoch{epoch:02}.pth', args.distributed)
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Teaching finished. Time cost: ' + total_time_str + ' . Best mAP50: ' + str(ap50_best), flush=args.flush)

# Evaluate only
def eval_only(model, device):
    if args.distributed:
        Warning('Evaluation with distributed mode may cause error in output result labels.')
    criterion = build_criterion(args, device)
    # Eval source or target dataset
    val_loader = build_dataloader(args, args.target_dataset, 'target', 'val', val_trans)
    ap50_per_class, epoch_loss_val, coco_data = evaluate(
        model=model,
        criterion=criterion,
        data_loader_val=val_loader,
        output_result_labels=True,
        device=device,
        print_freq=args.print_freq,
        flush=args.flush
    )
    print('Evaluation finished. mAPs: ' + str(ap50_per_class) + '. Evaluation loss: ' + str(epoch_loss_val))
    output_file = output_dir/'evaluation_result_labels.json'
    print("Writing evaluation result labels to " + str(output_file))
    with open(output_file, 'w', encoding='utf-8') as fp:
        json.dump(coco_data, fp)


def main():
    # Initialize distributed mode
    init_distributed_mode(args)
    # Set random seed
    if args.random_seed is None:
        args.random_seed = random.randint(1, 10000)
    set_random_seed(args.random_seed + get_rank())
    # Print args
    print('-------------------------------------', flush=args.flush)
    print('Logs will be written to ' + str(logs_dir))
    print('Checkpoints will be saved to ' + str(output_dir))
    print('-------------------------------------', flush=args.flush)
    for key, value in args.__dict__.items():
        print(key, value, flush=args.flush)
    # Build model
    device = torch.device(args.device)
    model = build_model(args, device)
    if args.resume != "":
        model = resume_and_load(model, args.resume, device)
    # Training or evaluation
    print('-------------------------------------', flush=args.flush)
    if args.mode == "single_domain":
        single_domain_training(model, device)
    elif args.mode == "teaching_standard" or args.mode == "teaching_mask":
        teaching(model, device)
    elif args.mode == 'teaching_coadaptation':
        teaching_coadaptation(model, device)
    elif args.mode == "eval":
        eval_only(model, device)
    else:
        raise ValueError('Invalid mode: ' + args.mode)


if __name__ == '__main__':
    # Parse arguments
    parser_main = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    get_args_parser(parser_main)
    args = parser_main.parse_args()
    # Set output directory
    if args.tag != "":
        output_dir = os.path.join(args.output_dir, args.tag)
    else:
        output_dir = Path(args.output_dir)
    logs_dir = output_dir/'data_logs'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(logs_dir))
    # Call main function
    main()