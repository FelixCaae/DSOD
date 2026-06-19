import time
import datetime
import json

import copy

import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader

from datasets.coco_style_dataset import DataPreFetcher
from datasets.coco_eval import CocoEvaluator

from models.criterion import post_process, merge_pseudo_labels, get_pseudo_labels, get_topk_outputs, SetCriterion
from utils.distributed_utils import is_main_process,is_dist_avail_and_initialized
from utils.box_utils import box_cxcywh_to_xyxy, convert_to_xywh
from collections import defaultdict
from typing import List

from datasets.masking import Masking
from scipy.optimize import linear_sum_assignment
from utils.box_utils import box_cxcywh_to_xyxy, generalized_box_iou
from utils import selective_reinitialize
import torch.distributed as dist
def train_one_epoch_standard(model: torch.nn.Module,
                             criterion: torch.nn.Module,
                             data_loader: DataLoader,
                             optimizer: torch.optim.Optimizer,
                             device: torch.device,
                             epoch: int,
                             clip_max_norm: float = 0.0,
                             print_freq: int = 20,
                             start_iter = 0,
                             flush: bool = True):
    """
    Train the standard detection model, using only labelled training set source.
    """
    start_time = time.time()
    model.train()
    criterion.train()
    fetcher = DataPreFetcher(data_loader, device=device)
    images, masks, annotations = fetcher.next()
    # Training statistics
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)
    epoch_loss_dict = defaultdict(float)
    for i in range(len(data_loader)):
        # Forward
        out = model(images, masks)
        # Loss
        loss, loss_dict = criterion(out, annotations)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        # Record loss
        epoch_loss += loss.detach()
        for k, v in loss_dict.items():
            epoch_loss_dict[k] += v.detach().cpu().item()
        # Data pre-fetch
        images, masks, annotations = fetcher.next()
        # Log
        if is_main_process() and (i + 1) % print_freq == 0:
            print('Training epoch ' + str(epoch) + ' : [ ' + str(i + 1) + '/' + str(len(data_loader)) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)
    # Final process of training statistic
    epoch_loss /= len(data_loader)
    for k, v in epoch_loss_dict.items():
        epoch_loss_dict[k] /= len(data_loader)
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Training epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_loss_dict



def train_one_epoch_teaching_standard(student_model: torch.nn.Module,
                                      teacher_model: torch.nn.Module,
                                      criterion_pseudo: torch.nn.Module,
                                      target_loader: DataLoader,
                                      optimizer: torch.optim.Optimizer,
                                      thresholds: List[float],
                                      alpha_ema: float,
                                      device: torch.device,
                                      epoch:int,
                                      clip_max_norm: float = 0.0,
                                      print_freq: int = 20,
                                      flush: bool = True,
                                      fix_update_iter: int = 1,
                                      test_samples = [],
                                      static_teacher_model = None,
                                      use_extra_pseudo_label = False,
                                      smooth_dino_factor = False,
                                      distill_dino_features = False,
                                      distill_weight_alpha = 0.5,
                                      distill_weight_beta = 1.0,
                                      debug=False,
                                      ):
    """
    Train the student model with the teacher model, using only unlabeled training set target .
    """
    start_time = time.time()
    student_model.train()
    teacher_model.train()
    criterion_pseudo.train()
    target_fetcher = DataPreFetcher(target_loader, device=device)
    target_images, target_masks, target_labels = target_fetcher.next()
    target_teacher_images, target_student_images = target_images[0], target_images[1]
    # Record epoch losses
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)

    # Training data statistics
    epoch_target_loss_dict = defaultdict(float)
    total_iters = len(target_loader)
    import math
    if smooth_dino_factor and teacher_model.module.VFM_backbone is not None:
        target_dino_factor = float(teacher_model.module.target_dino_factor)
        acc_iter = epoch * total_iters
        warm_up_iters = total_iters
        print(epoch, acc_iter, target_dino_factor)
    for iter in range(total_iters):
        if smooth_dino_factor:
            new_w = min(math.sqrt((iter+acc_iter)/warm_up_iters), 1) * target_dino_factor
            teacher_model.module.dino_factor.data = torch.tensor(new_w).cuda()
            student_model.module.dino_factor.data = torch.tensor(new_w).cuda()
            
        # from models.vis_tools import visualize_grad_cam
        # visualize_grad_cam(teacher_model,  target_teacher_images, target_masks, target_layer_idx=0)
        # visualize_grad_cam(teacher_model,  target_teacher_images, target_masks, target_layer_idx=1)
        # visualize_grad_cam(teacher_model,  target_teacher_images, target_masks, target_layer_idx=2)

        # Target teacher forward
        # progressive updating weight factor
        # alpha = 0.
        import math
        from models.vis_tools import vis_output

        with torch.no_grad():
            teacher_out = teacher_model(target_teacher_images, target_masks)
            pseudo_labels = get_pseudo_labels(teacher_out['logit_all'][-1], teacher_out['boxes_all'][-1], thresholds, is_nms=True)
            if static_teacher_model is not None:
                static_teacher_out = static_teacher_model(target_teacher_images, target_masks)
                static_pseudo_labels = get_pseudo_labels(static_teacher_out['logit_all'][-1], static_teacher_out['boxes_all'][-1], thresholds)
                # pseudo_labels = merge_pseudo_labels(pseudo_labels, static_pseudo_labels)
                if torch.rand(1) < 0.001:
                    vis_output(pseudo_labels, target_images[0], 'test_pseudo_gt.jpg', id_mapping=lambda x:x-1)
                    vis_output(static_pseudo_labels, target_images[0], 'test_static_pseudo_gt.jpg', id_mapping=lambda x:x-1)
                dino_features = static_teacher_out['dino_features']
            else:
                dino_features = None
        # Target student forward
            # vis_output(pseudo_labels, target_images[0], 'test_pseudo_gt.jpg')
            # import pdb;pdb.set_trace()
            if use_extra_pseudo_label:
                target_labels[0]['scores'] = torch.ones_like(target_labels[0]['labels']).float() * 0.35
                target_labels[0]['labels'] = target_labels[0]['labels']
                pseudo_labels = merge_pseudo_labels(pseudo_labels, target_labels, iou_threshold = 0.5)
                if torch.rand(1) < 0.01 :
                    vis_output(target_labels, target_images[0], 'test_pseudo_gt.jpg')
                    vis_output(pseudo_labels, target_images[0], 'test_full_pseudo_gt.jpg')

        # Target student forward
        
        target_student_out = student_model(target_student_images, target_masks, dino_features=dino_features)
        target_loss, target_loss_dict = criterion_pseudo(target_student_out, pseudo_labels)
        if static_teacher_model is not None:
            static_target_loss, static_target_loss_dict = criterion_pseudo(target_student_out, static_pseudo_labels)
            static_logit_loss, static_logit_distill_loss_dict = criterion_pseudo.forward_logits(target_student_out, static_teacher_out)
            target_loss_dict.update(static_logit_distill_loss_dict)
            target_loss_dict.update({'static_' + k:v for k,v in static_target_loss_dict.items()})
        else:
            static_target_loss  = 0.0
            static_logit_loss = 0.0
            
        loss = target_loss  + static_target_loss * distill_weight_alpha + static_logit_loss * distill_weight_beta

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip_max_norm)
        optimizer.step()

        # Record epoch losses
        epoch_loss += loss.detach()

        # update loss_dict
        for k, v in target_loss_dict.items():
            epoch_target_loss_dict[k] += v.detach().cpu().item()

        if iter % fix_update_iter == 0:
            with torch.no_grad():
                state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
                for key, value in state_dict.items():
                    state_dict[key] = alpha_ema * value + (1 - alpha_ema) * student_state_dict[key].detach()
                teacher_model.load_state_dict(state_dict)

        # Data pre-fetch
        target_images, target_masks, target_labels = target_fetcher.next()
        if target_images is not None:
            target_teacher_images, target_student_images = target_images[0], target_images[1]
        if debug:
            from myutils import get_gpu_memory_usage
            get_gpu_memory_usage()
        # Log
        if is_main_process() and (iter + 1) % print_freq == 0:
            str_out = ""
            for k,v in  target_loss_dict.items():
                str_out+= f"{k}:{float(v)} "
            
            print('Teaching epoch ' + str(epoch) + ' : [ ' + str(iter + 1) + '/' + str(total_iters) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()) + str_out, flush=flush)
                
    # Final process of loss dict
    epoch_loss /= total_iters
    for k, v in epoch_target_loss_dict.items():
        epoch_target_loss_dict[k] /= total_iters
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Teaching epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_target_loss_dict


def train_one_epoch_teaching_distill_standard(student_model: torch.nn.Module,
                                      teacher_model: torch.nn.Module,
                                      static_teacher_model,
                                      criterion_pseudo: torch.nn.Module,
                                      target_loader: DataLoader,
                                      optimizer: torch.optim.Optimizer,
                                      thresholds: List[float],
                                      alpha_ema: float,
                                      device: torch.device,
                                      epoch: int,
                                      clip_max_norm: float = 0.0,
                                      print_freq: int = 20,
                                      flush: bool = True,
                                      fix_update_iter: int = 1,
                                      test_samples = [],
                                      smooth_dino_factor: bool = False,
                                      acc_iter: int = 0,
                                      warm_up_iters: int = 1,
                                      target_dino_factor: float = 1.0
                                      ):
    """
    Train the student model with the teacher model, using only unlabeled training set target .
    """
    start_time = time.time()
    student_model.train()
    teacher_model.train()
    criterion_pseudo.train()
    target_fetcher = DataPreFetcher(target_loader, device=device)
    target_images, target_masks, _ = target_fetcher.next()
    target_teacher_images, target_student_images = target_images[0], target_images[1]
    # Record epoch losses
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)

    # Training data statistics
    epoch_target_loss_dict = defaultdict(float)
    total_iters = len(target_loader)


    for iter in range(total_iters):
        if smooth_dino_factor:
            new_w = min(math.sqrt((iter+acc_iter)/warm_up_iters), 1) * target_dino_factor
            teacher_model.module.dino_factor.data = torch.tensor(new_w).cuda()
            student_model.module.dino_factor.data = torch.tensor(new_w).cuda()
            
        # Target teacher forward
        # progressive updating weight factor
        # alpha = 0.
        import math
        with torch.no_grad():
            teacher_out = teacher_model(target_teacher_images, target_masks)
            pseudo_labels = get_pseudo_labels(teacher_out['logit_all'][-1], teacher_out['boxes_all'][-1], thresholds)
            # static_teacher_out = static_teacher_model(target_teacher_images, target_masks)
            # static_pseudo_labels = get_pseudo_labels(static_teacher_out['logit_all'][-1], static_teacher_out['boxes_all'][-1], thresholds)
            # pseudo_labels = merge_pseudo_labels(pseudo_labels, static_pseudo_labels)
        # Target student forward
        #Usining Teacher images
        target_student_out = student_model(target_teacher_images, target_masks)
        target_student_out['teacher_features'] = teacher_out['features']
        target_student_out['teacher_logit_all'] = teacher_out['logit_all']
        target_student_out['teacher_boxes_all'] = teacher_out['boxes_all']
        target_loss, target_loss_dict = criterion_pseudo(target_student_out, pseudo_labels)

        loss = target_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip_max_norm)
        optimizer.step()

        # Record epoch losses
        epoch_loss += loss.detach()

        # update loss_dict
        for k, v in target_loss_dict.items():
            epoch_target_loss_dict[k] += v.detach().cpu().item()


        if iter % fix_update_iter == 0:
            with torch.no_grad():
                state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
                for key, value in state_dict.items():
                    state_dict[key] = alpha_ema * value + (1 - alpha_ema) * student_state_dict[key].detach()
                teacher_model.load_state_dict(state_dict)

        # Data pre-fetch
        target_images, target_masks, _ = target_fetcher.next()
        if target_images is not None:
            target_teacher_images, target_student_images = target_images[0], target_images[1]

        # Log
        if is_main_process() and (iter + 1) % print_freq == 0:
            str_out = ""
            for k,v in  target_loss_dict.items():
                str_out+= f"{k}:{float(v)} "
            
            print('Teaching epoch ' + str(epoch) + ' : [ ' + str(iter + 1) + '/' + str(total_iters) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()) + str_out, flush=flush)
                
    # Final process of loss dict
    epoch_loss /= total_iters
    for k, v in epoch_target_loss_dict.items():
        epoch_target_loss_dict[k] /= total_iters
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Teaching epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_target_loss_dict


def train_one_epoch_teaching_standard_with_piece(student_model: torch.nn.Module,
                                      teacher_model: torch.nn.Module,
                                      criterion_pseudo: torch.nn.Module,
                                      target_loader: DataLoader,
                                      optimizer: torch.optim.Optimizer,
                                      thresholds: List[float],
                                      alpha_ema: float,
                                      device: torch.device,
                                      epoch: int,
                                      clip_max_norm: float = 0.0,
                                      print_freq: int = 20,
                                      flush: bool = True,
                                      fix_update_iter: int = 1,
                                      test_samples = [],
                                      piece_num = 10,
                                      piece_id = 0,
                                      target_fetcher= None,
                                      ):
    """
    Train the student model with the teacher model, using only unlabeled training set target .
    """
    start_time = time.time()
    student_model.train()
    teacher_model.train()
    criterion_pseudo.train()
    target_fetcher = DataPreFetcher(target_loader, device=device)
    target_images, target_masks, _ = target_fetcher.next()
    target_teacher_images, target_student_images = target_images[0], target_images[1]
    # Record epoch losses
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)

    # Training data statistics
    epoch_target_loss_dict = defaultdict(float)
    total_iters = len(target_loader) // piece_num

    for iter in range(total_iters):
        # Target teacher forward
        # progressive updating weight factor
        # alpha = 0.
        if target_images is None or target_masks is None:
            total_iters = iter
            break
        import math
        with torch.no_grad():
            teacher_out = teacher_model(target_teacher_images, target_masks)
            pseudo_labels = get_pseudo_labels(teacher_out['logit_all'][-1], teacher_out['boxes_all'][-1], thresholds)

        # Target student forward
        target_student_out = student_model(target_student_images, target_masks)
        target_loss, target_loss_dict = criterion_pseudo(target_student_out, pseudo_labels)

        loss = target_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip_max_norm)
        optimizer.step()

        # Record epoch losses
        epoch_loss += loss.detach()

        # update loss_dict
        for k, v in target_loss_dict.items():
            epoch_target_loss_dict[k] += v.detach().cpu().item()
        if iter % fix_update_iter == 0:
            with torch.no_grad():
                state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
                for key, value in state_dict.items():
                    state_dict[key] = alpha_ema * value + (1 - alpha_ema) * student_state_dict[key].detach()
                teacher_model.load_state_dict(state_dict)

        # Data pre-fetch
        target_images, target_masks, _ = target_fetcher.next()
        if target_images is not None:
            target_teacher_images, target_student_images = target_images[0], target_images[1]

        # Log
        if is_main_process() and (iter + 1) % print_freq == 0:
            str_out = ""
            for k,v in  target_loss_dict.items():
                str_out+= f"{k}:{float(v)} "
            
            print('Teaching epoch ' + str(epoch) + ' : [ ' + str(iter + 1) + '/' + str(total_iters) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()) + str_out, flush=flush)
                
    # Final process of loss dict
    epoch_loss /= total_iters
    for k, v in epoch_target_loss_dict.items():
        epoch_target_loss_dict[k] /= total_iters
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Teaching epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_target_loss_dict,target_fetcher


def train_one_epoch_teaching_mask(student_model: torch.nn.Module,
                                  teacher_model: torch.nn.Module,
                                  init_student_model: torch.nn.Module,
                                  criterion_pseudo: torch.nn.Module,
                                  criterion_pseudo_weak: torch.nn.Module,
                                  target_loader: DataLoader,
                                  optimizer: torch.optim.Optimizer,
                                  thresholds: List[float],
                                  coef_masked_img: float,
                                  alpha_ema: float,
                                  device: torch.device,
                                  epoch: int,
                                  keep_modules: List[str],
                                  clip_max_norm: float = 0.0,
                                  smooth_dino_factor = False,
                                  print_freq: int = 20,
                                  masking: Masking = None,
                                  flush: bool = True,
                                  fix_update_iter: int = 1,
                                  max_update_iter: int = 5,
                                  dynamic_update: bool = False,
                                  stu_buffer_cost: List[float] = None,
                                  stu_buffer_img: List[torch.Tensor] = None,
                                  stu_buffer_mask: List[torch.Tensor] = None,
                                  res_dict: dict = None,
                                  use_pseudo_label_weights: bool = False,
                                  static_teacher_model  =None,
                                  merge_weights=[1,1],
                                  distill_weight_alpha = 0.5,
                                  distill_weight_beta = 0.5,
                                  use_loss_student: bool = False):
    """
    Train the student model with the teacher model, using only unlabeled training set target (plus masked target image)
    """
    start_time = time.time()
    student_model.train()
    teacher_model.train()
    init_student_model.train()
    criterion_pseudo.train()
    criterion_pseudo_weak.train()
    target_fetcher = DataPreFetcher(target_loader, device=device)
    target_images, target_masks, _ = target_fetcher.next()
    target_teacher_images, target_student_images = target_images[0], target_images[1]
    # Record epoch losses
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)


    # Training data statistics
    epoch_target_loss_dict = defaultdict(float)
    total_iters = len(target_loader)
    # # Initialize buffers
    # stu_buffer_cost = []
    # stu_buffer_img = []
    # stu_buffer_mask = []
    target_dino_factor = float(teacher_model.module.target_dino_factor)
    acc_iter = epoch * total_iters
    warm_up_iters = total_iters
    import math
    for iter in range(total_iters):
        # Target teacher forward
        # progressive updating weight factor
        if smooth_dino_factor:
            new_w = min(math.sqrt((iter+acc_iter)/warm_up_iters), 1) * target_dino_factor
            teacher_model.module.dino_factor.data = torch.tensor(new_w).cuda()
            student_model.module.dino_factor.data = torch.tensor(new_w).cuda()
            init_student_model.module.dino_factor.data = torch.tensor(new_w).cuda()
        with torch.no_grad():
            teacher_out = teacher_model(target_teacher_images, target_masks)
            pseudo_labels = get_pseudo_labels(teacher_out['logit_all'][-1], teacher_out['boxes_all'][-1], thresholds)
            if static_teacher_model is not None:
                static_teacher_out = static_teacher_model(target_teacher_images, target_masks)
                static_pseudo_labels = get_pseudo_labels(static_teacher_out['logit_all'][-1], static_teacher_out['boxes_all'][-1], thresholds)
                # pseudo_labels = merge_pseudo_labels(pseudo_labels, static_pseudo_labels, weights=merge_weights)
        # Target student forward
        target_student_out = student_model(target_student_images, target_masks)
        # loss from pseudo labels of current teacher
        target_loss, target_loss_dict = criterion_pseudo(target_student_out, pseudo_labels)

        # Masked target student forward
        masked_target_images = masking(target_student_images)
        masked_target_student_out = student_model(masked_target_images, target_masks)
        # loss from pseudo labels of current teacher
        masked_target_loss, masked_target_loss_dict = criterion_pseudo(masked_target_student_out, pseudo_labels, mask_loss=True)
        if static_teacher_model is not None:
            static_target_loss, static_target_loss_dict = criterion_pseudo(target_student_out, static_pseudo_labels)
            static_logit_loss, static_logit_distill_loss_dict = criterion_pseudo.forward_logits(target_student_out, static_teacher_out)
            target_loss_dict.update(static_logit_distill_loss_dict)
            target_loss_dict.update({'static_' + k:v for k,v in static_target_loss_dict.items()})
        else:
            static_target_loss  = 0.0
            static_logit_loss = 0.0
            
        loss = target_loss  + static_target_loss * distill_weight_alpha + static_logit_loss * distill_weight_beta
        # loss = target_loss  + static_target_loss * distill_weight
        # Final loss
        loss = loss + coef_masked_img * masked_target_loss 

        # Loss from pseudo labels of previous student (just testing, not used)
        # if use_loss_student:
        #     # Loss from pseudo labels of previous student
        #     with torch.no_grad():
        #         student_out = student_model(target_teacher_images, target_masks)
        #         pseudo_labels_student = get_pseudo_labels(student_out['logit_all'][-1], student_out['boxes_all'][-1],
        #                                                   thresholds)
        #     target_loss_student, target_loss_dict_student = criterion_pseudo_weak(target_student_out,
        #                                                                         pseudo_labels_student, use_pseudo_label_weights)
        #     masked_target_loss_student, masked_target_loss_dict_student = criterion_pseudo_weak(masked_target_student_out,
        #                                                                                       pseudo_labels_student, use_pseudo_label_weights)
        #
        #     # Final loss
        #     loss_student = target_loss_student + coef_masked_img * masked_target_loss_student
        #     loss += loss_student

        # Dynamic update EMA teacher : Create buffer cost and buffer image in student model
        if dynamic_update:
            with torch.no_grad():
                student_out = student_model(target_teacher_images, target_masks)
            # variance logit
            student_out_var = student_out['logit_all'].var(dim=0)
            var_total = student_out_var.mean()#.item()
            if is_dist_avail_and_initialized():
                if not isinstance(var_total, torch.Tensor):
                    raise TypeError(f"Expected tensor, got {type(var_total)}")
                dist.all_reduce(var_total, op=dist.ReduceOp.SUM)
                var_total /= dist.get_world_size()
            stu_buffer_cost.append(var_total.item())

            # Store batch data to buffer
            stu_buffer_img.append(target_teacher_images.clone().detach())
            stu_buffer_mask.append(target_masks.clone().detach())

            if len(stu_buffer_cost) == 1:
                with torch.no_grad():
                    init_student_model.load_state_dict(student_model.state_dict())

            if len(stu_buffer_cost) >= 1:
                with torch.no_grad():
                    init_student_out = init_student_model(target_teacher_images, target_masks)
                    pseudo_labels_init_student = get_pseudo_labels(init_student_out['logit_all'][-1], init_student_out['boxes_all'][-1],
                                                              thresholds)
                # Loss from pseudo labels of init student
                init_student_loss, init_student_loss_dict = criterion_pseudo_weak(target_student_out,
                                                                                    pseudo_labels_init_student, use_pseudo_label_weights)
                masked_init_student_loss, masked_init_student_loss_dict = criterion_pseudo_weak(masked_target_student_out,
                                                                                                  pseudo_labels_init_student, use_pseudo_label_weights, mask_loss=True)
                loss_init_student = init_student_loss + coef_masked_img * masked_init_student_loss
                loss += loss_init_student

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip_max_norm)
        optimizer.step()

        # Record epoch losses
        epoch_loss += loss.detach()

        # update loss_dict
        for k, v in target_loss_dict.items():
            epoch_target_loss_dict[k] += v.detach().cpu().item()

        # Dynamic update EMA teacher : Update weight of teacher model
        if dynamic_update:
            if len(stu_buffer_cost) < max_update_iter:
                all_score = eval_stu(student_model, stu_buffer_img, stu_buffer_mask)
                compare_score = np.array(all_score) - np.array(stu_buffer_cost)
                # print(len(stu_buffer_cost), len(all_score), np.mean(compare_score<0))
                if np.mean(compare_score < 0) >= 0.5:

                    if is_main_process():
                        res_dict['stu_ori'].append(stu_buffer_cost)
                        res_dict['stu_now'].append(all_score)
                        res_dict['update_iter'].append(len(stu_buffer_cost))

                        df = pd.DataFrame(res_dict)
                        df.to_csv('dynamic_update.csv')

                    # 全局更新教师模型
                    with torch.no_grad():
                    # 主进程计算新状态字典
                        state_dict = {}
                        teacher_state_dict = teacher_model.state_dict()
                        student_state_dict = student_model.state_dict()
                        for key in teacher_state_dict.keys():
                            state_dict[key] = alpha_ema * teacher_state_dict[key] + (1 - alpha_ema) * student_state_dict[key].detach()
                        # 所有GPU加载相同的状态
                        teacher_model.load_state_dict(state_dict)
                        # Clear buffer
                    stu_buffer_cost = []
                    stu_buffer_img = []
                    stu_buffer_mask = []
                # print('update')
            else:
                # print('reinitialize')
                # print(len(stu_buffer_cost), 'Load previous student model weight')
                with torch.no_grad():
                    student_model = selective_reinitialize(student_model, init_student_model.state_dict(), keep_modules)

                # Clear buffer
                stu_buffer_cost = []
                stu_buffer_img = []
                stu_buffer_mask = []
        else:
            # EMA update teacher after fix iteration
            if iter % fix_update_iter == 0:
                with torch.no_grad():
                    state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
                    for key, value in state_dict.items():
                        state_dict[key] = alpha_ema * value + (1 - alpha_ema) * student_state_dict[key].detach()
                    teacher_model.load_state_dict(state_dict)


        # Data pre-fetch
        target_images, target_masks, _ = target_fetcher.next()
        if target_images is not None:
            target_teacher_images, target_student_images = target_images[0], target_images[1]

        # Log
        if is_main_process() and (iter + 1) % print_freq == 0:
            str_out = ""
            for k,v in  target_loss_dict.items():
                str_out+= f"{k}:{float(v)} "
            
            print('Teaching epoch ' + str(epoch) + ' : [ ' + str(iter + 1) + '/' + str(total_iters) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()) + str_out, flush=flush)
                
    # Final process of loss dict
    epoch_loss /= total_iters
    for k, v in epoch_target_loss_dict.items():
        epoch_target_loss_dict[k] /= total_iters
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Teaching epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_target_loss_dict


def train_one_epoch_teaching_mask_with_piece(student_model: torch.nn.Module,
                                  teacher_model: torch.nn.Module,
                                  init_student_model: torch.nn.Module,
                                  criterion_pseudo: torch.nn.Module,
                                  criterion_pseudo_weak: torch.nn.Module,
                                  target_loader: DataLoader,
                                  optimizer: torch.optim.Optimizer,
                                  thresholds: List[float],
                                  coef_masked_img: float,
                                  alpha_ema: float,
                                  device: torch.device,
                                  epoch: int,
                                  keep_modules: List[str],
                                  clip_max_norm: float = 0.0,
                                  print_freq: int = 20,
                                  masking: Masking = None,
                                  flush: bool = True,
                                  piece_num = 10,
                                  piece_id = 0,
                                  target_fetcher= None,
                                  fix_update_iter: int = 1,
                                  max_update_iter: int = 5,
                                  dynamic_update: bool = False,
                                  stu_buffer_cost: List[float] = None,
                                  stu_buffer_img: List[torch.Tensor] = None,
                                  stu_buffer_mask: List[torch.Tensor] = None,
                                  res_dict: dict = None,
                                  use_pseudo_label_weights: bool = False,
                                  use_loss_student: bool = False):
    """
    Train the student model with the teacher model, using only unlabeled training set target (plus masked target image)
    """
    start_time = time.time()
    student_model.train()
    teacher_model.train()
    init_student_model.train()
    criterion_pseudo.train()
    criterion_pseudo_weak.train()
    if piece_id == 0:
        target_fetcher = DataPreFetcher(target_loader, device=device)
    else:
        target_fetcher = target_fetcher
    target_images, target_masks, _ = target_fetcher.next()
    target_teacher_images, target_student_images = target_images[0], target_images[1]
    # Record epoch losses
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)

    # Training data statistics
    epoch_target_loss_dict = defaultdict(float)
    total_iters = len(target_loader) // piece_num

    #maybe should init here
    stu_buffer_cost = []
    stu_buffer_img = []
    stu_buffer_mask = []
    for iter in range(total_iters):
        # Target teacher forward
        # progressive updating weight factor
        if target_images is None or target_masks is None:
            total_iters = iter
            break
        with torch.no_grad():
            teacher_out = teacher_model(target_teacher_images, target_masks)
            pseudo_labels = get_pseudo_labels(teacher_out['logit_all'][-1], teacher_out['boxes_all'][-1], thresholds)

        # Target student forward
        target_student_out = student_model(target_student_images, target_masks)
        # loss from pseudo labels of current teacher
        target_loss, target_loss_dict = criterion_pseudo(target_student_out, pseudo_labels)

        # Masked target student forward
        masked_target_images = masking(target_student_images)
        masked_target_student_out = student_model(masked_target_images, target_masks)
        # loss from pseudo labels of current teacher
        masked_target_loss, masked_target_loss_dict = criterion_pseudo(masked_target_student_out, pseudo_labels)

        # Final loss
        loss = target_loss + coef_masked_img * masked_target_loss

        # Loss from pseudo labels of previous student (just testing, not used)
        # if use_loss_student:
        #     # Loss from pseudo labels of previous student
        #     with torch.no_grad():
        #         student_out = student_model(target_teacher_images, target_masks)
        #         pseudo_labels_student = get_pseudo_labels(student_out['logit_all'][-1], student_out['boxes_all'][-1],
        #                                                   thresholds)
        #     target_loss_student, target_loss_dict_student = criterion_pseudo_weak(target_student_out,
        #                                                                         pseudo_labels_student, use_pseudo_label_weights)
        #     masked_target_loss_student, masked_target_loss_dict_student = criterion_pseudo_weak(masked_target_student_out,
        #                                                                                       pseudo_labels_student, use_pseudo_label_weights)
        #
        #     # Final loss
        #     loss_student = target_loss_student + coef_masked_img * masked_target_loss_student
        #     loss += loss_student

        # Dynamic update EMA teacher : Create buffer cost and buffer image in student model
        if dynamic_update:
            with torch.no_grad():
                student_out = student_model(target_teacher_images, target_masks)
            # variance logit
            student_out_var = student_out['logit_all'].var(dim=0)
            var_total = student_out_var.mean()#.item()
            if is_dist_avail_and_initialized():
                if not isinstance(var_total, torch.Tensor):
                    raise TypeError(f"Expected tensor, got {type(var_total)}")
                dist.all_reduce(var_total, op=dist.ReduceOp.SUM)
                var_total /= dist.get_world_size()
            stu_buffer_cost.append(var_total.item())

            # Store batch data to buffer
            stu_buffer_img.append(target_teacher_images.clone().detach())
            stu_buffer_mask.append(target_masks.clone().detach())

            if len(stu_buffer_cost) == 1:
                with torch.no_grad():
                    #in this step, init_student model dino factor is updated
                    init_student_model.load_state_dict(student_model.state_dict())

            if len(stu_buffer_cost) >= 1:
                with torch.no_grad():
                    init_student_out = init_student_model(target_teacher_images, target_masks)
                    pseudo_labels_init_student = get_pseudo_labels(init_student_out['logit_all'][-1], init_student_out['boxes_all'][-1],
                                                              thresholds)
                # Loss from pseudo labels of init student
                init_student_loss, init_student_loss_dict = criterion_pseudo_weak(target_student_out,
                                                                                    pseudo_labels_init_student, use_pseudo_label_weights)
                masked_init_student_loss, masked_init_student_loss_dict = criterion_pseudo_weak(masked_target_student_out,
                                                                                                  pseudo_labels_init_student, use_pseudo_label_weights)
                loss_init_student = init_student_loss + coef_masked_img * masked_init_student_loss
                loss += loss_init_student

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip_max_norm)
        optimizer.step()

        # Record epoch losses
        epoch_loss += loss.detach()

        # update loss_dict
        for k, v in target_loss_dict.items():
            epoch_target_loss_dict[k] += v.detach().cpu().item()

        # Dynamic update EMA teacher : Update weight of teacher model
        if dynamic_update:
            if len(stu_buffer_cost) < max_update_iter:
                all_score = eval_stu(student_model, stu_buffer_img, stu_buffer_mask)
                compare_score = np.array(all_score) - np.array(stu_buffer_cost)
                # print(len(stu_buffer_cost), len(all_score), np.mean(compare_score<0))
                if np.mean(compare_score < 0) >= 0.5:

                    if is_main_process():
                        res_dict['stu_ori'].append(stu_buffer_cost)
                        res_dict['stu_now'].append(all_score)
                        res_dict['update_iter'].append(len(stu_buffer_cost))

                        df = pd.DataFrame(res_dict)
                        df.to_csv('dynamic_update.csv')

                    # 全局更新教师模型
                    with torch.no_grad():
                    # 主进程计算新状态字典
                        state_dict = {}
                        teacher_state_dict = teacher_model.state_dict()
                        student_state_dict = student_model.state_dict()
                        for key in teacher_state_dict.keys():
                            state_dict[key] = alpha_ema * teacher_state_dict[key] + (1 - alpha_ema) * student_state_dict[key].detach()
                        # 所有GPU加载相同的状态
                        teacher_model.load_state_dict(state_dict)
                        # Clear buffer
                    stu_buffer_cost = []
                    stu_buffer_img = []
                    stu_buffer_mask = []
                # print('update')
            else:
                # print('reinitialize')
                # print(len(stu_buffer_cost), 'Load previous student model weight')
                with torch.no_grad():
                    student_model = selective_reinitialize(student_model, init_student_model.state_dict(), keep_modules)

                # Clear buffer
                stu_buffer_cost = []
                stu_buffer_img = []
                stu_buffer_mask = []
        else:
            # EMA update teacher after fix iteration
            if iter % fix_update_iter == 0:
                with torch.no_grad():
                    state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
                    for key, value in state_dict.items():
                        state_dict[key] = alpha_ema * value + (1 - alpha_ema) * student_state_dict[key].detach()
                    teacher_model.load_state_dict(state_dict)


        # Data pre-fetch
        target_images, target_masks, _ = target_fetcher.next()
        if target_images is not None:
            target_teacher_images, target_student_images = target_images[0], target_images[1]

        # Log
        if is_main_process() and (iter + 1) % print_freq == 0:
            print('Teaching epoch ' + str(epoch) + ' : [ ' + str(iter + 1) + '/' + str(total_iters) + ' ] ' +
             'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)

    # Final process of loss dict
    epoch_loss /= total_iters
    for k, v in epoch_target_loss_dict.items():
        epoch_target_loss_dict[k] /= total_iters
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Teaching epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_target_loss_dict, target_fetcher

def vis_output(model_out, samples, prefix=""):
    import os
    import torch
    from detectron2.structures import Instances
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog,Metadata

    save_dir = 'vis_output/bdd100k'
    os.makedirs(save_dir, exist_ok=True)    
    metadata = MetadataCatalog.get("cityscape_2007_train_s")  # 注意：Cityscapes的注册名是"cityscapes"（小写）
 # ImageNet归一化参数
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])
    metadata  = Metadata(thing_classes=['person', 'car', 'train', 'rider', 'truck', 'motorcycle', 'bicycle', 'bus'])

    for idx, (target_teacher_images) in enumerate(samples):
        with torch.no_grad():
            # 获取图像尺寸（假设输入是 [B, C, H, W]）
            height, width = target_teacher_images.shape[-2:]
            instances = Instances(image_size=(height, width))
            # 提取预测框、类别和置信度
            out_scores = model_out['logit_all'][-1][0, :, 1:].sigmoid()
            out_boxes = model_out['boxes_all'][-1][0] #xywh
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
            image_np = target_teacher_images.cpu().numpy()  # [C, H, W]
            image_np = np.transpose(image_np, (1, 2, 0))  # -> [H, W, C]
            image_np = image_np * IMAGENET_STD.numpy() + IMAGENET_MEAN.numpy()  # 反归一化
            image_np = np.clip(image_np * 255, 0, 255).astype("uint8")
            #这块反归一化有点问题，因为图片是根据Image Net pretrain参数归一化的 
            vis = Visualizer(image_np, metadata=metadata, scale=1.0)
            vis_output = vis.draw_instance_predictions(instances.to(torch.device('cpu')))
            # 保存结果
            output_path = os.path.join(save_dir, f"{prefix}_pred_{idx}.png")
            vis_output.save(output_path)

def train_one_epoch_teaching_mask_with_piece_distill(student_model: torch.nn.Module,
                                static_teacher_model,
                                  teacher_model: torch.nn.Module,
                                  init_student_model: torch.nn.Module,
                                  criterion_pseudo: torch.nn.Module,
                                  criterion_pseudo_weak: torch.nn.Module,
                                  target_loader: DataLoader,
                                  optimizer: torch.optim.Optimizer,
                                  thresholds: List[float],
                                  coef_masked_img: float,
                                  alpha_ema: float,
                                  device: torch.device,
                                  epoch: int,
                                  keep_modules: List[str],
                                  clip_max_norm: float = 0.0,
                                  print_freq: int = 20,
                                  masking: Masking = None,
                                  flush: bool = True,
                                  piece_num = 10,
                                  piece_id = 0,
                                  target_fetcher= None,
                                  fix_update_iter: int = 1,
                                  max_update_iter: int = 5,
                                  dynamic_update: bool = False,
                                  stu_buffer_cost: List[float] = None,
                                  stu_buffer_img: List[torch.Tensor] = None,
                                  stu_buffer_mask: List[torch.Tensor] = None,
                                  res_dict: dict = None,
                                  use_pseudo_label_weights: bool = False,
                                  use_loss_student: bool = False):
    """
    Train the student model with the teacher model, using only unlabeled training set target (plus masked target image)
    """
    start_time = time.time()
    student_model.train()
    teacher_model.train()
    init_student_model.train()
    criterion_pseudo.train()
    criterion_pseudo_weak.train()
    if piece_id == 0:
        target_fetcher = DataPreFetcher(target_loader, device=device)
    else:
        target_fetcher = target_fetcher
    target_images, target_masks, _ = target_fetcher.next()
    target_teacher_images, target_student_images = target_images[0], target_images[1]
    # Record epoch losses
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)

    # Training data statistics
    epoch_target_loss_dict = defaultdict(float)
    total_iters = len(target_loader) // piece_num

    #maybe should init here
    stu_buffer_cost = []
    stu_buffer_img = []
    stu_buffer_mask = []
    for iter in range(total_iters):
        # Target teacher forward
        # progressive updating weight factor
        if target_images is None or target_masks is None:
            total_iters = iter
            break
        with torch.no_grad():
            teacher_out = teacher_model(target_teacher_images, target_masks)
            pseudo_labels = get_pseudo_labels(teacher_out['logit_all'][-1], teacher_out['boxes_all'][-1], thresholds)
            static_teacher_out = static_teacher_model(target_teacher_images, target_masks)
            # vis_output(teacher_out, target_teacher_images, f'ema_teacher_{iter}')
            # vis_output(static_teacher_out, target_teacher_images, f'static_teacher_{iter}')
            # import pdb;pdb.set_trace()
            static_pseudo_labels = get_pseudo_labels(static_teacher_out['logit_all'][-1], static_teacher_out['boxes_all'][-1], thresholds)
            pseudo_labels = merge_pseudo_labels(pseudo_labels, static_pseudo_labels)

        # Target student forward
        target_student_out = student_model(target_student_images, target_masks)
        # loss from pseudo labels of current teacher
        target_loss, target_loss_dict = criterion_pseudo(target_student_out, pseudo_labels)

        # Masked target student forward
        masked_target_images = masking(target_student_images)
        masked_target_student_out = student_model(masked_target_images, target_masks)
        # loss from pseudo labels of current teacher
        masked_target_loss, masked_target_loss_dict = criterion_pseudo(masked_target_student_out, pseudo_labels)

        # Final loss
        loss = target_loss + coef_masked_img * masked_target_loss

        # Loss from pseudo labels of previous student (just testing, not used)
        # if use_loss_student:
        #     # Loss from pseudo labels of previous student
        #     with torch.no_grad():
        #         student_out = student_model(target_teacher_images, target_masks)
        #         pseudo_labels_student = get_pseudo_labels(student_out['logit_all'][-1], student_out['boxes_all'][-1],
        #                                                   thresholds)
        #     target_loss_student, target_loss_dict_student = criterion_pseudo_weak(target_student_out,
        #                                                                         pseudo_labels_student, use_pseudo_label_weights)
        #     masked_target_loss_student, masked_target_loss_dict_student = criterion_pseudo_weak(masked_target_student_out,
        #                                                                                       pseudo_labels_student, use_pseudo_label_weights)
        #
        #     # Final loss
        #     loss_student = target_loss_student + coef_masked_img * masked_target_loss_student
        #     loss += loss_student

        # Dynamic update EMA teacher : Create buffer cost and buffer image in student model
        if dynamic_update:
            with torch.no_grad():
                student_out = student_model(target_teacher_images, target_masks)
            # variance logit
            student_out_var = student_out['logit_all'].var(dim=0)
            var_total = student_out_var.mean()#.item()
            if is_dist_avail_and_initialized():
                if not isinstance(var_total, torch.Tensor):
                    raise TypeError(f"Expected tensor, got {type(var_total)}")
                dist.all_reduce(var_total, op=dist.ReduceOp.SUM)
                var_total /= dist.get_world_size()
            stu_buffer_cost.append(var_total.item())

            # Store batch data to buffer
            stu_buffer_img.append(target_teacher_images.clone().detach())
            stu_buffer_mask.append(target_masks.clone().detach())

            if len(stu_buffer_cost) == 1:
                with torch.no_grad():
                    #in this step, init_student model dino factor is updated
                    init_student_model.load_state_dict(student_model.state_dict())

            if len(stu_buffer_cost) >= 1:
                with torch.no_grad():
                    init_student_out = init_student_model(target_teacher_images, target_masks)
                    pseudo_labels_init_student = get_pseudo_labels(init_student_out['logit_all'][-1], init_student_out['boxes_all'][-1],
                                                              thresholds)
                # Loss from pseudo labels of init student
                init_student_loss, init_student_loss_dict = criterion_pseudo_weak(target_student_out,
                                                                                    pseudo_labels_init_student, use_pseudo_label_weights)
                masked_init_student_loss, masked_init_student_loss_dict = criterion_pseudo_weak(masked_target_student_out,
                                                                                                  pseudo_labels_init_student, use_pseudo_label_weights)
                loss_init_student = init_student_loss + coef_masked_img * masked_init_student_loss
                loss += loss_init_student

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip_max_norm)
        optimizer.step()

        # Record epoch losses
        epoch_loss += loss.detach()

        # update loss_dict
        for k, v in target_loss_dict.items():
            epoch_target_loss_dict[k] += v.detach().cpu().item()

        # Dynamic update EMA teacher : Update weight of teacher model
        if dynamic_update:
            if len(stu_buffer_cost) < max_update_iter:
                all_score = eval_stu(student_model, stu_buffer_img, stu_buffer_mask)
                compare_score = np.array(all_score) - np.array(stu_buffer_cost)
                # print(len(stu_buffer_cost), len(all_score), np.mean(compare_score<0))
                if np.mean(compare_score < 0) >= 0.5:

                    if is_main_process():
                        res_dict['stu_ori'].append(stu_buffer_cost)
                        res_dict['stu_now'].append(all_score)
                        res_dict['update_iter'].append(len(stu_buffer_cost))

                        df = pd.DataFrame(res_dict)
                        df.to_csv('dynamic_update.csv')

                    # 全局更新教师模型
                    with torch.no_grad():
                    # 主进程计算新状态字典
                        state_dict = {}
                        teacher_state_dict = teacher_model.state_dict()
                        student_state_dict = student_model.state_dict()
                        for key in teacher_state_dict.keys():
                            state_dict[key] = alpha_ema * teacher_state_dict[key] + (1 - alpha_ema) * student_state_dict[key].detach()
                        # 所有GPU加载相同的状态
                        teacher_model.load_state_dict(state_dict)
                        # Clear buffer
                    stu_buffer_cost = []
                    stu_buffer_img = []
                    stu_buffer_mask = []
                # print('update')
            else:
                # print('reinitialize')
                # print(len(stu_buffer_cost), 'Load previous student model weight')
                with torch.no_grad():
                    student_model = selective_reinitialize(student_model, init_student_model.state_dict(), keep_modules)

                # Clear buffer
                stu_buffer_cost = []
                stu_buffer_img = []
                stu_buffer_mask = []
        else:
            # EMA update teacher after fix iteration
            if iter % fix_update_iter == 0:
                with torch.no_grad():
                    state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
                    for key, value in state_dict.items():
                        state_dict[key] = alpha_ema * value + (1 - alpha_ema) * student_state_dict[key].detach()
                    teacher_model.load_state_dict(state_dict)


        # Data pre-fetch
        target_images, target_masks, _ = target_fetcher.next()
        if target_images is not None:
            target_teacher_images, target_student_images = target_images[0], target_images[1]

        # Log
        if is_main_process() and (iter + 1) % print_freq == 0:
            print('Teaching epoch ' + str(epoch) + ' : [ ' + str(iter + 1) + '/' + str(total_iters) + ' ] ' +
             'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)

    # Final process of loss dict
    epoch_loss /= total_iters
    for k, v in epoch_target_loss_dict.items():
        epoch_target_loss_dict[k] /= total_iters
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Teaching epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_target_loss_dict, target_fetcher

def truncated_normal(mean, std, low, high, device):
    from torch.distributions import Normal
    """使用截断正态分布"""
    dist = Normal(mean, std)
    # 应用截断约束
    sample = dist.sample().to(device)  # 使用重参数化采样
    # 硬截断（可能会改变分布形状）
    return torch.clamp(sample, low, high)

def train_one_epoch_teaching_mask_distill(student_model: torch.nn.Module,
                                  teacher_model: torch.nn.Module,
                                  init_student_model: torch.nn.Module,
                                  static_teacher_model: torch.nn.Module,
                                  criterion_pseudo: torch.nn.Module,
                                  criterion_pseudo_weak: torch.nn.Module,
                                  target_loader: DataLoader,
                                  optimizer: torch.optim.Optimizer,
                                  thresholds: List[float],
                                  coef_masked_img: float,
                                  alpha_ema: float,
                                  device: torch.device,
                                  epoch: int,
                                  keep_modules: List[str],
                                  clip_max_norm: float = 0.0,
                                  print_freq: int = 20,
                                  masking: Masking = None,
                                  flush: bool = True,
                                  fix_update_iter: int = 1,
                                  max_update_iter: int = 5,
                                  dynamic_update: bool = False,
                                  stu_buffer_cost: List[float] = None,
                                  stu_buffer_img: List[torch.Tensor] = None,
                                  stu_buffer_mask: List[torch.Tensor] = None,
                                  res_dict: dict = None,
                                  use_pseudo_label_weights: bool = False,
                                  use_loss_student: bool = False):
    """
    Train the student model with the teacher model, using only unlabeled training set target (plus masked target image)
    """
    start_time = time.time()
    student_model.train()
    teacher_model.train()
    static_teacher_model.train()
    init_student_model.train()
    criterion_pseudo.train()
    criterion_pseudo_weak.train()
    target_fetcher = DataPreFetcher(target_loader, device=device)
    target_images, target_masks, _ = target_fetcher.next()
    target_teacher_images, target_student_images = target_images[0], target_images[1]
    # Record epoch losses
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)

    # Training data statistics
    epoch_target_loss_dict = defaultdict(float)
    total_iters = len(target_loader)
    # # Initialize buffers
    # stu_buffer_cost = []
    # stu_buffer_img = []
    # stu_buffer_mask = []
    for iter in range(total_iters):
        # Target teacher forward
        # progressive updating weight factor
        with torch.no_grad():
            teacher_out = teacher_model(target_teacher_images, target_masks)
            pseudo_labels = get_pseudo_labels(teacher_out['logit_all'][-1], teacher_out['boxes_all'][-1], thresholds, is_nms=True)
            # static teacher modeling
            # static_teacher_out = static_teacher_model(target_teacher_images, target_masks)
            # static_pseudo_labels = get_pseudo_labels(static_teacher_out['logit_all'][-1], static_teacher_out['boxes_all'][-1], thresholds)
            # pseudo_labels = merge_pseudo_labels(pseudo_labels, static_pseudo_labels)
            # pseudo_labels = static_pseudo_labels

            # unstable teacher
            if teacher_model is not None:
                old_factor = teacher_model.module.dino_factor.data
                new_factor = truncated_normal(mean=float(old_factor), std=0.1, low =0, high=1.0, device=student_model.module.dino_factor.device)
                # teacher_model.module.dino_factor = torch.tensor(0.2).cuda()
                teacher_model.module.dino_factor.data = new_factor
                static_teacher_out = teacher_model(target_teacher_images, target_masks)
                static_pseudo_labels = get_pseudo_labels(static_teacher_out['logit_all'][-1], static_teacher_out['boxes_all'][-1], thresholds)
                pseudo_labels = merge_pseudo_labels(pseudo_labels, static_pseudo_labels)
                teacher_model.module.dino_factor.data = old_factor
        # Target student forward
        target_student_out = student_model(target_student_images, target_masks)
        # loss from pseudo labels of current teacher
        target_loss, target_loss_dict = criterion_pseudo(target_student_out, pseudo_labels)

        # Masked target student forward
        masked_target_images = masking(target_student_images)
        masked_target_student_out = student_model(masked_target_images, target_masks)
        # loss from pseudo labels of current teacher
        masked_target_loss, masked_target_loss_dict = criterion_pseudo(masked_target_student_out, pseudo_labels)

        # Final loss
        loss = target_loss + coef_masked_img * masked_target_loss


        # Dynamic update EMA teacher : Create buffer cost and buffer image in student model
        if dynamic_update:
            with torch.no_grad():
                student_out = student_model(target_teacher_images, target_masks)
            # variance logit
            student_out_var = student_out['logit_all'].var(dim=0)
            var_total = student_out_var.mean()#.item()
            if is_dist_avail_and_initialized():
                if not isinstance(var_total, torch.Tensor):
                    raise TypeError(f"Expected tensor, got {type(var_total)}")
                dist.all_reduce(var_total, op=dist.ReduceOp.SUM)
                var_total /= dist.get_world_size()
            stu_buffer_cost.append(var_total.item())

            # Store batch data to buffer
            stu_buffer_img.append(target_teacher_images.clone().detach())
            stu_buffer_mask.append(target_masks.clone().detach())

            if len(stu_buffer_cost) == 1:
                with torch.no_grad():
                    init_student_model.load_state_dict(student_model.state_dict())

            if len(stu_buffer_cost) >= 1:
                with torch.no_grad():
                    init_student_out = init_student_model(target_teacher_images, target_masks)
                    pseudo_labels_init_student = get_pseudo_labels(init_student_out['logit_all'][-1], init_student_out['boxes_all'][-1],
                                                              thresholds)
                    pseudo_labels_init_student = merge_pseudo_labels(pseudo_labels, static_pseudo_labels)
                # Loss from pseudo labels of init student
                init_student_loss, init_student_loss_dict = criterion_pseudo_weak(target_student_out,
                                                                                    pseudo_labels_init_student, use_pseudo_label_weights)
                masked_init_student_loss, masked_init_student_loss_dict = criterion_pseudo_weak(masked_target_student_out,
                                                                                                  pseudo_labels_init_student, use_pseudo_label_weights)
                loss_init_student = init_student_loss + coef_masked_img * masked_init_student_loss
                loss += loss_init_student

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip_max_norm)
        optimizer.step()

        # Record epoch losses
        epoch_loss += loss.detach()

        # update loss_dict
        for k, v in target_loss_dict.items():
            epoch_target_loss_dict[k] += v.detach().cpu().item()

        # Dynamic update EMA teacher : Update weight of teacher model
        if dynamic_update:
            if len(stu_buffer_cost) < max_update_iter:
                all_score = eval_stu(student_model, stu_buffer_img, stu_buffer_mask)
                compare_score = np.array(all_score) - np.array(stu_buffer_cost)
                # print(len(stu_buffer_cost), len(all_score), np.mean(compare_score<0))
                if np.mean(compare_score < 0) >= 0.5:

                    if is_main_process():
                        res_dict['stu_ori'].append(stu_buffer_cost)
                        res_dict['stu_now'].append(all_score)
                        res_dict['update_iter'].append(len(stu_buffer_cost))

                        df = pd.DataFrame(res_dict)
                        df.to_csv('dynamic_update.csv')

                    # 全局更新教师模型
                    with torch.no_grad():
                    # 主进程计算新状态字典
                        state_dict = {}
                        teacher_state_dict = teacher_model.state_dict()
                        student_state_dict = student_model.state_dict()
                        for key in teacher_state_dict.keys():
                            state_dict[key] = alpha_ema * teacher_state_dict[key] + (1 - alpha_ema) * student_state_dict[key].detach()
                        # 所有GPU加载相同的状态
                        teacher_model.load_state_dict(state_dict)
                        # Clear buffer
                    stu_buffer_cost = []
                    stu_buffer_img = []
                    stu_buffer_mask = []
                # print('update')
            else:
                # print('reinitialize')
                # print(len(stu_buffer_cost), 'Load previous student model weight')
                with torch.no_grad():
                    student_model = selective_reinitialize(student_model, init_student_model.state_dict(), keep_modules)

                # Clear buffer
                stu_buffer_cost = []
                stu_buffer_img = []
                stu_buffer_mask = []
        else:
            # EMA update teacher after fix iteration
            if iter % fix_update_iter == 0:
                with torch.no_grad():
                    state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
                    for key, value in state_dict.items():
                        state_dict[key] = alpha_ema * value + (1 - alpha_ema) * student_state_dict[key].detach()
                    teacher_model.load_state_dict(state_dict)


        # Data pre-fetch
        target_images, target_masks, _ = target_fetcher.next()
        if target_images is not None:
            target_teacher_images, target_student_images = target_images[0], target_images[1]

        # Log
        if is_main_process() and (iter + 1) % print_freq == 0:
            print('Teaching epoch ' + str(epoch) + ' : [ ' + str(iter + 1) + '/' + str(total_iters) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)
    
    # Final process of loss dict
    epoch_loss /= total_iters
    for k, v in epoch_target_loss_dict.items():
        epoch_target_loss_dict[k] /= total_iters
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Teaching epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_target_loss_dict
@torch.no_grad()
def evaluate(model: torch.nn.Module,
             criterion: torch.nn.Module,
             data_loader_val: DataLoader,
             device: torch.device,
             print_freq: int,
             enable_nms=False,
             output_result_labels: bool = False,
             flush: bool = False):
    start_time = time.time()
    model.eval()
    criterion.eval()
    if hasattr(data_loader_val.dataset, 'coco') or hasattr(data_loader_val.dataset, 'anno_file'):
        evaluator = CocoEvaluator(data_loader_val.dataset.coco)
        coco_data = json.load(open(data_loader_val.dataset.anno_file, 'r'))
        # dataset_annotations = [[] for _ in range(len(coco_data['images']))]
        dataset_annotations = defaultdict(list)
    else:
        raise ValueError('Unsupported dataset type.')
    epoch_loss = 0.0
    for i, (images, masks, annotations) in enumerate(data_loader_val):
        # To CUDA
        images = images.to(device)
        masks = masks.to(device)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        # Forward
        out = model(images, masks)
        logit_all, boxes_all = out['logit_all'], out['boxes_all']
        # Get pseudo labels
        if output_result_labels:
            results = get_pseudo_labels(logit_all[-1], boxes_all[-1], [0.4 for _ in range(9)])
            for anno, res in zip(annotations, results):
                image_id = anno['image_id'].item()
                orig_image_size = anno['orig_size']
                img_h, img_w = orig_image_size.unbind(0)
                scale_fct = torch.stack([img_w, img_h, img_w, img_h])
                converted_boxes = convert_to_xywh(box_cxcywh_to_xyxy(res['boxes'] * scale_fct))
                converted_boxes = converted_boxes.detach().cpu().numpy().tolist()
                for label, box in zip(res['labels'].detach().cpu().numpy().tolist(), converted_boxes):
                    pseudo_anno = {
                        'id': 0,
                        'image_id': image_id,
                        'category_id': label,
                        'iscrowd': 0,
                        'area': box[-2] * box[-1],
                        'bbox': box
                    }
                    # dataset_annotations[image_id].append(pseudo_anno)
                    dataset_annotations[image_id].append(pseudo_anno)
        # Loss
        loss, loss_dict = criterion(out, annotations)
        epoch_loss += loss
        if is_main_process() and (i + 1) % print_freq == 0:
            print('Evaluation : [ ' + str(i + 1) + '/' + str(len(data_loader_val)) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)
        # mAP
        orig_image_sizes = torch.stack([anno['orig_size'] for anno in annotations], dim=0)
        results = post_process(logit_all[-1], boxes_all[-1], orig_image_sizes, 100, is_nms=False)
        results = {anno['image_id'].item(): res for anno, res in zip(annotations, results)}
        evaluator.update(results)
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    aps = evaluator.summarize()
    epoch_loss /= len(data_loader_val)
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Evaluation finished. Time cost: ' + total_time_str, flush=flush)
    # Save results
    if output_result_labels:
        dataset_annotations_return = []
        id_cnt = 0
        # for image_anno in dataset_annotations:
        for image_anno in dataset_annotations.values():
            for box_anno in image_anno:
                box_anno['id'] = id_cnt
                id_cnt += 1
                dataset_annotations_return.append(box_anno)
        coco_data['annotations'] = dataset_annotations_return
        return aps, epoch_loss / len(data_loader_val), coco_data
    return aps, epoch_loss / len(data_loader_val)


def eval_stu(student_model: torch.nn.Module,
             stu_buffer_img: List[torch.Tensor],
             stu_buffer_mask: List[torch.Tensor]):
    """
    Evaluate student model with variance of logit
    """
    student_model.eval()
    all_score = []
    with torch.no_grad():
        for i in range(len(stu_buffer_img)):
            # student_out['logit_all']: [num_decoder_layers, batch size, num_queries, num_classes]
            student_out = student_model(stu_buffer_img[i], stu_buffer_mask[i])

            student_out_var = student_out['logit_all'].var(dim=0)
            var_total = student_out_var.mean()
            if is_dist_avail_and_initialized():
                if not isinstance(var_total, torch.Tensor):
                    raise TypeError(f"Expected tensor, got {type(var_total)}")
                dist.all_reduce(var_total, op=dist.ReduceOp.SUM)
                var_total /= dist.get_world_size()
            all_score.append(var_total.item())

    return all_score
