#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""
import argparse
import random
import copy
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.nn.parallel import DistributedDataParallel
from collections import OrderedDict
from detectron2.engine.defaults import create_ddp_model
import detectron2.utils.comm as comm
from detectron2.utils.file_io import PathManager
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.utils.events import (
    get_event_storage,
    CommonMetricPrinter, 
    JSONWriter, 
    TensorboardXWriter
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch, SimpleTrainer, hooks, HookBase
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scgsheduler, build_optimizer
from detectron2.modeling.roi_heads import build_roi_heads, StandardROIHeads
from detectron2.structures.instances import Instances
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.config import LazyConfig
import tqdm
from torch.cuda.amp import autocast

from engine import *
from build_modules import *
from datasets.augmentations import train_trans, val_trans, strong_trans
from utils import get_rank, init_distributed_mode, resume_and_load, save_ckpt, selective_reinitialize, is_main_process
            
        
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
        self.width = 10
        self.weight = 0.5
        self.schedule_type = schedule_type
    def get_value(self, epoch, saturate_epoch=10, schedule_type='sine'):
        assert epoch >=0 
        import math
        return math.sin(epoch/self.width * math.pi / 2) * self.weight
    @torch.no_grad()
    def after_step(self):
        import math
        if (self.trainer.iter + 1) % self._period == 0 or (
            self.trainer.iter == self.trainer.max_iter - 1
        ) or (self.trainer.iter==0):
            epoch = (self.trainer.iter+1) // self._period
            factor1 = self.get_value(epoch, self.max_epoch, self.schedule_type)
            factor2 = self.get_value(epoch, self.max_epoch, self.schedule_type)
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

    def __init__(self, model_student, model_teacher, period, beta=0.99):
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

def test_sfda(cfg, model, prefix):
    # for dataset_name in cfg.DATASETS.TEST[0]:
    results = {}
    dataset_name = cfg.DATASETS.TEST[0]
    cfg.defrost()
    cfg.SOURCE_FREE.TYPE = False
    cfg.freeze()
    test_data_loader = build_detection_test_loader(cfg, dataset_name)
    test_metadata = MetadataCatalog.get(dataset_name)
    evaluator = get_evaluator(
        cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
    )
    results_i = inference_on_dataset(model, test_data_loader, evaluator)
    # print(results_i.keys())
    # results[dataset_name] = results_i['bbox']
    if comm.is_main_process():
        logger.info("Evaluation results for {} in csv format:".format(dataset_name))
        print(results_i)
        print_csv_format(results_i)
        #pdb.set_trace()
        cls_names = test_metadata.get("thing_classes")
        cls_aps = results_i['bbox']['class-AP50']
        results['AP'] = results_i['bbox']['AP']
        results['AP50'] = results_i['bbox']['AP50']
        results['AP75'] = results_i['bbox']['AP75']

        for i in range(len(cls_aps)):
            logger.info("AP for {}: {}".format(cls_names[i], cls_aps[i]))
            results['AP_' + cls_names[i]] = cls_aps[i]
        results = {prefix + "_" + k:v for k,v in results.items()}
        return results
    else:
        return None


class Trainer(SimpleTrainer):
    def __init__(
        self,
        model_teacher,
        model_student,
        dino_roi_head,
        dataloader,
        optimizer,
        criterion,
        criterion_pseudo,
        criterion_weak,
        clip_grad_params=None,
        amp=False,
        mode= None,
        len_epoch=None,
    ):
        super().__init__(model=model_student, data_loader=dataloader, optimizer=optimizer)
        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model_student, DistributedDataParallel):
            assert not (model_student.device_ids and len(model_student.device_ids) > 1), unsupported

        if amp:
            from torch.cuda.amp import GradScaler
            self.grad_scaler = GradScaler()
        self.model_student = model_student
        self.model_teacher = model_teacher
        self.criterion = criterion
        self.criterion_pseudo = criterion_pseudo
        self.criterion_weak = criterion_weak    

        self.clip_grad_params = clip_grad_params
        self.len_epoch = len_epoch
        self.enable_dino = cfg.DINOHEAD.ENABLED
        self.amp = amp
        self.model_teacher.eval()
        self.model_student.train()
    def run_step_standard(self, model, images, masks):
        out = model(images, masks)
        # Loss
        loss, loss_dict = self.criterion(out, annotations)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        # if is_main_process() and (i + 1) % print_freq == 0:
        #     print('Training epoch ' + str(epoch) + ' : [ ' + str(i + 1) + '/' + str(len(data_loader)) + ' ] ' +
        #           'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)
        self._write_metrics(loss_dict, data_time)
    def run_step_teaching_standard(self, model, images, target_masks):
        target_teacher_images, target_student_images = images[0], images[1]
        with torch.no_grad():
            teacher_out = self.teacher_model(target_teacher_images, target_masks)
            pseudo_labels = self.get_pseudo_labels(teacher_out['logit_all'][-1], teacher_out['boxes_all'][-1], thresholds)

        out = model(images, masks)
        # Loss
         # Target student forward
        target_student_out = self.student_model(target_student_images, target_masks)
        target_loss, target_loss_dict = self.criterion_pseudo(target_student_out, pseudo_labels)

        loss = target_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), clip_max_norm)
        optimizer.step()

        self._write_metrics(target_loss_dict, data_time)

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model_student.training, "[Trainer] model was changed to eval mode!"
        assert not self.model_teacher.training, "[Trainer] model was changed to eval mode!"

        assert torch.cuda.is_available(), "[Trainer] CUDA is required for AMP training!"

        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        images, masks, annotations = next(self._data_loader_iter)
        data_time = time.perf_counter() - start
        iter_start_time = time.time()
        if self.mode == 'teaching_standard':
            run_step_standard(self.model_stu, images, masks)
        """
        If you want to do something with the losses, you can wrap the model.
        """
    
        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()

    def clip_grads(self, params):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return torch.nn.utils.clip_grad_norm_(
                parameters=params,
                **self.clip_grad_params,
            )

    def state_dict(self):
        ret = super().state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)


def do_train(model, device, args):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    #build models and dino head
    # model_student = build_model(cfg)
    # model_student.cfg = cfg
    device = torch.device(args.device)
    model_stu = model
    init_model_stu = build_teacher(args, model_stu, device)
    model_tch = build_teacher(args, model_stu, device)
    if args.distributed:
        model_stu = DistributedDataParallel(model_stu, device_ids=[args.gpu], find_unused_parameters=False)
        model_tch = DistributedDataParallel(model_tch, device_ids=[args.gpu])
        init_model_stu = DistributedDataParallel(init_model_stu, device_ids=[args.gpu])
    #atatch dino head to models

    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model_student))
    # model_student.to(cfg.train.device)
    # model_student.to(cfg.train.device)
    
    # instantiate optimizer
    model_student.dino_roi_head = dino_roi_head
    optimizer = build_optimizer(args, model_stu)

    #init criterion
    criterion = build_criterion(args, device)
    criterion_pseudo = build_criterion(args, device)
    criterion_pseudo_weak = build_criterion(args, device, only_class_loss=args.only_class_loss)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epoch_lr_drop)

    # build training loader
    target_loader = build_dataloader_teaching(args, args.target_dataset, 'target', 'train')
    val_loader = build_dataloader(args, args.target_dataset, 'target', 'val', val_trans)    # cfg.SOLVER.MAX_ITER = len_epoch * 10
    idx_to_class = val_loader.dataset.coco.cats

    # create ddp model
    # if cfg.DYNAMIC_DINO.ENABLED:
    #     model_student.dino_backbone = model_teacher.dino_backbone
    #     model_student.dino_transform = model_teacher.dino_transform
        # model_teacher.proj = model_student.proj
    # build model ema
    len_epoch = 370
    trainer = Trainer(
        model_student=model_stu,
        model_teacher=model_tch,
        dataloader=data_loader,
        optimizer=optimizer,
        len_epoch=len_epoch,
        amp = args.amp,
        clip_grad_params=None,
        criterion = criterion,
        criterion_pseudo = criterion_pseudo,
        criterion_weak=criterion_weak,
        mode = args.mode,
        args= args
    )

    checkpointer_teacher = DetectionCheckpointer(
        model_teacher,
        cfg.OUTPUT_DIR,
        trainer=trainer,
        # save model ema
    )
    checkpointer_student = DetectionCheckpointer(
        model_student,
        cfg.OUTPUT_DIR,
        trainer=trainer,
        # save model ema
    )
    if comm.is_main_process():
        # writers = default_writers(cfg.OUTPUT_DIr, cfg.train.max_iter)
        output_dir = cfg.OUTPUT_DIR
        PathManager.mkdirs(output_dir)
        writers = [
            CommonMetricPrinter(cfg.SOLVER.MAX_ITER),
            JSONWriter(os.path.join(output_dir, "metrics.json")),
            TensorboardXWriter(output_dir, window_size=1),
            # WandbWriter(cfg)
        ]
    if args.fast:
        eval_period = 5 * len_epoch
    else:
        eval_period = len_epoch
    
    log_period = 20
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            # hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            EMAHook(model_student, model_teacher, 1, beta = args.alpha_ema),
            SINEFactorHook(model_student, model_teacher, len_epoch,  schedule_type='sine') if args.enable_dino,
            hooks.EvalHook(eval_period, lambda: do_eval(cfg, model_student, 'student')),
            hooks.EvalHook(eval_period, lambda: do_eval(cfg, model_teacher, 'teacher')),
            hooks.PeriodicCheckpointer(checkpointer_teacher, len_epoch, max_iter=cfg.SOLVER.MAX_ITER) if comm.is_main_process() else None,
            hooks.PeriodicWriter(writers,period=log_period) if comm.is_main_process() else None,
        ]
    )
    # DetectionCheckpointer(model_student, save_dir=cfg.OUTPUT_DIR).load(args.model_dir)
    # DetectionCheckpointer(model_teacher, save_dir=cfg.OUTPUT_DIR).load(args.model_dir)
  
    checkpointer_student.resume_or_load(args.model_dir, resume=args.resume)
    checkpointer_teacher.resume_or_load(args.model_dir, resume=args.resume)
   
    if args.cache_pl_label:
        cache_pl_label(cfg, model_teacher, 0.0,  './teacher_sfda_pl_00.json')
        return 0
   
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.SOLVER.MAX_ITER)
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
        do_train(model, device, args)
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