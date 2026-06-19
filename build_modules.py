import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.sampler import BatchSampler, RandomSampler
from datasets.coco_style_dataset import CocoStyleDataset, CocoStyleDatasetTeaching
from models.backbones import ResNet50MultiScale, ResNet18MultiScale, ResNet101MultiScale
from models.positional_encoding import PositionEncodingSine
from models.deformable_detr import DeformableDETR
from models.deformable_transformer import DeformableTransformer
from models.criterion import SetCriterion
from datasets.augmentations import weak_aug, strong_aug, mid_aug,base_trans


def build_sampler(args, dataset, split):
    if split == 'train':
        if args.distributed:
            sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
        else:
            sampler = RandomSampler(dataset)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    else:
        if args.distributed:
            sampler = DistributedSampler(dataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, args.eval_batch_size, drop_last=False)
    return batch_sampler


def build_dataloader(args, dataset_name, domain, split, trans):
    dataset = CocoStyleDataset(root_dir=args.data_root,
                               dataset_name=dataset_name,
                               domain=domain,
                               split=split,
                               transforms=trans)
    batch_sampler = build_sampler(args, dataset, split)
    data_loader = DataLoader(dataset=dataset,
                             batch_sampler=batch_sampler,
                             collate_fn=CocoStyleDataset.collate_fn,
                             num_workers=args.num_workers)
    return data_loader


def build_dataloader_teaching(args, dataset_name, domain, split, aug_level=2):
    aug = [weak_aug, mid_aug, strong_aug][aug_level]
    dataset = CocoStyleDatasetTeaching(root_dir=args.data_root,
                                       dataset_name=dataset_name,
                                       domain=domain,
                                       split=split,
                                       weak_aug=weak_aug,
                                       strong_aug=aug,
                                       final_trans=base_trans)
    batch_sampler = build_sampler(args, dataset, split)
    data_loader = DataLoader(dataset=dataset,
                             batch_sampler=batch_sampler,
                             collate_fn=CocoStyleDatasetTeaching.collate_fn_teaching,
                             num_workers=args.num_workers)
    return data_loader

def load_dinov2(dino_repo ='dinov2',model_type="dinov2_vitb14_reg", checkpoint_path="./weights/dinov2_vitb14_reg4_pretrain.pth"):
    dino = torch.hub.load(dino_repo, model_type,source='local',pretrained=False).cuda()
    dino.load_state_dict(torch.load(checkpoint_path))
    from torchvision import transforms
    transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正则化
    ])

    for param in dino.parameters():
        param.requires_grad = False
    return dino, transform
def load_clip_visual(model_type="ViT-B/16", checkpoint_path="./weights/clip_vitb16.pth"):
    import clip
    model, preprocess = clip.load(model_type, device="cuda")
    # if checkpoint_path is not None:
    #     checkpoint = torch.load(checkpoint_path, map_location="cuda")
    #     model.load_state_dict(checkpoint)
    transform = preprocess.transforms[-1]  # 获取CLIP的归一化变换
    for param in model.parameters():
        param.requires_grad = False
    return model.visual.float(), transform
def load_sam_visual_encoder(
    sam_checkpoint=None,
    model_type="vit_l", 
    device="cuda",
    freeze=True,
    output_stride=16
):
    """
    专门加载SAM的视觉编码器（图像编码器）
    
    Args:
        sam_checkpoint: SAM权重文件路径
        model_type: 模型类型 ["vit_h", "vit_l", "vit_b"]
        device: 运行设备
        freeze: 是否冻结参数
        output_stride: 输出步长（影响特征图尺寸）
    
    Returns:
        image_encoder: SAM图像编码器
        transform: 对应的预处理变换
    """
    import os
    from segment_anything import sam_model_registry
    # 设置默认权重路径
    if sam_checkpoint is None:
        checkpoint_map = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth", 
            "vit_b": "sam_vit_b_01ec64.pth"
        }
        default_name = checkpoint_map.get(model_type)
        sam_checkpoint = os.path.join("./weights", default_name)
    
    # 检查文件是否存在
    if not os.path.exists(sam_checkpoint):
        raise FileNotFoundError(f"SAM权重文件不存在: {sam_checkpoint}")
    
    try:
        # 加载完整SAM模型
        sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        
        # 提取视觉编码器
        image_encoder = sam_model.image_encoder
        
        # 移动到指定设备
        image_encoder.to(device)
        
        # 冻结参数
        if freeze:
            for param in image_encoder.parameters():
                param.requires_grad = False
            image_encoder.eval()
        
        # SAM的预处理参数
        transform = get_sam_transform()
        
        print(f"成功加载SAM视觉编码器: {model_type}")
        print(f"输入尺寸: 1024x1024")
        print(f"输出特征维度: {get_sam_output_dim(model_type)}")
        
        return image_encoder.half(), transform
        
    except Exception as e:
        print(f"加载SAM视觉编码器失败: {e}")
        return None, None

def get_sam_output_dim(model_type):
    """获取SAM不同模型的输出维度"""
    dim_map = {
        "vit_h": 1280,
        "vit_l": 1024, 
        "vit_b": 768
    }
    return dim_map.get(model_type, 1280)

def get_sam_transform():
    """
    SAM的标准化参数
    SAM使用ImageNet的均值和标准差
    """
    from torchvision import transforms
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
def build_model(args, device):
    if args.backbone == 'resnet50':
        backbone = ResNet50MultiScale()
        # backbone = ResNet50MultiScaleInject(
        # injection_layers=['layer2'],
        # injection_channels={'layer2':768},)
    elif args.backbone == 'resnet18':
        backbone = ResNet18MultiScale()
    elif args.backbone == 'resnet101':
        backbone = ResNet101MultiScale()
    else:
        raise ValueError('Invalid args.backbone name: ' + args.backbone)
    position_encoding = PositionEncodingSine()
    if args.enable_dino:
        VFM_backbone, VFM_transform = load_dinov2()
        VFM_channel = 768
        # VFM_backbone_2, VFM_transform_2 = load_clip_visual("ViT-B/16")
        # VFM_channel_2 = 768
        VFM_backbone_2, VFM_transform_2, VFM_channel_2 = None, None, None
        
        # VFM_backbone,VFM_transform = load_clip_visual("ViT-L/14@336px")
        # VFM_backbone, VFM_transform = load_sam_visual_encoder()
    else:
        VFM_backbone = None
        VFM_transform = None
        VFM_channel = None

    transformer = DeformableTransformer(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout
    )
    model = DeformableDETR(
        backbone=backbone,
        VFM_backbone = VFM_backbone,
        VFM_transform = VFM_transform,
        VFM_channel = VFM_channel,
        VFM_backbone_2 = VFM_backbone_2,
        VFM_transform_2 = VFM_transform_2,
        VFM_channel_2 = VFM_channel_2,
        position_encoding=position_encoding,
        transformer=transformer,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        fuse_type= args.fuse_type,
        enable_query_alignment = args.enable_query_alignment,
        enable_feature_alignment = args.enable_feature_alignment,
        enable_encoder_alignment = args.enable_encoder_alignment,
    )
    model.to(device)
    return model


def build_criterion(args, device, only_class_loss=False, high_quality_matches=False):
    criterion = SetCriterion(
        num_classes=args.num_classes,
        coef_class=args.coef_class,
        coef_boxes=0.0 if only_class_loss else args.coef_boxes,
        coef_giou=0.0 if only_class_loss else args.coef_giou,
        coef_feat = args.coef_feat,
        alpha_focal=args.alpha_focal,
        high_quality_matches=high_quality_matches,
        device=device
    )
    criterion.to(device)
    return criterion


def build_optimizer(args, model):
    params_backbone = [param for name, param in model.named_parameters()
                       if 'backbone' in name]
    params_linear_proj = [param for name, param in model.named_parameters()
                          if 'reference_points' in name or 'sampling_offsets' in name]
    params = [param for name, param in model.named_parameters()
              if 'backbone' not in name and 'reference_points' not in name and 'sampling_offsets' not in name]
    param_dicts = [
        {'params': params, 'lr': args.lr},
        {'params': params_backbone, 'lr': args.lr_backbone},
        {'params': params_linear_proj, 'lr': args.lr_linear_proj},
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer


def build_teacher(args, student_model, device):
    teacher_model = build_model(args, device)
    state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
    for key, value in state_dict.items():
        state_dict[key] = student_state_dict[key].clone().detach()
    teacher_model.load_state_dict(state_dict)
    return teacher_model

def get_copy(model):
    copy = build_model(model.args, model.device)
    copy.load_state_dict(model.state_dict())
    return copy
