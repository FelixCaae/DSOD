import math
import copy
import numpy as np
import torch
from torch import nn
from torch.nn.functional import relu, interpolate
import torch.nn.functional as F
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from matplotlib import patches
def improved_grad_cam(features, gradients):
    """
    Grad-CAM++ 改进版本，更好的定位
    """
    # 使用二阶梯度加权
    grads_power = gradients ** 2
    alpha = grads_power / (2 * grads_power + features.sum(dim=[2,3], keepdim=True) + 1e-8)
    
    weights = (alpha * gradients).sum(dim=[2,3], keepdim=True)
    cams = (weights * features).sum(dim=1)
    return F.relu(cams)
def visualize_grad_cam(model, images, masks, save_path, target_class=None,target_layer_idx=2, layer_name='fusion_layer' ):
    # 注册hook获取特征和梯度
    
    features = {}
    gradients = {}
    inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
    )
    images = inv_normalize(images).clamp(0, 1).cuda()
    def get_features_hook(module, input, output):
        features[layer_name] = output
        
    def get_gradients_hook(module, grad_input, grad_output):
        gradients[layer_name] = grad_output[0]
    
    # 注册hook到融合层
    if hasattr(model, 'module'):
        target_layer = model.module.projs[target_layer_idx]#getattr(model, layer_name)
    else:
        target_layer = model.projs[target_layer_idx]
    handle_forward = target_layer.register_forward_hook(get_features_hook)
    handle_backward = target_layer.register_backward_hook(get_gradients_hook)
    
    # 前向传播
    model.eval()
    outputs = model(images, masks)
    
    # 选择要可视化的目标（最高分检测框或指定类别）
  
        # 选择置信度最高的检测框
        # scores = outputs['logits_all'][0][0]  # 第一个batch
        # if len(scores) > 0:
        #     max_score_idx = scores.argmax()
        #     target_box = outputs['boxes'][0][max_score_idx]
        #     target_class = outputs['labels'][0][max_score_idx]
        # else:
        #     return None
    
    # 计算梯度（对目标类别的得分）
    model.zero_grad()
    scores = outputs['logit_all'][0][0] #　L,C
    boxes = outputs['boxes_all'][0][0]
    if target_class is None:
        class_idx_ = scores.argmax()
        query_idx = class_idx_ // scores.shape[-1]
        class_idx = class_idx_ % scores.shape[-1]
        # class_score = outputs['scores'][0][max_score_idx] if 'max_score_idx' in locals() else outputs['logits_all'][-1][0].mean()
    else:
        class_idx = target_class
        query_idx = scores[:,class_idx].argmax()
    box_target= boxes[query_idx].cpu()
    class_score = scores[query_idx][class_idx] 
    (class_score ).backward()
    box_target = box_target.detach().clone() * torch.tensor([images.shape[3], images.shape[2], images.shape[3], images.shape[2]])
    print(box_target, class_score, class_idx)
    # box_target = boxes[query_idx][class_idx]
    # 获取特征和梯度
    feats = features[layer_name]  # [B, C, H, W]
    grads = gradients[layer_name]  # [B, C, H, W]
    # cams = improved_grad_cam(feats,grads)
    # 计算Grad-CAM权重（全局平均池化梯度）
    weights = grads.mean(dim=[2, 3])  # [B, C]
    
    # # 生成CAM
    cams = (feats * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # [B, H, W]
    cams = F.relu(cams)  # 只保留正贡献
    
    # # 归一化并上采样到原图大小
    cams = F.interpolate(cams.unsqueeze(1), size=images.shape[2:], mode='bilinear')
    cams = (cams - cams.min()) / (cams.max() - cams.min() + 1e-8)
    model.zero_grad()
    # 清理hook
    handle_forward.remove()
    handle_backward.remove()
    # 画矩形框 - Rectangle((x, y), width, height)
    h,w = images.shape[2:]
    rect = patches.Rectangle(box_target[:2] - box_target[2:]/2, box_target[2], box_target[3], 
                        linewidth=2, 
                        edgecolor='red', 
                        facecolor='none',  # 透明填充
                        label='aa')
    plt.figure()
    plt.gca().add_patch(rect)
    plt.imshow(images.cpu()[0].permute(1,2,0))
    plt.imshow(cams[0][0].detach().cpu(), alpha=0.5)
    plt.savefig(save_path)
    plt.close()
    return cams.squeeze(), target_class


def cca_analysis(cnn_features, dino_features, n_components=10):
    """
    典型相关分析：找到使两组特征相关性最大的投影方向
    """
    from sklearn.cross_decomposition import CCA
    
    # 展平特征： [B, C, H, W] -> [B, C*H*W]
    B,C1,H,W = cnn_features.shape
    cnn_flat = cnn_features.reshape(B*H*W, C1).cpu().numpy()
    B,C2,H,W = dino_features.shape
    dino_flat = dino_features.reshape(B*H*W, C2).cpu().numpy()
    
    # CCA分析
    cca = CCA(n_components=n_components,scale=True)

    cca.fit(cnn_flat, dino_flat)
    
    # 转换到最大相关空间
    cnn_c, dino_c = cca.transform(cnn_flat, dino_flat)
    
    # 计算典型相关系数（代表线性相关强度）
    canonical_correlations = []
    for i in range(n_components):
        corr = np.corrcoef(cnn_c[:, i], dino_c[:, i])[0, 1]
        canonical_correlations.append(corr)
    
    return {
        'canonical_correlations': canonical_correlations,  # 各维度的相关性
        'mean_canonical_correlation': np.mean(canonical_correlations),
        'max_canonical_correlation': np.max(canonical_correlations),
        'variance_explained': np.cumsum(canonical_correlations) / np.sum(canonical_correlations)
    }
def visualize_pca(feats, images, save_path):
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )
    original_images = inv_normalize(images).clamp(0, 1)
    original_images = original_images.cpu().numpy().transpose(0, 2, 3, 1)
    with torch.no_grad():
        # 展平特征 [B, C, H*W]
        B, C, H, W = feats.shape
        feat_flat = feats.reshape(B, C, -1).permute(0, 2, 1).reshape(-1, C).cpu()
        # 随机采样避免点太多
        n_samples = min(1000, len(feat_flat))
        indices = torch.randperm(len(feat_flat))[:n_samples]
        feat_flat_sample = feat_flat[indices].cpu().numpy()

        # PCA降维
        pca = PCA(n_components=3)
        pca.fit_transform(feat_flat_sample)
        feat_flat = pca.transform(feat_flat)

        # 可视化
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_images[0])
        plt.subplot(1, 2, 2)
        plt.imshow(feat_flat.reshape(H,W,3))
        plt.title('PCA Feature Distribution')
        plt.legend()

        plt.legend()
        plt.savefig(save_path)
        return plt.gcf()    
# 使用示例
@torch.no_grad()
def visualize_feature_distribution(cnn_feats, dino_proj_feats,fused_feats, images,target_layer, n_components=3):
    """
    通过PCA可视化特征分布变化
    """

    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )
    original_images = inv_normalize(images).clamp(0, 1)
    original_images = original_images.cpu().numpy().transpose(0, 2, 3, 1)
    
    with torch.no_grad():
        # 展平特征 [B, C, H*W]
        B, C, H, W = cnn_feats.shape
        cnn_flat = cnn_feats.reshape(B, C, -1).permute(0, 2, 1).reshape(-1, C).cpu()
        fused_feats_flat = fused_feats.reshape(B, C, -1).permute(0, 2, 1).reshape(-1, C).cpu()
        B, C, H, W = dino_proj_feats.shape
        dino_flat = dino_proj_feats.reshape(B, C, -1).permute(0, 2, 1).reshape(-1, C).cpu()
        # 随机采样避免点太多
        n_samples = min(5000, len(cnn_flat))
        indices = torch.randperm(len(cnn_flat))[:n_samples]
        
        cnn_sampled = cnn_flat[indices].cpu().numpy()
        dino_sampled = dino_flat[indices].cpu().numpy()
        fused_feats_sampled = fused_feats_flat[indices].cpu().numpy()
        def percentile_normalize_pca(pca_result, lower=1, upper=99):
            """
            百分位归一化：截断异常值的影响
            科学意义：避免极端值主导颜色映射，显示主要分布
            """
            normalized = pca_result.copy()
            for i in range(pca_result.shape[1]):
                pc = pca_result[:, i]
                p_low, p_high = np.percentile(pc, lower), np.percentile(pc, upper)
                pc = np.clip(pc, p_low, p_high)
                normalized[:, i] = (pc - p_low) / (p_high - p_low + 1e-8)
            return normalized
        # PCA降维
        cnn_pca = PCA(n_components=3)
        cnn_pca.fit_transform(cnn_sampled)
        cnn_3d_flat = cnn_pca.transform(cnn_flat)

        dino_pca = PCA(n_components=3)
        dino_pca.fit_transform(dino_sampled)
        dino_3d_flat = dino_pca.transform(dino_flat)

        fused_pca =  PCA(n_components=3)
        fused_pca.fit_transform(fused_feats_sampled)
        fused_feats_3d_flat = fused_pca.transform(fused_feats_flat)

        # 可视化
        plt.figure(figsize=(15, 15))
        plt.axis('off')
        plt.subplot(2, 2, 1)
        plt.imshow(original_images[0][:,:,])
        plt.subplot(2, 2, 2)
        plt.imshow(percentile_normalize_pca(cnn_3d_flat).reshape(H,W,3))
        plt.title('CNN Feature Distribution')
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.imshow(percentile_normalize_pca(dino_3d_flat).reshape(H,W,3))
        plt.title('DINO Projected Feature Distribution')

        plt.subplot(2, 2, 4)
        plt.imshow(percentile_normalize_pca(fused_feats_3d_flat).reshape(H,W,3))
        plt.title('Fused Feature Distribution')


        plt.legend()
        plt.savefig(f'pca_{target_layer}.jpg')
        return plt.gcf()
def analyze_orthogonality(cnn_feats, dino_proj_feats, fused_feats):
    """系统分析特征间的正交性"""
    results = {}
    
    for i in range(cnn_feats.shape[0]):
        cnn = cnn_feats[i].flatten(1)  # [C, H*W]
        dino = dino_proj_feats[i].flatten(1)
        fused = fused_feats[i].flatten(1)
        
        # 1. 通道间余弦相似度矩阵
        cnn_dino_sim = F.cosine_similarity(cnn.unsqueeze(1), dino.unsqueeze(0), dim=2)
        avg_channel_sim = cnn_dino_sim.mean().item()
        
        # 2. 主成分正交性分析
        u_cnn, s_cnn, _ = torch.svd(cnn)
        u_dino, s_dino, _ = torch.svd(dino)
        
        # 计算主成分之间的夹角
        principal_angles = []
        for j in range(min(10, u_cnn.shape[1])):
            dot_product = torch.dot(u_cnn[:, j], u_dino[:, j])
            angle = torch.acos(torch.clamp(dot_product.abs(), -1, 1))
            principal_angles.append(torch.rad2deg(angle).item())
        
        results[i] = {
            'avg_channel_similarity': avg_channel_sim,
            'principal_angles': principal_angles,
            'orthogonality_score': 1 - avg_channel_sim,  # 越高越正交
        }
    
    return results
@torch.no_grad()
def visualize_heat_map(images,heat_map, save_path):
    """
    Visualize DINO features as heatmaps overlayed on original images
    
    Args:
        images: Preprocessed images (tensor) in ImageNet normalization
        dino_feats: Dictionary containing DINO features
        target_layer: Target layer name for feature extraction
        gate_scores: Optional gate scores for similarity visualization
    """
    from torchvision import transforms
    from matplotlib import pyplot as plt
    import torch.nn.functional as F
    # Reverse ImageNet normalization to get original image range
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )
    original_images = inv_normalize(images).clamp(0, 1)
    original_images = original_images.cpu().numpy().transpose(0, 2, 3, 1)
    import cv2
    img_to_save = (original_images[0] * 255).astype(np.uint8)

    # OpenCV使用BGR格式，如果原图是RGB需要转换
    # 如果你的images是RGB格式，需要转换为BGR
    img_bgr = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
    cv2.imwrite("original_image.png", img_bgr)
    # Get the spatial dimensions of the original image
    img_h, img_w = original_images.shape[1:3]
    
    # Create figure for visualization
    num_images = len(images)
    fig, axes = plt.subplots(num_images, 2, figsize=(15, 5*num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_images):
        # Original image
        axes[i, 0].imshow(original_images[i])
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        # L2 Norm heatmap
        if True:
            # Interpolate to original image size
            heat_map = F.interpolate(heat_map.unsqueeze(0).unsqueeze(0), 
                                     size=(img_h, img_w), 
                                     mode='bilinear',
                                     align_corners=False).squeeze().cpu().numpy()
            
            # Normalize heatmap
            
            # Overlay heatmap on image
            axes[i, 1].imshow(original_images[i])
            im = axes[i, 1].imshow(heat_map,  alpha=0.5)
            im = axes[i, 1].imshow(heat_map,  alpha=0.5)
            axes[i, 1].set_title(f' Heatmap ')
            axes[i, 1].axis('off')
            # plt.colorbar(im, ax=axes[i, 1])
    cbar = fig.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path)

def vis_output(model_out, target_teacher_images, save_path, id_mapping=lambda x:x):
    import os
    import torch
    from detectron2.structures import Instances
    from detectron2.utils.visualizer import Visualizer,ColorMode
    from detectron2.data import MetadataCatalog,Metadata

    save_dir = 'vis_output'
    os.makedirs(save_dir, exist_ok=True)    
    metadata = MetadataCatalog.get("cityscape_2007_train_s")  # 注意：Cityscapes的注册名是"cityscapes"（小写）
 # ImageNet归一化参数
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])
    metadata  = Metadata(thing_classes=['none','person', 'car ', 'train', 'rider', 'truck', 'motorcycle', 'bicycle', 'bus'])
    metadata.thing_colors = [
    (0, 0, 0),       # 'none' - 黑色
    (30, 122, 165),   #’person' 蓝色
    (220, 20, 60),      # 'car' - 红色
    (0, 100, 150),    # 'train' - 深青
    (255, 165, 0),   # 'rider' - 橙色
    (0, 0, 230),     # 'truck' - 深蓝
    (119, 11, 32),   # 'motorcycle' - 栗色
    (0, 60, 100),    # 'bicycle' - 深青
    (100, 0, 192)      # 'bus' - 亮蓝
    ]
    if True:    
        with torch.no_grad():
            # 获取图像尺寸（假设输入是 [B, C, H, W]）

            height, width = target_teacher_images.shape[-2:]
            instances = Instances(image_size=(height, width))
            # 提取预测框、类别和置信度
            out_scores, out_boxes = model_out[0]['scores'], model_out[0]['boxes']
            pred_classes = id_mapping(model_out[0]['labels'])  
            pred_boxes = out_boxes.cpu() * torch.tensor([[width,height,width,height]])  # [N, 4]
            #convert from cxcywh to xyxy
            pred_boxes = torch.cat([pred_boxes[:,:2] - pred_boxes[:,2:]/2,  pred_boxes[:,:2] + pred_boxes[:,2:]/2], dim=1) 
            # pred_logits = out_scores.cpu()  # [N, num_classes]
            pred_scores = out_scores.cpu()
            # pred_scores, pred_classes = pred_logits.max(dim=-1)  # [N], [N]
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
            vis = Visualizer(image_np, metadata=metadata, scale=4.0, instance_mode =  ColorMode.SEGMENTATION)
            instances = instances.to(torch.device('cpu'))
            vis_output = vis.draw_instance_predictions(instances.to(torch.device('cpu')))
            # 保存结果
            output_path = os.path.join(save_path)
            vis_output.save(output_path)
@torch.no_grad()
def visualize_feat_diff(images, dino_feats, cnn_feats, save_path):
    """
    Visualize DINO features as heatmaps overlayed on original images
    
    Args:
        images: Preprocessed images (tensor) in ImageNet normalization
        dino_feats: Dictionary containing DINO features
        target_layer: Target layer name for feature extraction
        gate_scores: Optional gate scores for similarity visualization
    """
    from torchvision import transforms
    from matplotlib import pyplot as plt
    import torch.nn.functional as F
    # Reverse ImageNet normalization to get original image range
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )
    original_images = inv_normalize(images).clamp(0, 1)
    original_images = original_images.cpu().numpy().transpose(0, 2, 3, 1)
    
    # Get the spatial dimensions of the original image
    img_h, img_w = original_images.shape[1:3]
    
    # Create figure for visualization
    num_images = len(images)
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5*num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_images):
        # Original image
        axes[i, 0].imshow(original_images[i])
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        # L2 Norm heatmap
        if True:

            l2_norm = (dino_feats - cnn_feats).norm(dim=1).unsqueeze(0)
            # Interpolate to original image size
            l2_heatmap = F.interpolate(l2_norm, 
                                     size=(img_h, img_w), 
                                     mode='bilinear',
                                     align_corners=False).squeeze().cpu().numpy()
            
            # Normalize heatmap
            l2_heatmap = (l2_heatmap - l2_heatmap.min()) / (l2_heatmap.max() - l2_heatmap.min() + 1e-8)
            
            # Overlay heatmap on image
            axes[i, 1].imshow(original_images[i])
            im = axes[i, 1].imshow(l2_heatmap,  alpha=0.5)
            im = axes[i, 1].imshow(l2_heatmap,  alpha=0.5)
            axes[i, 1].set_title(f'L2 Dist Heatmap ')
            axes[i, 1].axis('off')
            # plt.colorbar(im, ax=axes[i, 1])
        if True:
            cos_sim_map = 1-F.cosine_similarity(dino_feats, cnn_feats, dim=1).cpu()
            # Interpolate to original image size
            cos_heat_map = F.interpolate(cos_sim_map.unsqueeze(0), 
                                     size=(img_h, img_w), 
                                     mode='bilinear',
                                     align_corners=False).squeeze().numpy()
            
            # Normalize heatmap
            cos_heat_map = (cos_heat_map - cos_heat_map.min()) / (cos_heat_map.max() - cos_heat_map.min() + 1e-8)
            
            # Overlay heatmap on image
            axes[i, 2].imshow(original_images[i])
            im = axes[i, 2].imshow(cos_heat_map,  alpha=0.5)
            axes[i, 2].set_title(f'Cos Dist Heatmap')
            axes[i, 2].axis('off')

            # plt.colorbar(im, ax=axes[i, 1])
        
    
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path)
def fit_projection_matrix(A,B):
    # proj = torch.rand(A.shape[1],B.shape[1],requires_grad=True)
    proj = torch.nn.Sequential(nn.Conv2d(A.shape[1], B.shape[1], kernel_size=1), nn.ReLU(), nn.Conv2d(B.shape[1], B.shape[1],kernel_size=1)).cuda()
    A = A.cuda()
    B=  B.cuda()
    optimizer = torch.optim.SGD(params=proj.parameters(),lr=1)
    for i in range(1000):
        loss = 1- torch.nn.functional.cosine_similarity(proj(A), B, dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(i,loss)
    return proj
if __name__ == '__main__':
    cnn_features = []
    dino_features = []
    for i in range(100):
        path = f'cached_feats/{i}.pth'
        feats = torch.load(path)
        cnn_features.append(feats['cnn_features'])
        dino_features.append(feats['dino_features'])
    cnn_features = torch.cat(cnn_features, dim=0)
    dino_features = torch.cat(dino_features, dim=0)
    fit_projection_matrix(dino_features, cnn_features)
    print(cca_analysis(cnn_features[:10], dino_features[:10]))