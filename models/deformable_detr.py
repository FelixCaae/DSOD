import math
import copy

import torch
from torch.nn.functional import relu, interpolate
from torch import nn
from .grl import GradientReversal,DomainDiscriminator
from utils.distributed_utils import is_main_process,is_dist_avail_and_initialized
import torch.distributed as dist
from .vis_tools import * 
# 使用示例
class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class ConvBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Conv2d(n, k, kernel_size = 1) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        x_in = x
        for i, layer in enumerate(self.layers):
            x = relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SEAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(SEAttention, self).__init__()
        self.reduction = channels // reduction_ratio
        self.channels = channels
        
        # Squeeze-and-Excitation
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C, 1, 1]
            nn.Conv2d(channels, self.reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )
        torch.nn.init.constant_(self.se[3].bias, -2)
    
    def forward(self, x, y):
        se_weight = self.se(x)  # [B, C, 1, 1]
        return y * se_weight.expand_as(y), se_weight  # 通道权重
class SPAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(SPAttention, self).__init__()
        self.reduction = channels // reduction_ratio
        self.channels = channels
        
        # Squeeze-and-Excitation
        self.sp = nn.Sequential(
            nn.Conv2d(channels, self.reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.reduction, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x, y):
        sp_weight = self.sp(x)  # [B, 1, H, W]
        return y * sp_weight.expand_as(y)  # 通道权重
class DeformableDETR(nn.Module):

    def __init__(self,
                 backbone,
                 position_encoding,
                 transformer,
                 VFM_backbone=None,
                 VFM_transform=None,
                 VFM_channel=None,
                 VFM_backbone_2=None,
                 VFM_transform_2=None,
                 VFM_channel_2=None,
                 fuse_type = 'add',
                 enable_query_alignment=False,
                 enable_feature_alignment=False,
                 enable_encoder_alignment=False,
                 num_classes=9,
                 num_queries=300,
                 num_feature_levels=4):
        super().__init__()
        # Network hyperparameters
        self.hidden_dim = transformer.hidden_dim
        self.num_feature_levels = num_feature_levels
        self.num_queries = num_queries
        self.num_classes = num_classes
        # Backbone: multiscale outputs backbone network
        self.backbone = backbone
        # Build input projections
        self.input_proj = self._build_input_projections()
        # Position encoding
        self.position_encoding = position_encoding
        # Deformable transformer
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim * 2)
        self.transformer = transformer
        # Prediction of class and box
        self.class_embed = nn.Linear(self.hidden_dim, self.num_classes)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        # Initialize parameters
        self._init_params()
        self.class_embed = nn.ModuleList([self.class_embed for _ in range(transformer.decoder.num_layers)])
        self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(transformer.decoder.num_layers)])
        self.VFM_backbone = VFM_backbone
        self.VFM_backbone_2 = VFM_backbone_2
        self.VFM_transform = VFM_transform
        self.VFM_transform_2 = VFM_transform_2
        # self.learnable_weight = nn.Parameter(torch.tensor(0.))
        self.enable_feature_alignment = enable_feature_alignment
        self.enable_encoder_alignment = enable_encoder_alignment
        if VFM_backbone is not None:
            self.VFM_channel = VFM_channel
            if enable_query_alignment:
                self.inverse_proj = nn.Sequential(nn.Linear(256, self.VFM_channel), nn.ReLU(), nn.Linear(self.VFM_channel, self.VFM_channel))

            #build projection blocks
            self.fuse_type = fuse_type
            stride = [4,2,1]
            out_channels = [512,1024,2048]
            # self.projs = nn.ModuleList([ nn.Conv2d(768, i, kernel_size=1) for i in out_channels])
            self.projs = nn.ModuleList()
            enable_norm = False
            enable_simple_feature_pyramid = False
            for s,c in zip(stride,out_channels):
                if not enable_simple_feature_pyramid:
                    self.projs.append(nn.Sequential(nn.Conv2d(self.VFM_channel + c, c, kernel_size=1), nn.GroupNorm(32,c) if enable_norm else nn.Identity()))
                    # self.projs.append(nn.Conv2d(self.VFM_channel, c, kernel_size=1))
                    continue
                if s == 4:
                    self.projs.append( nn.Sequential(nn.ConvTranspose2d(self.VFM_channel, c, kernel_size=2,stride=2),
                                     nn.GELU(),
                                     nn.ConvTranspose2d(c, c, kernel_size=2,stride=2),
                                     nn.GroupNorm(32,c) if enable_norm else nn.Identity()))
                elif s == 2:
                    self.projs.append( nn.Sequential(nn.ConvTranspose2d(self.VFM_channel, c, kernel_size=2,stride=2),nn.GroupNorm(32,c) if enable_norm else nn.Identity()))
                elif s  ==1:
                    self.projs.append( nn.Sequential(nn.Conv2d(self.VFM_channel, c, kernel_size=1),nn.GroupNorm(32,c) if enable_norm else nn.Identity()))
                elif s == 0.5:
                    self.projs.append( nn.Sequential(nn.Conv2d(self.VFM_channel, 256, kernel_size=2, stride=2),nn.GroupNorm(32,c) if enable_norm else nn.Identity()))
            # self.inter_box_head =  nn.Linear(self.hidden_dim, self.num_classes)
            # self.inter_class_head = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
            if self.fuse_type == 'cat_add':
                self.cat_projs = nn.ModuleList()
                for c in out_channels:
                    self.cat_projs.append(nn.Sequential(nn.Conv2d(2*c, c, kernel_size=3,padding=1), nn.GELU(), nn.Conv2d(c,c,kernel_size=3,padding=1)))
            elif self.fuse_type == 'ch_add':
                self.gate_block = nn.ModuleList()
                for c in out_channels:
                    self.gate_block.append(nn.Sequential(nn.AdaptiveMaxPool2d((1,1)), nn.Conv2d(c, c, kernel_size=1,padding=0), nn.GELU(), nn.Conv2d(c,c,kernel_size=1,padding=0), nn.Sigmoid()))
            elif self.fuse_type == 'sp_ch_add':
                self.gate_block_1 = nn.ModuleList()
                self.gate_block_2 = nn.ModuleList()
                for c in out_channels:
                    self.gate_block_1.append(nn.Sequential(nn.AdaptiveMaxPool2d((1,1)), nn.Conv2d(c, c, kernel_size=1,padding=0), nn.GELU(), nn.Conv2d(c,c,kernel_size=1,padding=0), nn.Sigmoid()))
                    self.gate_block_2.append(nn.Sequential(nn.Conv2d(c, c, kernel_size=1,padding=0), nn.GELU(), nn.Conv2d(c,1,kernel_size=1,padding=0), nn.Sigmoid()))
            elif 'gate_add' in self.fuse_type:
                # for c in out_channels:
                c = 768
                self.gate_block = nn.Sequential(nn.Conv2d(c, c//2, kernel_size=1,padding=0), nn.GELU(), nn.Conv2d(c//2, num_classes,kernel_size=1,padding=0), nn.Sigmoid())
            elif self.fuse_type == 'grl_gate_add':
                self.gate_block = nn.ModuleList()
                for c in out_channels:
                    self.gate_block.append(nn.Sequential(nn.Conv2d(c, c, kernel_size=1,padding=0), nn.GELU(), nn.Conv2d(c,1,kernel_size=1,padding=0), nn.Sigmoid()))
                self.grl_block = GradientReversal()
                # self.discriminator = DomainDiscriminator(256)
                # self.rev_proj = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1), nn.GELU(), nn.Conv2d(256,768,kernel_size=1))
            elif self.fuse_type == 'residual_add':
                self.gate_block = nn.ModuleList()
                self.residual_projs = nn.ModuleList()
                for c in out_channels:
                    self.gate_block.append(nn.Sequential(nn.Conv2d(c, c, kernel_size=1,padding=0), nn.GELU(), nn.Conv2d(c,1,kernel_size=1,padding=0), nn.Sigmoid()))
                for c in out_channels:
                    self.residual_projs.append(nn.Sequential(nn.Conv2d(c, 1024, kernel_size=1,padding=0), nn.GELU(), nn.Conv2d(1024,c,kernel_size=1,padding=0)))
                
            #maybe some fusion block
            elif self.fuse_type == 'se':
                self.se_blocks = nn.ModuleList([SEAttention(i) for i in out_channels])
            elif self.fuse_type == 'sp':
                self.sp_blocks = nn.ModuleList([SPAttention(i) for i in out_channels])
            elif self.fuse_type == 'blending':
                self.fuse_blocks = nn.ModuleList([ConvBlock(i, 1024, i, 2) for i in out_channels])
            if enable_feature_alignment:
                self.cnn_to_dino_proj =  nn.ModuleList([ConvBlock(i, 1024, self.VFM_channel, 2) for i in out_channels])
            if enable_encoder_alignment:
                self.encoder_to_dino_proj = nn.ModuleList([ConvBlock(256, 1024, self.VFM_channel, 2) for i in range(4)])
           
            self.dino_factor = nn.Parameter(torch.tensor(0.0), requires_grad=False)
    @torch.no_grad()
    def forward_clip(self, data, output_shape=None, keyword='image_weak', max_length=336, patch_size=16):
        # data is an image tensor [B,C,H,W]
        transforms = self.VFM_transform_2
        model_clip = self.VFM_backbone_2
        feat_size, out_dim, input_size = [14, 768, 224]
        # feat_size, out_dim, input_size = [24, 1024, 336]
        import torch.nn.functional as F
        
        b, c, h, w = data.shape
        x = data        
        # CLIP通常使用固定输入尺寸，但这里保持与原函数类似的动态调整逻辑

        # 调整图像尺寸
        x = F.interpolate(x, (input_size, input_size), mode='bilinear')
        # CLIP预处理（归一化）
        if transforms is not None:
            x = transforms(x)
            x = x.cuda()
        # CLIP视觉编码器前向传播
        x = model_clip.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([model_clip.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + model_clip.positional_embedding.to(x.dtype)
        x = model_clip.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = model_clip.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x_ = x
        x = model_clip.ln_post(x[:, 1:, :])

        # if model_clip.proj is not None:
        #     x = x @ model_clip.proj
        feat = x
        # feat = model_clip(x)
        feat = feat.reshape(1, feat_size, feat_size, out_dim).permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        
        # 调整到目标输出尺寸
        if output_shape is not None:
            feat = F.interpolate(feat, output_shape, mode='bilinear')
        
        return feat
    @torch.no_grad()
    def forward_sam(self, data, output_shape=None, keyword='image_weak', max_length=560, patch_size=14,):
        #data is an image tensor [B,C,H,W] 
        transforms = self.VFM_transform
        model_dino = self.VFM_backbone
        import torch.nn.functional as F
        #resize image to the target size
        x = transforms(data).half().cuda()
        x = F.interpolate(x, (1024, 1024), mode='bilinear')
        #forward
        feat_dict = model_dino(x).float()
        
        #reshape to B,C,H,W
        feat = feat_dict.reshape(1, -1, 64, 64)
        if output_shape is not None:
            #reisze to outshape if given
            feat = F.interpolate(feat, output_shape, mode='bilinear')   
        return feat

    @torch.no_grad()
    def forward_dino(self, data, output_shape=None, keyword='image_weak', max_length=560, patch_size=14,):
        #data is an image tensor [B,C,H,W] 
        transforms = self.VFM_transform
        model_dino = self.VFM_backbone
        import torch.nn.functional as F
        output = []
        #compute the target size accoding to the original aspect ratio
        b,c,h,w = data.shape
        x = data
        if h <= w:
            s_w = max_length
            s_h = s_w * (h/w)
        else:
            s_h = max_length
            s_w = s_h * (w/h)
        #adjust slightly to fit the patch size
        s_h = (s_h // patch_size) * patch_size
        s_w = (s_w // patch_size) * patch_size
        s_h, s_w = int(s_h), int(s_w)

        #resize image to the target size
        x = F.interpolate(x, (s_h, s_w), mode='bilinear')

        #forward
        feat_dict = model_dino.forward_features(x)
        
        #reshape to B,C,H,W
        feat = feat_dict['x_norm_patchtokens'].reshape(b, s_h//patch_size, s_w//patch_size,-1).permute(0,3,1,2)
        if output_shape is not None:
            #reisze to outshape if given
            feat = F.interpolate(feat, output_shape, mode='bilinear')   
        return feat

    def fuse(self, features_cnn, features_dino,debug=True):
        import torch.nn.functional as F
        dino_feats = {}

        for target_layer in range(len(features_cnn)):
            dino_feats[f'cnn_feats_{target_layer}'] = features_cnn[target_layer]
            if self.fuse_type == 'add':
                dino_proj_feats = F.interpolate(self.projs[target_layer](features_dino), features_cnn[target_layer].shape[2:4], mode='bilinear') 
                features_cnn[target_layer] = features_cnn[target_layer] + dino_proj_feats * self.dino_factor  # * features_cnn[target_layer].norm()/dino_proj_feats.norm() #* torch.exp(self.learnable_weight)
            elif self.fuse_type == 'cat':
                f = F.interpolate(features_dino, features_cnn[target_layer].shape[2:4], mode='bilinear') 
                # dino_proj_feats = F.interpolate(self.projs[target_layer](features_dino), features_cnn[target_layer].shape[2:4], mode='bilinear') 
                # features_cnn[target_layer] = features_cnn[target_layer] + dino_proj_feats * self.dino_factor  
                dino_proj_feats = self.projs[target_layer](torch.cat([f, features_cnn[target_layer]], dim=1)) #features_cnn[target_layer] + dino_proj_feats * self.dino_factor  
                features_cnn[target_layer] = features_cnn[target_layer] + dino_proj_feats * self.dino_factor
            elif self.fuse_type == 'adaptive_add':
                dino_proj_feats = F.interpolate(self.projs[target_layer](features_dino), features_cnn[target_layer].shape[2:4], mode='bilinear') 
                with torch.no_grad():
                    scale = (features_cnn[target_layer].norm(dim=1)/dino_proj_feats.norm(dim=1)).mean()
                if is_dist_avail_and_initialized():
                    if not isinstance(scale, torch.Tensor):
                        raise TypeError(f"Expected tensor, got {type(scale)}")
                    dist.all_reduce(scale, op=dist.ReduceOp.SUM)
                scale /= dist.get_world_size()
                if not hasattr(self, f'scale'):
                    self.register_buffer('scale', torch.tensor([1.,1.,1.],requires_grad=False).cuda())
                else:
                    self.scale[target_layer] = self.scale[target_layer] * 0.99 + scale * 0.01
                features_cnn[target_layer] = features_cnn[target_layer] + dino_proj_feats * self.dino_factor * self.scale[target_layer].clone()

            elif self.fuse_type == 'mul':
                #blending
                dino_proj_feats = F.interpolate(self.projs[target_layer](features_dino), features_cnn[target_layer].shape[2:4], mode='bilinear') 
                #norm
                # cnn_feat_norm = feature_cnn[target_layer].norm(dim=1).mean()
                # dino_feat_norm = 
                dino_feats.update({f'cnn_feats_{target_layer}':features_cnn[target_layer],  f'dino_feats_{target_layer}':dino_proj_feats})
                features_cnn[target_layer] = features_cnn[target_layer] * (1 + dino_proj_feats * self.dino_factor)  # * features_cnn[target_layer].norm()/dino_proj_feats.norm() #* torch.exp(self.learnable_weight)
                # dino_feats.update({f'l2_norm_{target_layer}':dino_proj_feats.norm(dim=1)})
            elif self.fuse_type == 'grl_gate_add':
                #blending
                # import pdb;pdb.set_trace()
                dino_proj_feats = F.interpolate(self.projs[target_layer](features_dino), features_cnn[target_layer].shape[2:4], mode='bilinear') 
                gate_scores = self.grl_block(self.gate_block[target_layer](dino_proj_feats)) 
                features_cnn[target_layer] = features_cnn[target_layer] + dino_proj_feats * self.dino_factor * gate_scores  / gate_scores.mean()
                dino_feats[f'metric_sim_score_{target_layer}'] = gate_scores[0][0].mean()
                # if self.train:
                #     import torch.nn.functional as F
                #     # grl_feats_cnn = self.grl_block(features_cnn[2])
                #     grl_feats_cnn = features_cnn[2]
                #     grl_feats_dino = self.grl_block(dino_proj_feats)
                #     dis_score_cnn = self.discriminator(grl_feats_cnn.detach())
                #     dis_score_dino = self.discriminator(grl_feats_dino)
                #     dino_feats['dis_cnn_loss'] = F.binary_cross_entropy_with_logits(dis_score_cnn, torch.zeros_like(dis_score_cnn))
                #     dino_feats['dis_dino_loss'] = F.binary_cross_entropy_with_logits(dis_score_dino, torch.ones_like(dis_score_cnn))
                #     # rev_feats_dino = self.rev_proj(grl_feats_dino)
                    # dino_feats['construct_loss'] = F.mse_loss(F.interpolate(rev_feats_dino, features_dino.shape[2:4], mode='bilinear'), features_dino)
            elif self.fuse_type == 'gate_add':
                dino_proj_feats = F.interpolate(self.projs[target_layer](features_dino), features_cnn[target_layer].shape[2:4], mode='bilinear') 
                # gate_scores = self.gate_block[target_layer](dino_proj_feats) 
                gate_scores = self.gate_block(features_dino) 
                dino_feats[f'sim_score_{target_layer}'] = gate_scores[0]
                gate_scores = F.interpolate(gate_scores, features_cnn[target_layer].shape[2:4], mode='bilinear')
                features_cnn[target_layer] = features_cnn[target_layer] + dino_proj_feats * (self.dino_factor + (gate_scores.max(dim=1,keepdims=True)[0]) *  self.fg_weight)
            elif self.fuse_type == 'gate_add_classaware':
                dino_proj_feats = F.interpolate(self.projs[target_layer](features_dino), features_cnn[target_layer].shape[2:4], mode='bilinear') 
                # gate_scores = self.gate_block[target_layer](dino_proj_feats) 
                gate_scores = self.gate_block(features_dino) 
                gate_scores = F.interpolate(gate_scores, features_cnn[target_layer].shape[2:4], mode='bilinear')
                weight_map = torch.zeros(features_cnn[target_layer].shape[2:4], device=features_cnn[target_layer].device)
                gate_scores = gate_scores.softmax(dim=1)
                for i in range(self.num_classes):
                    weight_map = weight_map  + gate_scores[:,i,:,:] * self.weight_dict[i]
                weight_map = weight_map 
                features_cnn[target_layer] = features_cnn[target_layer] + dino_proj_feats * weight_map
                dino_feats[f'sim_score_{target_layer}'] = gate_scores[0]
                dino_feats[f'weight_map'] = weight_map
                
                # plt.figure()
                # plt.imshow(weight_map.cpu()[0])  
                # plt.colorbar()
                # plt.savefig('test.jpg')
            elif self.fuse_type == 'gate_add_cnn':
                dino_proj_feats = F.interpolate(self.projs[target_layer](features_dino), features_cnn[target_layer].shape[2:4], mode='bilinear') 
                gate_scores = self.gate_block[target_layer](dino_proj_feats) 
                features_cnn[target_layer] = features_cnn[target_layer] + dino_proj_feats * self.dino_factor *  ( 1 +  gate_scores)
                dino_feats[f'sim_score_{target_layer}'] = gate_scores[0][0]

            elif self.fuse_type == 'impr_gate_add':
                dino_proj_feats = F.interpolate(self.projs[target_layer](features_dino), features_cnn[target_layer].shape[2:4], mode='bilinear') 
                gate_scores = self.gate_block[target_layer](dino_proj_feats) 
                features_cnn[target_layer] = features_cnn[target_layer] + dino_proj_feats * self.dino_factor * gate_scores / gate_scores.mean()
                dino_feats[f'sim_score_{target_layer}'] = gate_scores[0]
            elif self.fuse_type == 'ch_add':
                #blending
                # import pdb;pdb.set_trace()
                dino_proj_feats = F.interpolate(self.projs[target_layer](features_dino), features_cnn[target_layer].shape[2:4], mode='bilinear') 
                #norm
                # cnn_feat_norm = feature_cnn[target_layer].norm(dim=1).mean()
                # dino_feat_norm = 
                # cat_feats = torch.cat([features_cnn[target_layer], residual_feats],dim=1) #* self.dino_factor  # * features_cnn[target_layer].norm()/dino_proj_feats.norm() #* torch.exp(self.learnable_weight)
                gate_scores = self.gate_block[target_layer](dino_proj_feats) 
                features_cnn[target_layer] = features_cnn[target_layer] + dino_proj_feats * self.dino_factor * gate_scores /gate_scores.mean()
            elif self.fuse_type == 'sp_ch_add':
                dino_proj_feats = F.interpolate(self.projs[target_layer](features_dino), features_cnn[target_layer].shape[2:4], mode='bilinear') 
                gate_scores_1 = self.gate_block_1[target_layer](dino_proj_feats) 
                gate_scores_1 = gate_scores_1/ gate_scores_1.mean()
                gate_scores_2 = self.gate_block_2[target_layer](dino_proj_feats)
                gate_scores_2 = gate_scores_2 /gate_scores_2.mean()
                features_cnn[target_layer] = features_cnn[target_layer] + dino_proj_feats * self.dino_factor * gate_scores_1 * gate_scores_2
            elif self.fuse_type == 'residual_add':
                #blending
                # import pdb;pdb.set_trace()
                cnn_proj_feats =  (features_cnn[target_layer])
                dino_proj_feats = F.interpolate(self.projs[target_layer](features_dino), cnn_proj_feats.shape[2:4], mode='bilinear') 
                residual_feats = dino_proj_feats - cnn_proj_feats
                residual_feats_proj = self.residual_projs[target_layer](residual_feats)
                #norm
                # cnn_feat_norm = feature_cnn[target_layer].norm(dim=1).mean()
                # dino_feat_norm = 
                # cat_feats = torch.cat([features_cnn[target_layer], residual_feats],dim=1) #* self.dino_factor  # * features_cnn[target_layer].norm()/dino_proj_feats.norm() #* torch.exp(self.learnable_weight)
                gate_scores = self.gate_block[target_layer](residual_feats) 
                features_cnn[target_layer] = features_cnn[target_layer] + residual_feats_proj * self.dino_factor * gate_scores
                dino_feats[f'l2_norm_{target_layer}'] = (features_cnn[target_layer] - dino_proj_feats).norm(dim=1, p=2)
                dino_feats[f'sim_score_{target_layer}'] = gate_scores[0][0]
        
            elif self.fuse_type == 'cat_add':
                #blending
                # import pdb;pdb.set_trace()
                dino_proj_feats = F.interpolate(self.projs[target_layer](features_dino), features_cnn[target_layer].shape[2:4], mode='bilinear') 
                #norm
                # cnn_feat_norm = feature_cnn[target_layer].norm(dim=1).mean()
                # dino_feat_norm = 
                cat_feats = torch.cat([features_cnn[target_layer], dino_proj_feats],dim=1) #* self.dino_factor  # * features_cnn[target_layer].norm()/dino_proj_feats.norm() #* torch.exp(self.learnable_weight)
                cat_feats = self.cat_projs[target_layer](cat_feats) 
                features_cnn[target_layer] = features_cnn[target_layer] + cat_feats * self.dino_factor
                dino_feats[f'dino_feats_{target_layer}'] = dino_proj_feats
    
            elif self.fuse_type == 'attn':
                B,C,H,W = features_cnn[target_layer].shape
                dino_feats_proj = self.proj(features_dino)
                query = features_cnn[target_layer].flatten(2).permute(2,0,1) #B,C,H,W -> HW,B,C
                key = value = dino_feats_proj.flatten(2).permute(2,0,1)
                attn_out = self.fuse_attn(query, key, value)[0]
                attn_out = self.fuse_mlp(attn_out)
                # import pdb;pdb.set_trace()
                attn_out = self.fuse_ln(value + attn_out)
                attn_out = attn_out.permute(1,2,0).view(B,C,H,W)
                features_cnn[target_layer] = features_cnn[target_layer] + attn_out * self.dino_factor
            else:
                raise  NotImplementedError
            dino_feats[f'dino_feats_{target_layer}'] = dino_proj_feats
        return features_cnn, dino_feats

    def _build_input_projections(self):
        input_proj_list = []
        if self.num_feature_levels > 1:
            for i in range(self.backbone.num_outputs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(self.backbone.num_channels[i], self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
            in_channels = self.backbone.num_channels[-1]
            for _ in range(self.num_feature_levels - self.backbone.num_outputs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
                in_channels = self.hidden_dim
        else:
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(self.backbone.num_channels[0], self.hidden_dim, kernel_size=1),
                nn.GroupNorm(32, self.hidden_dim),
            ))
        return nn.ModuleList(input_proj_list)

    def _init_params(self):
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)

    @staticmethod
    def inverse_sigmoid(x, eps=1e-5):
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1 / x2)
    def forward(self, images, masks, dino_features=None):
        # Backbone forward
        #images: B,C,H,W
        #masks: B,H,W
        orig_features = features = self.backbone(images)  

        #check whether has a dino backbone
        if self.VFM_backbone is not None :
            dino_features = self.forward_dino(images, output_shape=None, max_length = max(images.shape)//2)#features[target_layer].shape[2:])
        if self.VFM_backbone_2 is not None:
            clip_features = self.forward_clip(images, output_shape=dino_features.shape[2:], max_length = max(images.shape)//2)
            dino_features = dino_features + clip_features
        # import pdb;pdb.set_trace()
            #the stride of dino features should be 2 * 14 ~ 28 and is close to 32 of the CNN features
            # with torch.no_grad():
            #     import torch.nn.functional as F
            #     dict_feat = {"cnn_features":features[2].cpu(), 'dino_features':F.interpolate(dino_features.cpu(), features[2].shape[2:], mode='bilinear')}
            #     torch.save(dict_feat, f"{iter}.pth")
                # caa_results = cca_analysis(features[0], F.interpolate(dino_features, features[0].shape[2:], mode='bilinear'), n_components=10)
            # caa_results = cca_analysis(features[1], F.interpolate(dino_features, features[1].shape[2:], mode='bilinear'), n_components=10)

        #check whether dino features is usable
        if dino_features is not None and self.VFM_backbone is not None:
            features, vis_dict = self.fuse(features, dino_features)
        # Prepare input features for transformer
        else:
            vis_dict = {}
        
            # compute cnn to dino alignment by using the regularization effect on feature-level
        if self.enable_feature_alignment and dino_features is not None:
            import torch.nn.functional as F
            loss_feat_align = 0 
            for layer,feat in enumerate(orig_features):
                inv_feat = self.cnn_to_dino_proj[layer](feat)
                loss_mse = F.mse_loss(F.interpolate(inv_feat, dino_features.shape[2:],mode='bilinear'), dino_features, reduction='none')
                # loss_cos = 1 - F.cosine_similarity(F.interpolate(inv_feat, dino_features.shape[2:],mode='bilinear'), dino_features, dim=1)
                vis_dict[f'loss_feat_align_{layer}'] = loss_mse
        if hasattr(self, 'dino_factor'):
            vis_dict['metric_dino_factor'] = self.dino_factor

        debug = False
        #debug or visualization code
        if torch.rand(1)< 1e-3 and debug:
            # visualize_dino_heatmaps(images, vis_dict, 0)
            # visualize_dino_heatmaps(images, vis_dict, 1)
            # visualize_dino_heatmaps(images, vis_dict, 2)

            # visualize_pca(dino_features, images, 'pca_dino.png')
            # visualize_feature_distribution(vis_dict['cnn_feats_0'], vis_dict['dino_feats_0'], features[0], images, 0)
            # visualize_feature_distribution(vis_dict['cnn_feats_1'], vis_dict['dino_feats_1'], features[1], images, 1)
            visualize_feature_distribution(vis_dict['cnn_feats_2'], vis_dict['dino_feats_2'], features[2], images, 2)
            # visualize_feat_diff(images,vis_dict['dino_feats_0'], vis_dict['cnn_feats_0'], 'feat_diff_0_trained.png')
            # visualize_feat_diff(images,vis_dict['dino_feats_1'], vis_dict['cnn_feats_1'], 'feat_diff_1_trained.png')
            # visualize_feat_diff(images,vis_dict['dino_feats_2'], vis_dict['cnn_feats_2'], 'feat_diff_2_trained.png')

            # import pdb;pdb.set_trace()
            # images is a tensor of preprocess image
            # reverse images to original range by ImageNet standard
            # dino_feats[f'l2_norm_{target_layer}'] = (features_cnn[target_layer] - dino_proj_feats).norm(dim=1, p=2).cpu()
            # dino_feats[f'sim_score_{target_layer}'] = gate_scores[0][0].cpu()
            # interpolate these two hot map and overlap it with images
            # visualize it with plt
            if 'sim_score_0' in vis_dict:
                # for target_layer in range(3):
                for target_layer in range(3):
                    class_names = ['person', 'car', 'train', 'rider', 'truck', 'motorcycle', 'bicycle', 'bus']
                    for class_ in range(8):
                        visualize_heat_map(images, vis_dict[f'sim_score_{target_layer}'][class_+1], f'sim_map_{target_layer}_{class_names[class_]}.jpg')
                        vis_dict[f'metric_sim_score_{target_layer}'] = vis_dict[f'sim_score_{target_layer}'].mean()
                    visualize_heat_map(images, vis_dict[f'weight_map'][0], f'weight_map_{target_layer}.jpg')
                    
            if debug:
                pass

                #compute cca score
            with torch.no_grad():
                import torch.nn.functional as F
                for layer in range(3):
                    vis_dict[f'metric_cos_fuse_cnn_{layer}'] = torch.round(F.cosine_similarity(vis_dict[f'cnn_feats_{layer}'], features[layer], dim=1).mean(),decimals=3)
                    vis_dict[f'metric_cos_fuse_dino_{layer}'] = torch.round(F.cosine_similarity(vis_dict[f'dino_feats_{layer}'], features[layer], dim=1).mean(),decimals=3)
                    vis_dict[f'metric_cos_cnn_dino_{layer}'] = torch.round(F.cosine_similarity(vis_dict[f'dino_feats_{layer}'],vis_dict[f'cnn_feats_{layer}'], dim=1).mean(),decimals=3)
                    # print(vis_dict[f'metric_cos_fuse_dino_{layer}'])
                #         print(analyze_orthogonality(vis_dict[f'cnn_feats_{layer}'].cpu(), vis_dict[f'dino_feats_{layer}'].cpu(),  features[layer].cpu()))
                #         import pdb;pdb.set_trace()

        src_list, mask_list = [], []
        for i, feature in enumerate(features):
            src = self.input_proj[i](feature)
            mask = interpolate(masks[None].float(), size=feature.shape[-2:]).to(torch.bool)[0]
            src_list.append(src)
            mask_list.append(mask)
        if self.num_feature_levels > len(features):
            for i in range(len(features), self.num_feature_levels):
                src = self.input_proj[i](features[-1]) if i == len(features) else self.input_proj[i](src_list[-1])
                mask = interpolate(masks[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                src_list.append(src)
                mask_list.append(mask)


        pos_list = [self.position_encoding(src, mask) for src, mask in zip(src_list, mask_list)]
        query_embeds = self.query_embed.weight
        # Transformer forward
        hs, init_reference, inter_references, _, _, inter_memory, inter_object_query = self.transformer(
            src_list,
            mask_list,
            pos_list,
            query_embeds
        )
        if self.enable_encoder_alignment:
            import torch.nn.functional as F
            loss_encoder_align = 0 
            start_idx = 0
            for layer,feat in enumerate(src_list):
                h,w = feat.shape[2:]
                inter_feat = inter_memory[0, 0, start_idx:start_idx + h*w].reshape(1, h, w, -1).permute(0,3,1,2) #  b,c,h,w
                inv_feat = self.encoder_to_dino_proj[layer](inter_feat)
                loss_mse = F.mse_loss(F.interpolate(inv_feat, dino_features.shape[2:],mode='bilinear'), dino_features)
                loss_encoder_align += loss_mse
                start_idx += h*w 
            vis_dict['loss_encoder_align'] = loss_encoder_align

        # Prepare outputs
        outputs_classes, outputs_coords = [], []
        for lvl in range(hs.shape[0]):
            outputs_class = self.class_embed[lvl](hs[lvl])
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference = self.inverse_sigmoid(reference)
            tmp = self.bbox_embed[lvl](hs[lvl])
            tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        out = {
            'logit_all': outputs_class,
            'boxes_all': outputs_coord,
            'features': [f for f in features],
        }
        if self.VFM_backbone is not None:
            out['dino_features'] = dino_features
            # out['dino_features_proj'] = dino_feats_proj
        if hasattr(self, 'inverse_proj'):
            out['query_embed'] = self.inverse_proj(hs)
        if hasattr(self, 'dino_factor'):
            out['dino_factor'] = self.dino_factor
        out.update(vis_dict)
        # out['dis_cnn_loss'] = vis_dict['dis_cnn_loss']
        # out['dis_dino_loss'] = vis_dict['dis_dino_loss']
        # out['']
        # out['sim_score'] = [vis_dict['sim_score_0'], vis_dict['sim_score_1'], vis_dict['sim_score_2']]
        return out
