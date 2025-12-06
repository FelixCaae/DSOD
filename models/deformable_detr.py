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
        x_out = x + x_in
        return x_out


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
                 dino_backbone=None,
                 dino_transform=None,
                 fuse_type = 'add',
                 enable_query_alignment=False,
                 enable_feature_alignment=False,
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
        self.dino_backbone = dino_backbone
        # self.learnable_weight = nn.Parameter(torch.tensor(0.))
        if dino_backbone is not None:
            # import pdb;pdb.set_trace()
            for param in dino_backbone.parameters():
                param.requires_grad = False
            if enable_query_alignment:
                self.inverse_proj = nn.Sequential(nn.Linear(256, 768), nn.ReLU(), nn.Linear(768, 768))

            #build projection blocks
            self.fuse_type = fuse_type
            out_channel = 768
            stride = [4,2,1]
            out_channels = [512,1024,2048]
            # self.projs = nn.ModuleList([ nn.Conv2d(768, i, kernel_size=1) for i in out_channels])
            self.projs = nn.ModuleList()
            enable_norm = False
            enable_simple_feature_pyramid = False
            for s,c in zip(stride,out_channels):
                if not enable_simple_feature_pyramid:
                    self.projs.append(nn.Sequential(nn.Conv2d(out_channel, c, kernel_size=1), nn.GroupNorm(32,c) if enable_norm else nn.Identity()))
                    continue
                if s == 4:
                    self.projs.append( nn.Sequential(nn.ConvTranspose2d(out_channel, c, kernel_size=2,stride=2),
                                     nn.GELU(),
                                     nn.ConvTranspose2d(c, c, kernel_size=2,stride=2),
                                     nn.GroupNorm(32,c) if enable_norm else nn.Identity()))
                elif s == 2:
                    self.projs.append( nn.Sequential(nn.ConvTranspose2d(out_channel, c, kernel_size=2,stride=2),nn.GroupNorm(32,c) if enable_norm else nn.Identity()))
                elif s  ==1:
                    self.projs.append( nn.Sequential(nn.Conv2d(out_channel, c, kernel_size=1),nn.GroupNorm(32,c) if enable_norm else nn.Identity()))
                elif s == 0.5:
                    self.projs.append( nn.Sequential(nn.Conv2d(out_channel, 256, kernel_size=2, stride=2),nn.GroupNorm(32,c) if enable_norm else nn.Identity()))
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
                self.gate_block = nn.ModuleList()
                for c in out_channels:
                    self.gate_block.append(nn.Sequential(nn.Conv2d(c, c, kernel_size=1,padding=0), nn.GELU(), nn.Conv2d(c,1,kernel_size=1,padding=0), nn.Sigmoid()))
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

            self.dino_transform = dino_transform
            self.dino_factor = nn.Parameter(torch.tensor(0.1), requires_grad=False)
            self.dino_step = 0.01
    @torch.no_grad()
    def forward_dino(self, data, output_shape=None, keyword='image_weak', max_length=560, patch_size=14,):
        #data is an image tensor [B,C,H,W] 
        transforms = self.dino_transform
        model_dino = self.dino_backbone
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
                gate_scores = self.gate_block[target_layer](dino_proj_feats) 
                features_cnn[target_layer] = features_cnn[target_layer] + dino_proj_feats * (self.dino_factor + gate_scores * 0.2)
                dino_feats[f'sim_score_{target_layer}'] = gate_scores[0][0]
            elif self.fuse_type == 'gate_add_cnn':
                dino_proj_feats = F.interpolate(self.projs[target_layer](features_dino), features_cnn[target_layer].shape[2:4], mode='bilinear') 
                gate_scores = self.gate_block[target_layer](dino_proj_feats) 
                features_cnn[target_layer] = features_cnn[target_layer] + dino_proj_feats * self.dino_factor *  ( 1 +  gate_scores)
                dino_feats[f'sim_score_{target_layer}'] = gate_scores[0][0]

            elif self.fuse_type == 'impr_gate_add':
                dino_proj_feats = F.interpolate(self.projs[target_layer](features_dino), features_cnn[target_layer].shape[2:4], mode='bilinear') 
                gate_scores = self.gate_block[target_layer](dino_proj_feats) 
                features_cnn[target_layer] = features_cnn[target_layer] + dino_proj_feats * self.dino_factor * gate_scores / gate_scores.mean()
                dino_feats[f'sim_score_{target_layer}'] = gate_scores[0][0]
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
                dino_feats.append(dino_proj_feats)
    
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
    def forward(self, images, masks):
        # Backbone forward
        features = self.backbone(images)  
        dino_features = None

        if self.dino_backbone is not None:
            dino_features = self.forward_dino(images, output_shape=None, max_length = max(images.shape)//2)#features[target_layer].shape[2:])
            #the stride of dino features should be 2 * 14 ~ 28 and is close to 32 of the CNN features
            # with torch.no_grad():
            #     import torch.nn.functional as F
            #     dict_feat = {"cnn_features":features[2].cpu(), 'dino_features':F.interpolate(dino_features.cpu(), features[2].shape[2:], mode='bilinear')}
            #     torch.save(dict_feat, f"{iter}.pth")
                # caa_results = cca_analysis(features[0], F.interpolate(dino_features, features[0].shape[2:], mode='bilinear'), n_components=10)
            # caa_results = cca_analysis(features[1], F.interpolate(dino_features, features[1].shape[2:], mode='bilinear'), n_components=10)
            features, vis_dict = self.fuse(features, dino_features)

            debug = False
            if torch.rand(1)<0.00 or debug:
                # visualize_dino_heatmaps(images, vis_dict, 0)
                # visualize_dino_heatmaps(images, vis_dict, 1)
                # visualize_dino_heatmaps(images, vis_dict, 2)

                # visualize_pca(dino_features, images, 'pca_dino.png')
                # visualize_feature_distribution(vis_dict['cnn_feats_0'], vis_dict['dino_feats_0'], features[0], images, 0)
                # visualize_feature_distribution(vis_dict['cnn_feats_1'], vis_dict['dino_feats_1'], features[1], images, 1)
                # visualize_feature_distribution(vis_dict['cnn_feats_2'], vis_dict['dino_feats_2'], features[2], images, 2)
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
                    for target_layer in range(3):
                        visualize_heat_map(images, vis_dict[f'sim_score_{target_layer}'], f'sim_map_{target_layer}.jpg')
                        vis_dict[f'metric_sim_score_{target_layer}'] = vis_dict[f'sim_score_{target_layer}'].mean()
                if debug:
                    import pdb;pdb.set_trace()
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
        # Prepare input features for transformer
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
        if self.dino_backbone is not None:
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
