import math
import copy

import torch
from torch.nn.functional import relu, interpolate
from torch import nn


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
            out_channels = [512,1024,2048]
            self.projs = nn.ModuleList([ nn.Conv2d(768, i, kernel_size=1) for i in out_channels])
            
            #maybe some fusion block
            if self.fuse_type == 'se':
                self.se_blocks = nn.ModuleList([SEAttention(i) for i in out_channels])
            elif self.fuse_type == 'sp':
                self.sp_blocks = nn.ModuleList([SPAttention(i) for i in out_channels])
            elif self.fuse_type == 'blending':
                self.fuse_blocks = nn.ModuleList([ConvBlock(i, 1024, i, 2) for i in out_channels])

            self.dino_transform = dino_transform
            self.dino_factor = nn.Parameter(torch.tensor(0.01), requires_grad=False)
            self.dino_step = 0.01
    @torch.no_grad()
    def forward_dino(self, data, output_shape=None, keyword='image_weak', max_length=560, patch_size=14,):
        #data is an image tensor [B,C,H,W] 
        transforms = self.dino_transform
        model_dino = self.dino_backbone
        assert max_length % patch_size == 0
        import torch.nn.functional as F
        output = []
        h,w = data.shape[2:4]
        x = data
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
        feat = model_dino.forward_features(x)
        feat = feat['x_norm_patchtokens'].reshape(1, s_h//patch_size, s_w//patch_size,-1).permute(0,3,1,2)
        if output_shape is None:
            output_shape = s_h//patch_size, s_w//patch_size
        feat = F.interpolate(feat, output_shape, mode='bilinear')   
        return feat

    def fuse(self, features_cnn, features_dino, target_layer=1):
        import torch.nn.functional as F
        dino_feats = []
        if self.fuse_type == 'add':
            for target_layer in range(3):
                #blending
                dino_proj_feats = F.interpolate(self.projs[target_layer](features_dino), features_cnn[target_layer].shape[2:4], mode='bilinear') 
                features_cnn[target_layer] = features_cnn[target_layer] + dino_proj_feats * self.dino_factor #* torch.exp(self.learnable_weight)
                dino_feats.append(dino_proj_feats)
        elif self.fuse_type == 'blending':
            for target_layer in range(3):
                #blending
                # self.projs[target_layer]
                # import pdb;pdb.set_trace()
                dino_proj_feats = F.interpolate(self.projs[target_layer](features_dino), features_cnn[target_layer].shape[2:4], mode='bilinear') 
                features_cnn[target_layer] = features_cnn[target_layer] + dino_proj_feats * self.dino_factor * torch.exp(self.learnable_weight)
                dino_feats.append(dino_proj_feats)
        elif self.fuse_type == 'se':
            for target_layer in range(3):
                #blending
                dino_proj_feats = F.interpolate(self.projs[target_layer](features_dino), features_cnn[target_layer].shape[2:4], mode='bilinear') 
                features_cnn[target_layer] = features_cnn[target_layer] + dino_proj_feats * self.dino_factor   
                features_cnn[target_layer], attn_weight = self.se_blocks[target_layer](features_cnn[target_layer], features_cnn[target_layer])
                dino_feats.append(dino_proj_feats) 
                # print(target_layer, attn_weight.mean())
        elif self.fuse_type == 'sp':
            for target_layer in range(3):
                #blending
                dino_proj_feats = F.interpolate(self.projs[target_layer](features_dino), features_cnn[target_layer].shape[2:4], mode='bilinear') 
                dino_proj_feats = self.sp_blocks[target_layer](features_cnn[target_layer], dino_proj_feats)
                features_cnn[target_layer] = features_cnn[target_layer] + dino_proj_feats * self.dino_factor    

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
            target_layer = 1
            dino_features = self.forward_dino(images, output_shape=features[target_layer].shape[2:])
            features, dino_feats_proj = self.fuse(features, dino_features, target_layer=target_layer)
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
            out['dino_features_proj'] = dino_feats_proj
        if hasattr(self, 'inverse_proj'):
            out['query_embed'] = self.inverse_proj(hs)
        if hasattr(self, 'dino_factor'):
            out['dino_factor'] = self.dino_factor
        return out
