import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from collections import OrderedDict

# SimCLR的NT-Xent损失
def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    sim_i_j = torch.diag(sim, N)
    sim_j_i = torch.diag(sim, -N)
    positive = torch.cat([sim_i_j, sim_j_i], dim=0).exp()
    negative = sim.exp().sum(dim=1) - positive
    return -torch.log(positive / (positive + negative)).mean()

# 层次化一致性损失
def region_consistency_loss(z1_global, z1_region, z2_global, z2_region, temperature=0.1):
    z1_global = F.normalize(z1_global, dim=1)
    z2_global = F.normalize(z2_global, dim=1)
    z1_region = F.normalize(z1_region, dim=1)
    z2_region = F.normalize(z2_region, dim=1)
    sim_global = torch.mm(z1_global, z2_global.t()) / temperature
    sim_region = torch.mm(z1_region, z2_region.t()) / temperature
    return -torch.mean(torch.diag(sim_global) + torch.diag(sim_region))

class DKM_SSL(nn.Module):
    def __init__(self, feature_dim=128, weights_path=None, num_attrs=40):
        super(DKM_SSL, self).__init__()
        # SimCLR骨干网络：ResNet-18，从头初始化
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Identity()  # 移除全连接层，仅保留卷积特征
        
        # 手动加载预训练权重
        if weights_path is not None:
            state_dict = torch.load(weights_path)
            # 过滤掉 fc 相关的键
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if not k.startswith('fc'):
                    new_state_dict[k] = v
            self.backbone.load_state_dict(new_state_dict)
            print(f"Loaded pretrained weights from {weights_path} (filtered out fc layer)")
        
        # 图像投影头（SimCLR风格）
        self.image_proj = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 128))
        
        # 关键点编码器
        self.kp_encoder = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 128))
        
        # 引入注意力机制：基于关键点的注意力
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=8)
        
        # 层次化投影头（SimCLR扩展）
        self.global_head = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, feature_dim))
        self.region_head = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, feature_dim))
        
        # 对抗扰动网络
        self.adv_perturbation = nn.Sequential(nn.Linear(10, 10), nn.Tanh())
        
        # 属性分类头（40 个二分类任务）
        self.attr_classifier = nn.Sequential(nn.Linear(feature_dim, 256), nn.ReLU(), nn.Linear(256, num_attrs))
    
    def forward(self, image, keypoints, adversarial=False):
        # SimCLR骨干提取图像特征
        img_feat = self.backbone(image)  # [batch_size, 512]
        img_feat = self.image_proj(img_feat)  # [batch_size, 128]
        
        # 关键点特征
        kp_feat = self.kp_encoder(keypoints)  # [batch_size, 128]
        
        # 引入注意力机制：利用关键点信息增强特征
        attn_output, _ = self.attention(img_feat.unsqueeze(0), kp_feat.unsqueeze(0), kp_feat.unsqueeze(0))
        attn_output = attn_output.squeeze(0)  # [batch_size, 128]
        
        # 融合特征：结合注意力增强后的特征
        fused_feat = attn_output + img_feat
        
        # 层次化表示：全局和区域特征
        global_out = self.global_head(torch.cat([img_feat, fused_feat], dim=1))  # [batch_size, feature_dim]
        region_out = self.region_head(fused_feat)  # [batch_size, feature_dim]
        
        # 属性分类
        attr_logits = self.attr_classifier(global_out)  # [batch_size, num_attrs]
        
        if adversarial:
            pert = self.adv_perturbation(keypoints)
            keypoints_adv = keypoints + 0.05 * pert
            kp_feat_adv = self.kp_encoder(keypoints_adv)
            attn_output_adv, _ = self.attention(img_feat.unsqueeze(0), kp_feat_adv.unsqueeze(0), kp_feat_adv.unsqueeze(0))
            attn_output_adv = attn_output_adv.squeeze(0)
            fused_feat_adv = attn_output_adv + img_feat
            global_out_adv = self.global_head(torch.cat([img_feat, fused_feat_adv], dim=1))
            region_out_adv = self.region_head(fused_feat_adv)
            attr_logits_adv = self.attr_classifier(global_out_adv)
            return (global_out, region_out), (global_out_adv, region_out_adv), attr_logits, attr_logits_adv
        
        return (global_out, region_out), attr_logits