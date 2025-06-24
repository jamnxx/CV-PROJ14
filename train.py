import torch
import torch.nn as nn
import torch.optim as optim
from models.ssl import DKM_SSL, nt_xent_loss, region_consistency_loss
from utils.loader import get_dataloader
from utils.kp_mask import generate_kp_mask
from torchvision import transforms
import yaml
import numpy as np
import pandas as pd
import os

def train():
    with open('/public/home/2022141520174/face/configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_loader = get_dataloader('/public/home/2022141520174/face/da',
                                  '/public/home/2022141520174/face/dataset/list_landmarks_align_celeba.txt',
                                  '/public/home/2022141520174/face/dataset/list_attr_celeba.txt',  # 添加属性文件路径
                                  split='train', batch_size=config['batch_size'])
    val_loader = get_dataloader('/public/home/2022141520174/face/da', 
                                '/public/home/2022141520174/face/dataset/list_landmarks_align_celeba.txt',
                                '/public/home/2022141520174/face/dataset/list_attr_celeba.txt',  # 添加属性文件路径
                                split='val', batch_size=config['batch_size'], shuffle=False)
    
    # 加载属性标签
    attr_file = '/public/home/2022141520174/face/dataset/list_attr_celeba.txt'
    attr_data = pd.read_csv(attr_file, sep='\s+', skiprows=1)
    attr_dict = {row.iloc[0]: row.iloc[1:].values for _, row in attr_data.iterrows()}  # 图像名到属性映射
    
    # 指定手动加载的权重路径
    weights_path = './resnet18-f37072fd.pth'
    model = DKM_SSL(feature_dim=config['feature_dim'], weights_path=weights_path, num_attrs=40).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion_attr = nn.BCEWithLogitsLoss()  # 二分类损失
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for img, keypoints, kp_coords, attr_labels in train_loader:  # 更新为接收属性标签
            img, keypoints, kp_coords, attr_labels = img.cuda(), keypoints.cuda(), kp_coords.cuda(), attr_labels.cuda()
            
            # 获取图像名称并加载属性标签（已通过 DataLoader 提供）
            batch_size = img.shape[0]
            
            # 动态生成 kp_mask_img
            kp_mask_imgs = []
            for i in range(batch_size):
                img_single = img[i].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5
                img_single = (img_single * 255.0).astype(np.uint8)
                keypoints_single = [(keypoints[i][j], keypoints[i][j+1]) for j in range(0, len(keypoints[i]), 2)]
                attention_weights = torch.softmax(torch.rand(len(keypoints_single)), dim=0)
                kp_mask_img = generate_kp_mask(img_single, keypoints_single, 
                                               dynamic_select=True, attention_weights=attention_weights)
                kp_mask_img = torch.tensor(kp_mask_img).permute(2, 0, 1) / 255.0
                kp_mask_img = transforms.Normalize((0.5,), (0.5,))(kp_mask_img)
                kp_mask_imgs.append(kp_mask_img)
            kp_mask_img = torch.stack(kp_mask_imgs).cuda()
            
            # SimCLR对比学习：正样本和对抗样本
            (z1_global, z1_region), (z1_global_adv, z1_region_adv), attr_logits, attr_logits_adv = model(img, keypoints, adversarial=True)
            (z2_global, z2_region), (z2_global_adv, z2_region_adv), _, _ = model(kp_mask_img, keypoints, adversarial=True)
            
            # SimCLR NT-Xent损失
            loss_global = nt_xent_loss(z1_global, z2_global)
            loss_global_adv = nt_xent_loss(z1_global_adv, z2_global_adv)
            
            # 层次化一致性损失
            loss_region = region_consistency_loss(z1_global, z1_region, z2_global, z2_region)
            loss_region_adv = region_consistency_loss(z1_global_adv, z1_region_adv, z2_global_adv, z2_region_adv)
            
            # 属性分类损失
            loss_attr = criterion_attr(attr_logits, attr_labels)
            loss_attr_adv = criterion_attr(attr_logits_adv, attr_labels)
            
            # 总损失
            loss = loss_global + 0.5 * loss_global_adv + 0.3 * (loss_region + loss_region_adv) + 0.1 * (loss_attr + loss_attr_adv)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader):.4f}")
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for img, keypoints, kp_coords, attr_labels in val_loader:  # 更新为接收属性标签
                img, keypoints, kp_coords, attr_labels = img.cuda(), keypoints.cuda(), kp_coords.cuda(), attr_labels.cuda()
                kp_mask_imgs = []
                batch_size = img.shape[0]
                for i in range(batch_size):
                    img_single = img[i].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5
                    img_single = (img_single * 255.0).astype(np.uint8)
                    keypoints_single = [(keypoints[i][j], keypoints[i][j+1]) for j in range(0, len(keypoints[i]), 2)]
                    attention_weights = torch.softmax(torch.rand(len(keypoints_single)), dim=0)
                    kp_mask_img = generate_kp_mask(img_single, keypoints_single, 
                                                   dynamic_select=True, attention_weights=attention_weights)
                    kp_mask_img = torch.tensor(kp_mask_img).permute(2, 0, 1) / 255.0
                    kp_mask_img = transforms.Normalize((0.5,), (0.5,))(kp_mask_img)
                    kp_mask_imgs.append(kp_mask_img)
                kp_mask_img = torch.stack(kp_mask_imgs).cuda()
                
                (z1_global, _), _ = model(img, keypoints, adversarial=False)
                (z2_global, _), _ = model(kp_mask_img, keypoints, adversarial=False)
                val_loss += nt_xent_loss(z1_global, z2_global).item()
        print(f"Val Loss: {val_loss / len(val_loader):.4f}")
        
        torch.save(model.state_dict(), f'/public/home/2022141520174/face/ch2/dkm_ssl_epoch_{epoch+1}.pth')

if __name__ == "__main__":
    train()