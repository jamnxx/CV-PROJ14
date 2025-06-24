import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
from torchvision import transforms
import numpy as np

class CelebADataset(Dataset):
    def __init__(self, data_dir, keypoints_file, attr_file, split='train', transform=None, target_size=(224, 224)):
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform
        self.target_size = target_size
        self.image_ids = [f for f in os.listdir(self.data_dir) if f.endswith('.jpg')]
        
        # 加载关键点数据
        with open(keypoints_file, 'r') as f:
            lines = f.readlines()[2:]
        
        data = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 11:
                record = {
                    'file_name': parts[0],
                    'lefteye_x': float(parts[1]),
                    'lefteye_y': float(parts[2]),
                    'righteye_x': float(parts[3]),
                    'righteye_y': float(parts[4]),
                    'nose_x': float(parts[5]),
                    'nose_y': float(parts[6]),
                    'mouth_left_x': float(parts[7]),
                    'mouth_left_y': float(parts[8]),
                    'mouth_right_x': float(parts[9]),
                    'mouth_right_y': float(parts[10])
                }
                data.append(record)
        
        self.keypoints = pd.DataFrame(data)
        self.keypoints = self.keypoints[self.keypoints['file_name'].isin(self.image_ids)]
        
        if len(self.keypoints) == 0:
            raise ValueError("No matching keypoints found for any image!")
        if len(self.keypoints) != len(self.image_ids):
            print(f"Warning: Only {len(self.keypoints)}/{len(self.image_ids)} images have keypoints")
        
        # 加载属性标签
        attr_columns = ['file_name'] + [
            '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',
            'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
            'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
            'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
            'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
            'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
            'Wearing_Necktie', 'Young'
        ]
        # 强制将属性列转换为数值类型
        attr_data = pd.read_csv(attr_file, sep='\s+', skiprows=2, names=attr_columns)
        for col in attr_columns[1:]:  # 跳过 file_name 列
            attr_data[col] = pd.to_numeric(attr_data[col], errors='coerce')
        
        # 调试：打印前几个文件名
        print(f"First 5 image_ids: {self.image_ids[:5]}")
        print(f"First 5 attr_data filenames: {attr_data['file_name'].head().tolist()}")
        
        # 优化匹配逻辑：忽略大小写
        attr_data['file_name'] = attr_data['file_name'].astype(str).str.lower()
        self.image_ids_lower = [f.lower() for f in self.image_ids]
        self.attrs = attr_data[attr_data['file_name'].isin(self.image_ids_lower)]
        
        if len(self.attrs) != len(self.image_ids):
            print(f"Warning: Only {len(self.attrs)}/{len(self.image_ids)} images have attributes")
            if len(self.attrs) == 0:
                raise ValueError("No matching attributes found for any image! Check filenames.")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        orig_width, orig_height = image.size
        
        # 获取关键点坐标
        row = self.keypoints[self.keypoints['file_name'] == img_name]
        if row.empty:
            available_files = self.keypoints['file_name'].unique()[:5]
            raise ValueError(
                f"No keypoints found for {img_name}\n"
                f"First 5 available files: {available_files}"
            )
            
        keypoints = [
            (row['lefteye_x'].values[0], row['lefteye_y'].values[0]),
            (row['righteye_x'].values[0], row['righteye_y'].values[0]),
            (row['nose_x'].values[0], row['nose_y'].values[0]),
            (row['mouth_left_x'].values[0], row['mouth_left_y'].values[0]),
            (row['mouth_right_x'].values[0], row['mouth_right_y'].values[0])
        ]
        
        kp_array = np.array(keypoints, dtype=np.float32)
        width_scale = self.target_size[0] / orig_width
        height_scale = self.target_size[1] / orig_height
        kp_array[:, 0] *= width_scale
        kp_array[:, 1] *= height_scale
        kp_coords = torch.from_numpy(kp_array.flatten()).float()
        
        # 获取属性标签
        attr_row = self.attrs[self.attrs['file_name'] == img_name.lower()]
        if attr_row.empty:
            attr_label = np.zeros(40, dtype=np.float32)
        else:
            attr_label = attr_row.iloc[0, 1:].values.astype(np.float32)
            attr_label = (attr_label + 1) / 2  # 转换为 0/1
        
        if self.transform:
            image = self.transform(image)
        
        return image, kp_coords, kp_coords, torch.tensor(attr_label, dtype=torch.float32)

def get_dataloader(data_dir, keypoints_file, attr_file, split='train', batch_size=64, shuffle=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Resize((224, 224)),
    ])
    dataset = CelebADataset(data_dir, keypoints_file, attr_file, split, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)