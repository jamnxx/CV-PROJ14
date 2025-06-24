import cv2
import numpy as np
import torch

def generate_kp_mask(image, keypoints, dynamic_select=False, attention_weights=None):
    """
    生成关键点遮挡图像。
    
    参数：
    - image: 输入图像（numpy数组，HWC格式）
    - keypoints: 关键点列表，格式为 [(x1, y1), (x2, y2), ...]
    - dynamic_select: 是否动态选择关键点
    - attention_weights: 注意力权重，用于动态选择关键点
    
    返回：
    - masked_image: 遮挡后的图像（numpy数组，HWC格式）
    """
    # 确保 image 是 uint8 类型
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    # 初始化单通道掩码
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)  # (H, W) 单通道
    selected_keypoints = keypoints
    
    # 如果启用动态选择，基于注意力权重选择关键点
    if dynamic_select and attention_weights is not None:
        weights = attention_weights[:len(keypoints)]
        indices = torch.topk(weights, k=min(3, len(keypoints))).indices  # 选择前3个关键点
        selected_keypoints = [keypoints[i] for i in indices]
    
    # 在图像上绘制遮挡区域
    for kp in selected_keypoints:
        x, y = int(kp[0]), int(kp[1])
        size = np.random.randint(15, 25)  # 随机遮挡区域大小
        cv2.rectangle(mask, (x-size//2, y-size//2), (x+size//2, y+size//2), 255, -1)
    
    # 应用遮挡到原始图像
    masked_image = image.copy()
    masked_image[mask == 255] = np.mean(image[mask == 255], axis=(0, 1))  # 按通道平均值填充
    return masked_image