/import torch
import torch.nn as nn
from models.ssl import DKM_SSL
from utils.loader import get_dataloader
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

def evaluate(model_path, data_dir, keypoints_file, attr_file):
    # 加载模型
    model = DKM_SSL(feature_dim=128, weights_path=None).cuda()  # 移除 num_attrs，保持与现有模型一致
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 加载测试数据
    test_loader = get_dataloader(data_dir, keypoints_file, attr_file, split='test', batch_size=32, shuffle=False)

    # 提取特征和属性标签
    features = []
    attr_labels_all = []
    with torch.no_grad():
        for img, keypoints, kp_coords, attr_labels in test_loader:
            img, keypoints, kp_coords, attr_labels = img.cuda(), keypoints.cuda(), kp_coords.cuda(), attr_labels.cuda()
            (z_global, _), attr_logits = model(img, keypoints, adversarial=False)
            features.append(z_global.cpu().numpy())
            attr_labels_all.append(attr_labels.cpu().numpy())
    
    features = np.concatenate(features)
    attr_labels_all = np.concatenate(attr_labels_all)

    # 属性识别性能评估
    device = torch.device("cuda")
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    attr_logits_tensor = model.attr_classifier(features_tensor)
    attr_preds = (torch.sigmoid(attr_logits_tensor).detach().cpu().numpy() > 0.5).astype(int)
    f1_scores = [f1_score(attr_labels_all[:, i], attr_preds[:, i]) for i in range(40)]
    mean_f1 = np.mean(f1_scores)
    print(f"Attribute F1 Scores: Mean {mean_f1:.4f}, Per Attribute: {f1_scores}")

    # 特征质量评估：Silhouette Score
    kmeans = KMeans(n_clusters=10, random_state=42).fit(features)
    silhouette_avg = silhouette_score(features, kmeans.labels_)
    print(f"Silhouette Score: {silhouette_avg:.4f}")

    # t-SNE 可视化
    tsne = TSNE(n_components=2, random_state=42).fit_transform(features)
    plt.scatter(tsne[:, 0], tsne[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.colorbar()
    plt.title('t-SNE Visualization of Features')
    plt.savefig('/public/home/2022141520174/face/tsne.png')
    print("t-SNE visualization saved to /public/home/2022141520174/face/tsne.png")

if __name__ == "__main__":
    evaluate(
        model_path='/public/home/2022141520174/face/ch2/dkm_ssl_epoch_50.pth',
        data_dir='/public/home/2022141520174/face/da',
        keypoints_file='/public/home/2022141520174/face/dataset/list_landmarks_align_celeba.txt',
        attr_file='/public/home/2022141520174/face/dataset/list_attr_celeba.txt'
    )