# CV-PROJ14
This is for CS1674 COMPUTER VISION.
# DKM SSL: Multi-Scale Keypoint-Guided Contrastive Face Representation Learning

A novel self-supervised learning framework for facial representation learning on the CelebA dataset, integrating multi-scale keypoint-guided contrastive learning, adversarial perturbations, and attribute classification to address occlusion and pose challenges.

## Table of Contents

- About the Project
- Features
- Getting Started
  - Prerequisites
  - Installation
- Usage
- Project Structure
- Contributing
- License
- Acknowledgments

## About the Project

DKM SSL (Dynamic Keypoint-Guided Multi-Scale Self-Supervised Learning) is a Python-based framework designed for robust facial representation learning. It leverages the CelebA dataset to learn high-quality representations resilient to real-world challenges like occlusions (e.g., hands, masks) and pose variations. The model integrates:

- **Multi-scale keypoint-guided contrastive learning** to capture global identity and local facial details.
- **Adversarial perturbations** (coefficient: 0.05) to enhance robustness against annotation noise.
- **Attribute classification** for downstream tasks, achieving a mean F1 score of 0.3824 across 40 CelebA attributes.

Trained on the full CelebA dataset (162,770 training, 19,867 validation/test images), DKM SSL achieves a Silhouette Score of 0.6123 and Davies-Bouldin Index of 0.5241, demonstrating strong clustering quality.

## Features

- Multi-scale feature fusion for robust facial representations.
- Keypoint-based modeling to handle pose and occlusion.
- Adversarial perturbations for improved generalization.
- Attribute classification for downstream applicability.
- Reproducible setup with fixed random seeds (`torch.manual_seed(42)`).

## Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

- Python 3.10.14
- PyTorch 3.10.14 with CUDA 12.4
- NVIDIA GPU (e.g., A800, 80GB recommended)
- CelebA dataset (download from CelebA official site)
- Required Python packages (listed in `requirements.txt`)

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/dkm-ssl.git
   cd dkm-ssl
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download CelebA dataset**:

   - Place images in `data/celeba/img_align_celeba/`.
   - Place landmark and attribute files (`list_landmarks_align_celeba.txt`, `list_attr_celeba.txt`) in `data/celeba/`.

4. **Set up environment**:

   - Ensure CUDA 12.4 is installed for GPU support.
   - Verify PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"` (should output `True`).

## Usage

To train the DKM SSL model:

```bash
python train.py --data_dir data/celeba --batch_size 64 --epochs 50 --seed 42
```

Key arguments:

- `--data_dir`: Path to CelebA dataset.
- `--batch_size`: Batch size (default: 64).
- `--epochs`: Number of training epochs (default: 50).
- `--seed`: Random seed for reproducibility (default: 42).

To evaluate the model:

```bash
python evaluate.py --checkpoint_path checkpoints/model_best.pth --data_dir data/celeba
```

Expected training time: \~20 hours for 50 epochs on an NVIDIA A800 (80GB). Peak GPU memory usage: \~60GB.

### Preprocessing Details

- Images are resized to 224x224 and normalized (`transforms.Normalize((0.5,), (0.5,))`).
- Occlusion masks are generated around keypoints with random sizes (15–25 pixels) using `kp_mask.py`.
- Landmarks and attributes are loaded via `loader.py`.

### Example Output

After training, evaluate clustering and attribute classification:

```bash
Silhouette Score: 0.6123
Davies-Bouldin Index: 0.5241
Mean F1 Score (40 attributes): 0.3824
```

## Project Structure

```
dkm-ssl/
├── data/                # CelebA dataset (not included)
├── checkpoints/         # Model checkpoints
├── loader.py            # Data loading and preprocessing
├── kp_mask.py           # Occlusion mask generation
├── train.py             # Training script
├── eval.py              # Evaluation script
└── README.md            # Project documentation
```

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Acknowledgments

- CelebA dataset for providing the data.
- PyTorch for the deep learning framework.
- Inspired by SimCLR and Transformer literature (Vaswani et al., 2017).
