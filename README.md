<div align="center">
    <h1>👁️ Seeing Eye to AI: Comparing Human Gaze and Model Attention in Video Memorability</h1>
    <a><img src="https://img.shields.io/badge/python-3.8-blue"></a>
    <a"><img src="https://img.shields.io/badge/made_with-pytorch-red"></a>
    <a><img src="https://img.shields.io/badge/dataset-Memento-orange"></a>
    <a href="https://arxiv.org/abs/2311.16484"><img src="https://img.shields.io/badge/arXiv-2311.16484-f9f107.svg"></a>
    <a href="https://katha-ai.github.io/projects/video-memorability/"><img src="https://img.shields.io/website?up_message=up&up_color=green&down_message=down&down_color=red&url=https%3A%2F%2Fkatha-ai.github.io%2Fprojects%2Fvideo-memorability%2F&link=https%3A%2F%2Fkatha-ai.github.io%2Fprojects%2Fvideo-memorability%2F"></a>
    <a href="https://www.youtube.com/watch?v=oS_10WeHiHQ"><img src="https://badges.aleen42.com/src/youtube.svg"></a>
</div>

## 📑 Contents
1. [About](#about)
2. [Setup](#setup)
3. [Dataset](#dataset)
4. [Repository Structure](#repository-structure)
5. [Usage](#usage)
6. [Citation](#citation)

## 🤖 About
Understanding what makes a video memorable has important applications in advertising or education technology. Towards this goal, we investigate spatio-temporal attention mechanisms underlying video memorability. Different from previous works that fuse multiple features, we adopt a simple CNN+Transformer architecture that enables analysis of spatio-temporal attention while matching state-of-the-art (SoTA) performance on video memorability prediction. We compare model attention against human gaze fixations collected through a small-scale eye-tracking study where humans perform the video memory task. We uncover the following insights: (i) Quantitative saliency metrics show that our model, trained only to predict a memorability score, exhibits similar spatial attention patterns to human gaze, especially for more memorable videos. (ii) The model assigns greater importance to initial frames in a video, mimicking human attention patterns. (iii) Panoptic segmentation reveals that both (model and humans) assign a greater share of attention to things and less attention to stuff as compared to their occurrence probability.

For more details, please visit our [project website](https://kumar-prajneya.github.io/SeeingEyeToAI) or read our [paper](https://arxiv.org/abs/2311.16484).

## 🛠️ Setup

1. **Clone the repository**
```bash
git clone [repository-url]
cd [repository-name]
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📊 Dataset

The Memento dataset can be downloaded from [http://memento.csail.mit.edu/#Dataset](http://memento.csail.mit.edu/#Dataset). 

## 📁 Repository Structure

```
.
├── main.py                # Main training script
├── embed.py               # Video embedding generation
├── attention.py           # Attention matrix extraction
├── panoptic.py            # Panoptic segmentation
├── requirements.txt       # Python dependencies
├── eyetracking/           # Eye-tracking data and related processing
├── utils/
│   ├── model.py           # Transformer model implementation
│   └── dataset.py         # Dataset handling
```

## 🚀 Usage

### 1. Generate Embeddings
```bash
python embed.py --path /path/to/videos
```

### 2. Train Model
```bash
python main.py \
    --path /path/to/embeddings \
    --train_data_path /path/to/train.csv \
    --val_data_path /path/to/val.csv
```

### 3. Attention Analysis
Extract attention matrices to analyze model's focus:
```bash
python attention.py \
    --model_path /path/to/trained/model.pt \
    --val_path /path/to/val.csv \
    --features_path /path/to/features
```

### 4. Panoptic Segmentation
Generate panoptic segmentation results:
```bash
python panoptic.py \
    --video_path /path/to/videos
```

## 📝 Citation
If you use this code in your research, please cite our paper:
```
@article{kumar2025eyetoai,
    title = {{Seeing Eye to AI: Comparing Human Gaze and Model Attention in Video Memorability}},
    author = {Kumar, Prajneya and Khandelwal, Eshika and Tapaswi, Makarand and Sreekumar, Vishnu},
    year = {2025},
    booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)}
}
```
