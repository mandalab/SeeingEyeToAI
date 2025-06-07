import os
import clip
import torch
import numpy as np
import cv2
from PIL import Image
import pickle
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
import argparse


def preprocess_frame(frame):
    """Preprocess a single frame for CLIP."""
    frame = Image.fromarray(frame)
    return custom_transforms(frame).unsqueeze(0)


def get_video_embeddings(video_path, output_dir):
    """Extract frames from video and calculate embeddings."""
    video_id = os.path.basename(video_path).split(".")[0]

    output_file_path = os.path.join(output_dir, f"{video_id}.npy")
    if os.path.exists(output_file_path):
        print(f"Embeddings for video {video_id} already exist. Skipping...")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video file: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_embedding = []

    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = preprocess_frame(frame).to(device).half()
        with torch.no_grad():
            image_features = model.encode_image(tensor)

        video_embedding.append(image_features.cpu().numpy())

    cap.release()
    video_embedding = np.array(video_embedding)
    np.save(os.path.join(output_dir, f"{video_id}.npy"), video_embedding)


def get_folder_embeddings(folder_path, output_dir):
    video_files = os.listdir(folder_path)
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(folder_path, video_file)
        get_video_embeddings(video_path, output_dir)


class CustomCLIPModel(nn.Module):
    def __init__(self, original_model):
        super(CustomCLIPModel, self).__init__()
        self.visual = nn.Sequential(*list(original_model.visual.children())[:-1])

    def encode_image(self, image):
        return self.visual(image)

def main(args):

    PATH = args.path
    H = args.h
    W =args.w
    CROP = args.crop

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = clip.load("RN50", device=device)
    model = CustomCLIPModel(model).to(device)

    if CROP:
        custom_transforms = transform  # default transforms (centre crop)
    else:
        custom_transforms = transforms.Compose(
            [
                transforms.Resize((H, W), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    print(custom_transforms)

    output_dir = f"{PATH}_RN50_{H}x{W}"
    if CROP:
        output_dir += "_crop"
    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    get_folder_embeddings(PATH, output_dir)
    print(f"Embeddings completed for {PATH}")

    if device == "cuda":
        torch.cuda.empty_cache()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to input video directory")
    parser.add_argument("--h", type=int, default=224)
    parser.add_argument("--w", type=int, default=224)
    parser.add_argument("--crop", type=bool, default=False)

    args = parser.parse_args()
    main(args)
