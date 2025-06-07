from torch.utils.data import Dataset
import numpy as np
import torch
import random

class MemorabilityDataset(Dataset):
    def __init__(
        self,
        df,
        path,
        split="train",
        num_frames=5,
        frame_randomisation=True,
    ):
        self.split = split
        self.frame_randomisation = frame_randomisation
        self.num_frames = num_frames
        self.path = path
        self.labels = df["scores_short_term"].values
        self.video_id = df["video_id"].values
        self.video_path = df["video_id"].apply(
            lambda x: f"{self.path}/" + str(x) + ".npy"
        ).values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        video_id = self.video_id[idx]
        label = torch.tensor(label, dtype=torch.float32)

        video_emb = np.load(self.video_path[idx])
        total_frames = video_emb.shape[0]
        frame_range = total_frames // self.num_frames

        if self.frame_randomisation:
            frame_indices = [
                random.randint(i * frame_range, (i + 1) * frame_range - 1)
                for i in range(self.num_frames)
            ]
        else:
            frame_indices = [
                i * frame_range + frame_range // 2 for i in range(self.num_frames)
            ]

        frame_embeddings = video_emb[frame_indices]
        frame_embeddings = torch.tensor(frame_embeddings, dtype=torch.float32)
        frame_embeddings = frame_embeddings.permute(1, 0, 3, 4, 2)
        frame_embeddings = frame_embeddings.squeeze(0)  # ( 5, 7, 7, 2048)

        return video_id, frame_embeddings, label

