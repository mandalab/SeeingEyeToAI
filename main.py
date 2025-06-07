from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import argparse
from scipy import stats
import numpy as np
import os
import random
from utils.model import MultiModalTransformerEncoder
from utils.dataset import MemorabilityDataset

def train_one_epoch(
    model,
    train_dataloader,
    optimizer,
    loss,
    device,
):
    model.train()
    train_loss = 0.0

    true = []
    preds = []

    for batch in tqdm(train_dataloader):
        _, video_features, score = batch

        video_features = video_features.to(device)

        output = model(video_features)

        preds.append(output.cpu().detach().numpy())
        true.append(score.cpu().detach().numpy())

        score = score.to(device).float()
        loss_value = loss(output, score.unsqueeze(1))

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        train_loss += loss_value.item()

    train_loss = train_loss / len(train_dataloader)

    preds = np.concatenate(preds)
    true = np.concatenate(true)
    train_corr, _ = stats.spearmanr(preds, true)

    return train_loss, train_corr, true, preds


def validate(
    model,
    val_dataloader,
    loss,
    device,
):
    with torch.no_grad():
        model.eval()
        val_loss = 0.0

        preds = []
        true = []
        video_ids = []

        for batch in tqdm(val_dataloader):
            video_id, video_features, score = batch
            video_features = video_features.to(device)

            output = model(video_features)

            preds.append(output.cpu().detach().numpy())
            true.append(score.cpu().detach().numpy())
            video_ids.append(video_id)

            score = score.to(device).float()
            loss_value = loss(output, score.unsqueeze(1))

            val_loss += loss_value.item()

        val_loss = val_loss / len(val_dataloader)

        preds = np.concatenate(preds)
        true = np.concatenate(true)
        video_ids = np.concatenate(video_ids)

        val_corr, _ = stats.spearmanr(preds, true)

    return val_loss, val_corr, true, preds, video_ids


def trigger_training(
    batch_size,
    path,
    epochs,
    lr,
    num_frames,
    frame_randomisation,
    num_encoder_layers,
    dropout,
    num_transformer_heads,
    patch_x,
    patch_y,
    train_data_path,
    val_data_path,
    hidden_dim,
    input_feature_dim,
    model_save_dir,
    results_dir,
):
    BEST_VAL_CORR = -1
    LOSS = nn.MSELoss()

    print("Loss: ", LOSS)
    print("Batch size: ", batch_size)
    print("Learning rate: ", lr)
    print("Frame randomisation: ", frame_randomisation)
    print("Number of encoder layers: ", num_encoder_layers)
    print("Number of transformer heads: ", num_transformer_heads)
    print("Dropout: ", dropout)
    print("Number of frames: ", num_frames)

    with open(train_data_path, "rb") as f:
        train = pd.read_csv(f)

    with open(val_data_path, "rb") as f:
        val = pd.read_csv(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = MemorabilityDataset(
        train,
        path,
        split="train",
        num_frames=num_frames,
        frame_randomisation=frame_randomisation,
    )
    val_dataset = MemorabilityDataset(
        val,    
        path,
        split="val",
        num_frames=num_frames,
        frame_randomisation=False,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    model = MultiModalTransformerEncoder(
        video_feature_dim=input_feature_dim,
        hidden_dim=hidden_dim,
        num_frames=num_frames,
        num_encoder_layers=num_encoder_layers,
        n_heads=num_transformer_heads,
        dropout_rate=dropout,
        patch_x=patch_x,
        patch_y=patch_y,
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initial validation
    initial_val_loss, initial_val_corr, initial_true_val, initial_preds_val, video_ids = validate(
        model, val_dataloader, LOSS, device
    )
    print(f"Initial Val Loss: {initial_val_loss}, Correlation: {initial_val_corr}")

    # Training loop
    for epoch in range(epochs):
        train_loss, train_corr, true_train, preds_train = train_one_epoch(
            model, train_dataloader, optimizer, LOSS, device
        )
        print(f"Epoch: {epoch}, Train Loss: {train_loss}, Correlation: {train_corr}")

        val_loss, val_corr, true_val, preds_val, video_ids = validate(
            model, val_dataloader, LOSS, device
        )

        print(f"Epoch: {epoch}, Val Loss: {val_loss}, Correlation: {val_corr}")

        if val_corr > BEST_VAL_CORR:
            BEST_VAL_CORR = val_corr

            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)

            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            df_best_corr = pd.DataFrame(
                {"video_id": video_ids, "true": true_val, "preds": preds_val.squeeze(1)}
            )
            df_best_corr.to_csv(
                f"{results_dir}/{lr}_layers_{num_encoder_layers}_frames_{num_frames}_rand_{frame_randomisation}.csv",
                index=False,
            )
            print("Saving best correlation model at epoch: ", epoch)

            # save model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                f"{model_save_dir}/{lr}_layers_{num_encoder_layers}_frames_{num_frames}_rand_{frame_randomisation}.pt",
            )


def seed_everything(seed=0, harsh=False):
    """
    Seeds all important random functions
    Args:
        seed (int, optional): seed value. Defaults to 0.
        harsh (bool, optional): torch backend deterministic. Defaults to False.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    if harsh:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    os.environ["PYTHONHASHSEED"] = str(seed)


#################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Memorability")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--path", type=str, required=True, help="Path to the embeddings directory")
    parser.add_argument("--patch_x", type=int, default=7, help="Number of patches in x dimension")
    parser.add_argument("--patch_y", type=int, default=7, help="Number of patches in y dimension")
    parser.add_argument("--frame_randomisation", type=bool, default=True, help="Frame randomisation")
    parser.add_argument("--num_encoder_layers", type=int, default=2, help="Number of encoder layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--num_transformer_heads", type=int, default=8, help="Number of transformer heads")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--num_frames", type=int, default=5, help="Number of frames")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to training data pickle file")
    parser.add_argument("--val_data_path", type=str, required=True, help="Path to validation data pickle file")
    parser.add_argument("--hidden_dim", type=int, default=768, help="Hidden dimension size")
    parser.add_argument("--input_feature_dim", type=int, default=2048, help="Input feature dimension")
    parser.add_argument("--model_save_dir", type=str, default="./models", help="Directory to save models")
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory to save results")
    args = parser.parse_args()

    seed_everything(args.seed)

    trigger_training(
        args.batch_size,
        args.path,
        args.epochs,
        args.lr,
        args.num_frames,
        args.frame_randomisation,
        args.num_encoder_layers,
        args.dropout,
        args.num_transformer_heads,
        args.patch_x,
        args.patch_y,
        args.train_data_path,
        args.val_data_path,
        args.hidden_dim,
        args.input_feature_dim,
        args.model_save_dir,
        args.results_dir,
    )