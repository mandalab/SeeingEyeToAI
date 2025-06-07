from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import pandas as pd
import tqdm as tqdm
import argparse
import numpy as np
import os
from model import MultiModalTransformerEncoder
from dataset import MemorabilityDataset
import pickle

class AttExtractor(nn.Module):
    def __init__(self, model):
        super(AttExtractor, self).__init__()
        self.model = model

    def forward(self, video_features):
        #### DATA PREP BEGINS
        device = video_features.device
        batch_size = video_features.shape[0]
        num_frames = video_features.shape[1]

        self.init_cls_token = torch.zeros(batch_size, 1).long()
        cls_tokens = self.model.cls_token(self.init_cls_token.to(device))
        self.init_cls_att_mask = torch.ones(batch_size, 1).bool()

        self.init_video_att_mask = torch.ones(
            batch_size, num_frames * self.model.patch_x * self.model.patch_y
        ).bool()
        video_embeddings = self.model.video_linear(video_features.float())

        video_pos_encodings_temporal = self.model.pos_temporal_embed(
            self.model.init_pos_temporal_embed.repeat(batch_size, 1).to(device)
        )
    
        video_pos_encodings_temporal = video_pos_encodings_temporal.unsqueeze(
            2
        ).unsqueeze(3)
        video_embeddings += video_pos_encodings_temporal

        # flatten video
        video_embeddings = video_embeddings.view(
            batch_size, -1, self.model.hidden_dim
        )  # (B x (5x7x7) x hidden_dim)
        video_embeddings = self.model.layer_norm(video_embeddings)

        combined_embeddings = torch.cat([cls_tokens, video_embeddings], dim=1)
        combined_attention_mask = torch.cat(
            [
                self.init_cls_att_mask.to(device),
                self.init_video_att_mask.to(device),
            ],
            dim=1,
        )
        
        combined_attention_mask = ~combined_attention_mask

        #### DATA PREP ENDS

        # ALL THE CODE TILL ABOVE IS EXACTLY THE SAME AS MultiModalTransformerEncoder's forward function, next part has changes
        # MANUALLY CALLING THE ENCODER LAYER INSTEAD OF SIMPLY calling sefl.model.encoder(   combined_embeddings, src_key_padding_mask=combined_attention_mask)

        att_out1 = self.model.transformer_encoder.layers[0].self_attn(
            combined_embeddings, combined_embeddings, combined_embeddings, key_padding_mask=combined_attention_mask,
        )
    
        out1 = self.model.transformer_encoder.layers[0](combined_embeddings, src_key_padding_mask=combined_attention_mask)

        att_out2 = self.model.transformer_encoder.layers[1].self_attn(
            out1, out1, out1, key_padding_mask=combined_attention_mask, 
        )

        out2 = self.model.transformer_encoder.layers[1](out1, src_key_padding_mask=combined_attention_mask)

        return att_out1[1], att_out2[1]    

def main(args):

    PATCH_X = 7
    PATCH_Y = 7
    INPUT_FEATURE_DIM = 2048
    HIDDEN_DIM = 768
    NUM_FRAMES = 5
    NUM_ENCODER_LAYERS = 2
    NUM_HEADS = 8
    DROPOUT = 0.1
    BATCH_SIZE = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_df = pd.read_csv(args.val_path)
    val_dataset = MemorabilityDataset(
        val_df,    
        args.features_path,
        split="val",
        num_frames=NUM_FRAMES,
        frame_randomisation=False,
    )

    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = MultiModalTransformerEncoder(
            video_feature_dim=INPUT_FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_frames=NUM_FRAMES,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            n_heads=NUM_HEADS,
            dropout_rate=DROPOUT,
            patch_x=PATCH_X,
            patch_y=PATCH_Y,
        )

    model.load_state_dict(torch.load(args.model_path)['model_state_dict'])
    model = model.to(device)
    att_tx = AttExtractor(model)
    att_tx = att_tx.to(device)
    att_tx.eval()

    att_dict1 = {}
    att_dict2 = {}
    for batch_idx, (video_id, video_features, label) in enumerate(val_dataloader):
        video_features = video_features.to(device)
        att_out1, att_out2 = att_tx(video_features)

        for i in range(len(video_id)):
            att_dict1[(video_id[i])] = att_out1[i].detach().cpu().numpy()
            att_dict2[(video_id[i])] = att_out2[i].detach().cpu().numpy()

    os.makedirs(os.path.dirname(args.att_dict1_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.att_dict2_path), exist_ok=True)

    with open(args.att_dict1_path, 'wb') as f:
        pickle.dump(att_dict1, f)

    with open(args.att_dict2_path, 'wb') as f:
        pickle.dump(att_dict2, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract attention weights from the model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--val_path", type=str, required=True, help="Path to the validation data")
    parser.add_argument("--features_path", type=str, required=True, help="Path to the video features")
    parser.add_argument("--att_dict1_path", type=str, default="./att/att_dict1.pkl", help="Path to save first attention dictionary")
    parser.add_argument("--att_dict2_path", type=str, default="./att/att_dict2.pkl", help="Path to save second attention dictionary")
    args = parser.parse_args()
    main(args)

        