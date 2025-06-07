from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from PIL import Image
import requests
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import transforms
import torch
from torchvision.transforms import ToTensor, Resize, Compose
import numpy as np
import tqdm as tqdm
import pickle
import argparse


def preprocess_frame(frame):
    preprocess = Compose([
        Resize((224, 224),  interpolation=Image.BICUBIC),  # Resize the image to 224x224
    ])

    frame = Image.fromarray(frame)
    tensor = preprocess(frame)

    return tensor


def get_video_embeddings(folder_path, video_id, num_frames):

    video_path = os.path.join(folder_path, video_id)
    print(f"Processing video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video file: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_range = total_frames // num_frames
    
    frame_indices = [
        i * frame_range + frame_range // 2 for i in range(num_frames)
    ]

    vid_frames = []

    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num  in frame_indices:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Preprocess the frame and get embedding
            tensor = preprocess_frame(frame)

            vid_frames.append(tensor)

    cap.release()

    return vid_frames


def main(args):
    id2_label = {
        "0": "person",
        "1": "bicycle",
        "2": "car",
        "3": "motorcycle",
        "4": "airplane",
        "5": "bus",
        "6": "train",
        "7": "truck",
        "8": "boat",
        "9": "traffic light",
        "10": "fire hydrant",
        "11": "stop sign",
        "12": "parking meter",
        "13": "bench",
        "14": "bird",
        "15": "cat",
        "16": "dog",
        "17": "horse",
        "18": "sheep",
        "19": "cow",
        "20": "elephant",
        "21": "bear",
        "22": "zebra",
        "23": "giraffe",
        "24": "backpack",
        "25": "umbrella",
        "26": "handbag",
        "27": "tie",
        "28": "suitcase",
        "29": "frisbee",
        "30": "skis",
        "31": "snowboard",
        "32": "sports ball",
        "33": "kite",
        "34": "baseball bat",
        "35": "baseball glove",
        "36": "skateboard",
        "37": "surfboard",
        "38": "tennis racket",
        "39": "bottle",
        "40": "wine glass",
        "41": "cup",
        "42": "fork",
        "43": "knife",
        "44": "spoon",
        "45": "bowl",
        "46": "banana",
        "47": "apple",
        "48": "sandwich",
        "49": "orange",
        "50": "broccoli",
        "51": "carrot",
        "52": "hot dog",
        "53": "pizza",
        "54": "donut",
        "55": "cake",
        "56": "chair",
        "57": "couch",
        "58": "potted plant",
        "59": "bed",
        "60": "dining table",
        "61": "toilet",
        "62": "tv",
        "63": "laptop",
        "64": "mouse",
        "65": "remote",
        "66": "keyboard",
        "67": "cell phone",
        "68": "microwave",
        "69": "oven",
        "70": "toaster",
        "71": "sink",
        "72": "refrigerator",
        "73": "book",
        "74": "clock",
        "75": "vase",
        "76": "scissors",
        "77": "teddy bear",
        "78": "hair drier",
        "79": "toothbrush",
        "80": "banner",
        "81": "blanket",
        "82": "bridge",
        "83": "cardboard",
        "84": "counter",
        "85": "curtain",
        "86": "door-stuff",
        "87": "floor-wood",
        "88": "flower",
        "89": "fruit",
        "90": "gravel",
        "91": "house",
        "92": "light",
        "93": "mirror-stuff",
        "94": "net",
        "95": "pillow",
        "96": "platform",
        "97": "playingfield",
        "98": "railroad",
        "99": "river",
        "100": "road",
        "101": "roof",
        "102": "sand",
        "103": "sea",
        "104": "shelf",
        "105": "snow",
        "106": "stairs",
        "107": "tent",
        "108": "towel",
        "109": "wall-brick",
        "110": "wall-stone",
        "111": "wall-tile",
        "112": "wall-wood",
        "113": "water-other",
        "114": "window-blind",
        "115": "window-other",
        "116": "tree-merged",
        "117": "fence-merged",
        "118": "ceiling-merged",
        "119": "sky-other-merged",
        "120": "cabinet-merged",
        "121": "table-merged",
        "122": "floor-other-merged",
        "123": "pavement-merged",
        "124": "mountain-merged",
        "125": "grass-merged",
        "126": "dirt-merged",
        "127": "paper-merged",
        "128": "food-other-merged",
        "129": "building-other-merged",
        "130": "rock-merged",
        "131": "wall-other-merged",
        "132": "rug-merged"
    }

    num_frames = args.num_frames
    videos = args.video_path

    # load MaskFormer fine-tuned on COCO panoptic segmentation
    feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-small-coco")
    model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-small-coco")

    for filename in os.listdir(videos):
        print(filename)
        if os.path.exists(f"{args.save_dir}/{filename}.npy"):
            print(f"Skipping {filename}")
            continue

        video_frames = get_video_embeddings(videos, filename, num_frames)

        panoptic_frames = []
        for i in range(num_frames):  
            try:  
                inputs = feature_extractor(images=[video_frames[i]], return_tensors="pt")

                outputs = model(**inputs)

                result = feature_extractor.post_process_panoptic_segmentation(outputs, target_sizes=[video_frames[i].size[::-1]])[0]
                predicted_panoptic_map = result["segmentation"]
                predicted_masks = result["segments_info"]

                pix2label = {el["id"]: el["label_id"] for el in predicted_masks}
                pix2label[0] = -1

                predicted_panoptic = predicted_panoptic_map.apply_(lambda x: pix2label[x])
                
                # for number of objects
                # predicted_panoptic = len(result["segments_info"])
                
                panoptic_frames.append(predicted_panoptic)
            except:
                print(f"Error in {filename}")
                continue

        if len(panoptic_frames) == num_frames:
            panoptic_frames = torch.stack(panoptic_frames)
            panoptic_frames = panoptic_frames.numpy()

            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            np.save(f"{args.save_dir}/{filename[:-4]}.npy", panoptic_frames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Panoptic Segmentation")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video folder")
    parser.add_argument("--num_frames", type=int, default=5, help="Number of frames to process")
    parser.add_argument("--save_dir", type=str, default="./panoptic_memento", help="Directory to save panoptic segmentation results")
    args = parser.parse_args()
    main(args)