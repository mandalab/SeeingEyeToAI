import argparse
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import pickle


def read_data(filepath: Path) -> pd.DataFrame:
    """
    Read a fixation file, trying Excel first then CSV.
    """
    try:
        return pd.read_excel(filepath, engine='xlrd')
    except Exception:
        return pd.read_csv(filepath, sep='\t')


def calculate_sigma(distance_to_screen: float,
                   screen_height_in: float,
                   screen_resolution_y: int,
                   visual_angle: float) -> float:
    """
    Compute Gaussian sigma (px) from visual angle parameters.
    """
    import math
    visual_angle_rad = math.radians(visual_angle)
    ppd = (2 * distance_to_screen * math.tan(visual_angle_rad / 2)) / screen_height_in * screen_resolution_y
    return ppd / 2.355  # FWHM to sigma


def get_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def get_video_dimensions(video_path: Path) -> tuple[int,int]:
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def adjust_gaze_coordinates(df: pd.DataFrame,
                            dataset: str,
                            video_width: int,
                            video_height: int,
                            screen_size: tuple[int,int]) -> pd.DataFrame:
    """
    Align raw gaze coords to video frame coords, cropping edges.
    """
    screen_w, screen_h = screen_size
    # drop invalid
    df = df[(df['CURRENT_FIX_X'] != '.') & (df['CURRENT_FIX_Y'] != '.')].copy()
    df['CURRENT_FIX_X'] = df['CURRENT_FIX_X'].astype(float)
    df['CURRENT_FIX_Y'] = df['CURRENT_FIX_Y'].astype(float)
    # offsets
    offset_x = (screen_w - video_width) / 2 if dataset == 'memento' else 0
    offset_y = (screen_h - video_height) / 2
    df['CURRENT_FIX_X'] = ((df['CURRENT_FIX_X'] - offset_x)
                           .clip(0, video_width - 1))
    df['CURRENT_FIX_Y'] = ((df['CURRENT_FIX_Y'] - offset_y)
                           .clip(0, video_height - 1))
    return df


def calculate_gaussian_map(frame_data: pd.DataFrame,
                           video_width: int,
                           video_height: int,
                           sigma: float) -> np.ndarray:
    """
    Build a smoothed fixation map for one frame of data.
    """
    fixation_map = np.zeros((video_height, video_width), dtype=float)
    for _, row in frame_data.iterrows():
        x, y = int(row['CURRENT_FIX_X']), int(row['CURRENT_FIX_Y'])
        if 0 <= x < video_width and 0 <= y < video_height:
            fixation_map[y, x] = 1.0
    return gaussian_filter(fixation_map, sigma=sigma)


def process_dataset(cfg: dict) -> dict[str, list[np.ndarray]]:
    """
    Load raw fixation tables, compute averaged heatmaps for each video.
    Returns a dict: {video_id: [heatmap_frame0, ..., heatmap_frameN]}.
    """
    data_dir = Path(cfg['data_directory'])
    video_dir = Path(cfg['video_directory'])
    num_frames = cfg['num_frames']
    screen_size = tuple(cfg['screen_resolution'])  # e.g. [1024, 768]

    # Aggregate per-video DataFrames
    video_dfs: dict[str, list[pd.DataFrame]] = {}
    for subj_folder in sorted(data_dir.iterdir()):
        fix_path = subj_folder / 'Output' / f"{subj_folder.name}_fixation.xls"
        if not fix_path.exists():
            continue
        df = read_data(fix_path)
        for vid in df['video_clip'].unique():
            video_dfs.setdefault(vid, []).append(df[df['video_clip'] == vid])

    # Precompute sigma
    sigma = calculate_sigma(
        cfg['distance_to_screen'],
        cfg['screen_height_in'],
        cfg['screen_resolution_y'],
        cfg['visual_angle']
    )

    averaged_maps: dict[str, np.ndarray] = {}

    for vid, subject_list in tqdm(video_dfs.items(), desc='Videos'):
        vid_path = video_dir / vid
        vid_w, vid_h = get_video_dimensions(vid_path)
        total_frames = get_frame_count(vid_path)
        frame_bins = total_frames // num_frames
        frame_centers = [i*frame_bins + frame_bins//2 for i in range(num_frames)]

        sums = [np.zeros((vid_h, vid_w), dtype=float) for _ in range(num_frames)]
        counts = [0]*num_frames

        for df_subj in subject_list:
            df_subj = df_subj.fillna({'VIDEO_FRAME_INDEX_START': 0,
                                       'VIDEO_FRAME_INDEX_END': 0})
            df_subj['VIDEO_FRAME_INDEX_START'] = df_subj['VIDEO_FRAME_INDEX_START'].astype(int)
            df_subj['VIDEO_FRAME_INDEX_END'] = df_subj['VIDEO_FRAME_INDEX_END'].astype(int)
            adj = adjust_gaze_coordinates(df_subj, cfg['dataset'], vid_w, vid_h, screen_size)
            for i, center in enumerate(frame_centers):
                frame_data = adj[(adj['VIDEO_FRAME_INDEX_START'] <= center) &
                                 (adj['VIDEO_FRAME_INDEX_END'] >= center) &
                                 (adj.get('repeat', 0) == 0)]
                if not frame_data.empty:
                    gm = calculate_gaussian_map(frame_data, vid_w, vid_h, sigma)
                    sums[i] += gm
                    counts[i] += 1

        # Average
        averaged_maps[vid] = [sums[i] / counts[i] if counts[i]>0 else sums[i]
                              for i in range(num_frames)]
    return averaged_maps


def save_heatmaps(heatmaps: dict[str, list[np.ndarray]], out_path: Path) -> None:
    """
    Serialize the heatmap dict to a pickle file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(heatmaps, f)


def main():
    parser = argparse.ArgumentParser(description='Preprocess eye-tracking fixations into heatmaps')
    parser.add_argument('-c', '--config', type=Path, required=True, help='Path to YAML config')
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    heatmaps = process_dataset(cfg)
    save_heatmaps(heatmaps, Path(cfg['output_file']))
    print(f"Saved heatmaps to {cfg['output_file']}")


if __name__ == '__main__':
    main()
