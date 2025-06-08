import argparse
from pathlib import Path
import yaml
import pickle
import numpy as np
from skimage.transform import resize
from scipy.special import kl_div
from scipy.stats import sem, percentileofscore
from skimage.io import imsave

def normalize(x, method='standard', axis=None):
    """
    Normalize an array according to the specified method.
    Methods: 'standard' (z-score), 'range' (min-max), 'sum' (unit sum).
    """
    x = np.array(x, copy=False)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape), int)
        shape[axis] = x.shape[axis]
        if method == 'standard':
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
        elif method == 'range':
            res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
        elif method == 'sum':
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError("Invalid normalization method")
    else:
        if method == 'standard':
            res = (x - np.mean(x)) / np.std(x)
        elif method == 'range':
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == 'sum':
            res = x / float(np.sum(x))
        else:
            raise ValueError("Invalid normalization method")
    return res


def auc_judd(saliency_map, fixation_map, jitter=True):
    """
    Compute AUC_Judd between saliency and binary fixation maps.
    """
    sal = np.array(saliency_map, copy=False)
    fix = normalize(fixation_map, method='range') > 0.5
    if not np.any(fix):
        return np.nan
    if sal.shape != fix.shape:
        sal = resize(sal, fix.shape, order=3, mode='nearest')
    if jitter:
        sal += np.random.rand(*sal.shape) * 1e-7
    sal = normalize(sal, method='range')
    S = sal.ravel(); F = fix.ravel()
    S_fix = S[F]
    n_fix = len(S_fix); n_pix = len(S)
    thresholds = sorted(S_fix, reverse=True)
    tp = np.zeros(len(thresholds)+2); fp = np.zeros(len(thresholds)+2)
    tp[0], tp[-1] = 0, 1; fp[0], fp[-1] = 0, 1
    for k, thr in enumerate(thresholds):
        above = np.sum(S >= thr)
        tp[k+1] = (k+1)/float(n_fix)
        fp[k+1] = (above - k - 1)/float(n_pix - n_fix)
    return np.trapz(tp, fp)


def nss(saliency_map, fixation_map):
    """Normalized Scanpath Saliency"""
    sal = np.array(saliency_map, copy=False)
    fix = normalize(fixation_map, method='range') > 0.5
    if sal.shape != fix.shape:
        sal = resize(sal, fix.shape)
    sal = normalize(sal, method='standard')
    return np.mean(sal[fix])


def cc(saliency_map, sal2):
    """Pearson CC between two maps"""
    m1, m2 = np.array(saliency_map, copy=False), np.array(sal2, copy=False)
    if m1.shape != m2.shape:
        m1 = resize(m1, m2.shape, order=3, mode='nearest')
    m1 = normalize(m1, method='standard'); m2 = normalize(m2, method='standard')
    return np.corrcoef(m1.ravel(), m2.ravel())[0,1]


def kld(sal_map, fix_map):
    """KL Divergence between saliency and fixation maps"""
    p = sal_map.astype(np.float64); q = fix_map.astype(np.float64)
    if p.sum()!=0: p/=p.sum()
    if q.sum()!=0: q/=q.sum()
    eps = np.finfo(np.float64).eps
    return np.sum(kl_div(q+eps, p+eps))


def calculate_scores(fixation_dict, saliency_dict, metrics, num_frames):
    """
    Compute average per-video metric over num_frames.
    Returns dict of {metric: {video_id: score}}
    """
    scores = {m:{} for m in metrics}
    for vid, sal_maps in saliency_dict.items():
        key_str = f"{vid}.mp4"
        if key_str not in fixation_dict:
            continue
        fix_maps = fixation_dict[key_str]
        for m in metrics:
            vals = []
            for i in range(min(num_frames, len(fix_maps))):
                if m=='AUC_Judd': s=auc_judd(sal_maps[i], fix_maps[i])
                elif m=='NSS': s=nss(sal_maps[i], fix_maps[i])
                elif m=='CC': s=cc(sal_maps[i], fix_maps[i])
                elif m=='KLD': s=kld(sal_maps[i], fix_maps[i])
                else: continue
                vals.append(0 if np.isnan(s) else s)
            scores[m][vid] = np.mean(vals) if vals else np.nan
    return scores


def calculate_similarity(fixation_dict, saliency_dict, num_frames):
    """
    For each video, compute self-similarity and others-similarity lists
    Returns two dicts: others, self
    """
    others_all, self_all = {}, {}
    for key in fixation_dict:
        vid = key[:-4]
        fix_maps = fixation_dict[key]
        other_sims=[]; self_sim=np.nan
        for j, sm_vid in saliency_dict.items():
            key2=f"{sm_vid}.mp4"
            if key2 not in fixation_dict:
                continue
            s_vals=[]
            for i in range(min(num_frames, len(fix_maps))):
                auc = auc_judd(saliency_dict[j][i], fix_maps[i])
                if not np.isnan(auc): s_vals.append(auc)
            if key==key2: self_sim=np.mean(s_vals) if s_vals else np.nan
            else: other_sims.append(np.mean(s_vals) if s_vals else np.nan)
        others_all[key]=[v for v in other_sims if not np.isnan(v)]
        self_all[key]=self_sim
    return others_all, self_all


def aggregate_percentiles(others_all, self_all):
    """
    Compute percentile of self within others for each video
    Returns dict of percentiles
    """
    percentiles={}
    for k, others in others_all.items():
        if not others: continue
        p = percentileofscore(others, self_all[k])
        if not np.isnan(p): percentiles[k]=p
    return percentiles


def calculate_mean_sem(scores):
    return {m:{'mean':np.nanmean(list(d.values())), 'SEM':sem(list(d.values()), nan_policy='omit')}
            for m, d in scores.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config', type=Path, required=True)
    args=parser.parse_args()
    cfg=yaml.safe_load(args.config.read_text())
    fix_dict=pickle.load(open(cfg['eyetracking_path'],'rb'))
    sal_dict=pickle.load(open(cfg['model_path'],'rb'))
    metrics = cfg.get('metrics',['AUC_Judd','NSS','CC','KLD'])
    scores = calculate_scores(fix_dict, sal_dict, metrics, cfg['num_frames'])
    others, selfs = calculate_similarity(fix_dict, sal_dict, cfg['num_frames'])
    percentiles = aggregate_percentiles(others, selfs)
    mean_sem = calculate_mean_sem(scores)
    # Save results
    out_dir=Path(cfg['figures_path'])
    out_dir.mkdir(parents=True, exist_ok=True)
    pickle.dump({'scores':scores,'percentiles':percentiles,'mean_sem':mean_sem}, open(out_dir/'metrics_results.pkl','wb'))
    print("Metrics computed and saved.")

if __name__=='__main__':
    main()
