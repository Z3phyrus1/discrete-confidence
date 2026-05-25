# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 13:48:12 2025
@author: WANGLIANGFU
"""
# -*- coding: utf-8 -*-
import os
from glob import glob
import numpy as np
import pandas as pd
import mne
import re
from shutil import copyfile

from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from mne.decoding import SlidingEstimator, cross_val_multiscore

copyfile('../common_settings.py', 'common_settings.py')
copyfile('../utils.py', 'utils.py')
import common_settings as CS
import utils as utils


if __name__ == "__main__":
    subject = 'sub18'
    trial = '1'
    test_size = 0.2
    n_splits = 100
    random_state = 12345
    
    sub_id_int = int(re.search(r'\d+', subject).group())
    working_dir = os.path.join('../../..', 'data', 'clean_EEG_dual', subject)
    
    print(f"\n===== START {subject} Trial {trial} =====")
    
    # =========================
    # 找文件
    # =========================
    task1_resp_files = sorted(glob(os.path.join(working_dir, f'clean_epochs_response1_ICA_*{trial}_sess*.fif')))
    task2_stim_files = sorted(glob(os.path.join(working_dir, f'clean_epochs_task2_ICA_*{trial}_sess*.fif')))
    task1_stim_files = sorted(glob(os.path.join(working_dir, f'clean_epochs_task1_ICA_*{trial}_sess*.fif')))
    
    print(f"task1_resp_files: {len(task1_resp_files)}")
    print(f"task2_stim_files: {len(task2_stim_files)}")
    print(f"task1_stim_files: {len(task1_stim_files)}")
    
    if len(task1_resp_files) == 0 or len(task2_stim_files) == 0 or len(task1_stim_files) == 0:
        raise RuntimeError("有文件没找到，先检查文件名模式和trial变量。")
    
    # =========================
    # 读行为
    # =========================
    behavior_path = os.path.join('../../..', 'data', 'raw_behavior', 'data_wide_wmPred.txt')
    fit_par_path = os.path.join('../../..', 'data', 'raw_behavior', 'fit_par.txt')
    
    df_pred = pd.read_csv(behavior_path, sep=';')
    w_val = utils.get_idsess_w_threshold(fit_par_path, sub_id_int)
    if w_val is None:
        raise RuntimeError("w_val is None")
    
    id_prefix = df_pred['id'].astype(str).str.split('_').str[0]
    df_sub = df_pred[id_prefix == str(sub_id_int)].copy()
    beh_aligned = df_sub[df_sub['seq'].astype(str) == str(trial)].reset_index(drop=True)
    
    print(f"len(df_sub)={len(df_sub)}, len(beh_aligned)={len(beh_aligned)}, w_val={w_val}")
    
    if len(beh_aligned) == 0:
        raise RuntimeError("beh_aligned为空")
    
    # =========================
    # =========================
    eps_t1r = [mne.read_epochs(f, verbose=False) for f in task1_resp_files]
    for e in eps_t1r:
        e.apply_baseline(CS.baseline_response)
    
    ep_t1_r = mne.concatenate_epochs(eps_t1r)
    ep_t1_r.crop(-0.5, CS.tmax_response).resample(100)
    
    X_t1 = ep_t1_r.get_data(copy=False)
    times_t1 = ep_t1_r.times
    print(f"X_t1 shape = {X_t1.shape}  (n_epochs={X_t1.shape[0]})")
    
    # =========================
    # 读 Task2 Stim EEG（展开 read_concat）
    # =========================
    eps_t2s = [mne.read_epochs(f, verbose=False) for f in task2_stim_files]
    for e in eps_t2s:
        e.apply_baseline(CS.baseline_task2)
    
    ep_t2_s = mne.concatenate_epochs(eps_t2s)
    ep_t2_s.crop(CS.tmin_task2, CS.tmax_task2).resample(100)
    
    X_t2 = ep_t2_s.get_data(copy=False)
    times_t2 = ep_t2_s.times
    print(f"X_t2 shape = {X_t2.shape}  (n_epochs={X_t2.shape[0]})")
    
    # ========================
    # =========================
    ep_t1_s = mne.concatenate_epochs([mne.read_epochs(f, verbose=False) for f in task1_stim_files])
    print(f"ep_t1_s n_epochs = {len(ep_t1_s)}")
    print(f"ep_t1_r n_epochs = {len(ep_t1_r)}")
    
    # 标签
    y_acc = utils.compute_accuracy_from_epochs(ep_t1_s, ep_t1_r)
    y_conf = (beh_aligned['abs1'] > w_val).astype(int).values
    
    print(f"len(y_acc)={len(y_acc)}, counts={np.unique(y_acc, return_counts=True)}")
    print(f"len(y_conf)={len(y_conf)}, counts={np.unique(y_conf, return_counts=True)}")
    
    print("\n===== LENGTH CHECK =====")
    print(f"Task1_Resp vs y_acc  : {len(X_t1)} vs {len(y_acc)}")
    print(f"Task2_Stim vs y_acc  : {len(X_t2)} vs {len(y_acc)}")
    print(f"Task1_Resp vs y_conf : {len(X_t1)} vs {len(y_conf)}")
    print(f"Task2_Stim vs y_conf : {len(X_t2)} vs {len(y_conf)}")
    
    # 你当前报错点大概率在这里：
    if len(X_t1) != len(y_conf):
        raise ValueError(f"[定位成功] Task1_Resp|Confidence 样本不一致: X_t1={len(X_t1)} y_conf={len(y_conf)}")
    
    # =========================
    # 若长度一致再跑（你可先注释掉）
    # =========================
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            solver='liblinear',
            class_weight='balanced',
            random_state=random_state,
            penalty='l1',
            Cs=np.logspace(-3, 3, 7),
            cv=5,
            scoring='roc_auc',
            max_iter=int(1e3),
            n_jobs=1,
            verbose=0,
        ),
    )
    slider = SlidingEstimator(clf, scoring='roc_auc', n_jobs=1, verbose=False)
    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    
    scores = cross_val_multiscore(slider, X_t1, y_conf, cv=cv, n_jobs=1, verbose=1)
    
    scores_mean = scores.mean(axis=0)
    scores_sem  = scores.std(axis=0) / np.sqrt(scores.shape[0])
    
    figures_dir = os.path.join('../../..', 'figures', 'decoding_idsess', subject, trial)
    results_dir = os.path.join('../../..', 'results', 'decoding_idsess', subject, trial)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 1) 普通解码曲线图
    utils.plot_time_decoding(
        times=times_t1,
        scores=scores_mean,
        title=f'Task1_Resp: Confidence ({subject} Tr{trial})',
        save_path=os.path.join(figures_dir, f'{subject}_{trial}_Task1_Resp_Confidence_sliding_basic.png'),
        chance_level=0.5
    )
    
    # 2) 显著性高光图（标准 cluster，不用你之前那套“几乎全高光”逻辑）
    T_obs, clusters, cluster_p_vals, H0 = mne.stats.permutation_cluster_1samp_test(
        scores - 0.5,
        n_permutations=2000,
        tail=1,
        threshold=None,
        out_type='mask',
        seed=12345,
        n_jobs=1,
        verbose=False
    )
    
    utils.plot_with_cluster_highlight(
        times=times_t1,
        scores_mean=scores_mean,
        scores_sem=scores_sem,
        permutation_result=(T_obs, clusters, cluster_p_vals, H0),
        title=f'Task1_Resp: Confidence ({subject} Tr{trial})',
        save_path=os.path.join(figures_dir, f'{subject}_{trial}_Task1_Resp_Confidence_sliding_cluster.png'),
        chance_level=0.5,
        alpha=0.05
    )
    
    # 3) 保存数值
    np.savez(
        os.path.join(results_dir, f'{subject}_{trial}_Task1_Resp_Confidence_scores.npz'),
        scores=scores,
        scores_mean=scores_mean,
        scores_sem=scores_sem,
        times=times_t1,
        T_obs=T_obs,
        cluster_p_vals=cluster_p_vals
    )
    print("scores shape:", scores.shape)
    print("DONE")