# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 13:48:12 2025
@author: WANGLIANGFU
"""

import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from shutil import copyfile


import common_settings as CS
import utils as utils
import mne
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from mne.decoding import SlidingEstimator, cross_val_multiscore
import argparse
if __name__ == "__main__":  


    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="sub01")
    parser.add_argument("--trial",   type=str, default="2")
    args = parser.parse_args()

    # ==========================
    # 0. 参数设置
    # ==========================
    subject = args.subject
    trial = args.trial
    test_size = 0.2
    n_splits = 100
    random_state = 42
    
    sub_id_int = int(re.search(r'\d+', subject).group())
    working_dir = os.path.join('../..', 'data', 'clean_EEG', subject)
    print(f"被试: {subject}, Trial: {trial}")
    print(f"策略: SlidingEstimator (80% Train / 20% Test), 重复 {n_splits} 次")
    
    # ============================================================
    # 1. 准备数据
    # ============================================================
    task1_resp_files = sorted(glob(os.path.join(working_dir, f'clean_epochs_response1_ICA_*{trial}_sess*.fif')))
    task2_stim_files = sorted(glob(os.path.join(working_dir, f'clean_epochs_task2_ICA_*{trial}_sess*.fif')))
    task1_stim_files = sorted(glob(os.path.join(working_dir, f'clean_epochs_task1_ICA_*{trial}_sess*.fif')))

    behavior_path = os.path.join('../..', 'data', 'behavior', 'idsess', 'data_wide_wmPred.txt')
    fit_par_path = os.path.join('../..', 'data', 'behavior', 'idsess', 'fit_par.txt')
    df_pred = pd.read_csv(behavior_path, sep=';')
    w_val = utils.get_idsess_w_threshold(fit_par_path, sub_id_int)
    
    df_sub = df_pred[df_pred['id'].str.split('_').str[0].astype(int) == sub_id_int].copy()
    beh_aligned = df_sub[df_sub['seq'].astype(str) == str(trial)].reset_index(drop=True)

    # 读取 Epochs
    def read_concat(files, baseline, crop_tmin, crop_tmax):
        print(f"Loading {len(files)} files...")
        eps = [mne.read_epochs(f, verbose=False) for f in files]
        for e in eps: e.apply_baseline(baseline)
        epochs = mne.concatenate_epochs(eps)
        # 注意：resample 会导致 events 丢失或不准，如果后面不用 events 可忽略警告
        epochs.crop(crop_tmin, crop_tmax).resample(100) 
        return epochs

    print("Loading Task1 Response...")
    ep_t1_r = read_concat(task1_resp_files, CS.baseline_response, -0.5, CS.tmax_response)
    X_t1 = ep_t1_r.get_data(copy=False)
    times_t1 = ep_t1_r.times

    print("Loading Task2 Stimulus...")
    ep_t2_s = read_concat(task2_stim_files, CS.baseline_task2, CS.tmin_task2, CS.tmax_task2)
    X_t2 = ep_t2_s.get_data(copy=False)
    times_t2 = ep_t2_s.times
    
    ep_t1_s = mne.concatenate_epochs([mne.read_epochs(f, verbose=False) for f in task1_stim_files])
    
    # # 对齐
    # n_min = min(len(X_t1), len(X_t2), len(beh_aligned), len(ep_t1_s))
    # X_t1 = X_t1[:n_min]
    # X_t2 = X_t2[:n_min]
    # beh_aligned = beh_aligned.iloc[:n_min]
    # ep_t1_s = ep_t1_s[:n_min]
    # ep_t1_r = ep_t1_r[:n_min]

    # 标签
    y_acc = utils.compute_accuracy_from_epochs(ep_t1_s, ep_t1_r)
    y_conf = (beh_aligned['abs1'] > w_val).astype(int).values
    
    # print(f"Data Loaded. Trials: {n_min}")
    print(f"Acc counts: {np.unique(y_acc, return_counts=True)}")
    
    results_dir = os.path.join('../..', 'results', 'decoding_idsess', subject, trial)
    figures_dir = os.path.join('../..', 'figures', 'decoding_idsess', subject, trial)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # ============================================================
    # 核心函数：运行 SlidingEstimator 并绘图
    # ============================================================
    def run_sliding_analysis(X, y, times, task_name, label_name):
        print(f"\n--- Running: {task_name} | {label_name} ---")
        
        # 1. 定义分类器
        clf = make_pipeline(StandardScaler(), LogisticRegressionCV(solver='liblinear', class_weight='balanced',random_state=random_state,penalty='l1',Cs = np.logspace(-3,3,7),cv = 5,scoring = 'roc_auc',max_iter = int(1e3),n_jobs = 1,verbose = 0))
        slider = SlidingEstimator(clf, scoring='roc_auc', n_jobs=1, verbose=False)
        
        # 2. 定义 80/20 交叉验证
        cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
        
        # 3. 运行解码
        scores = cross_val_multiscore(slider, X, y, cv=cv, n_jobs=1,verbose = 1)
        print("Running permutation test...")
        cluster_stats = mne.stats.permutation_cluster_1samp_test(
            scores - 0.5, 
            tail=1, 
            n_permutations=1000,
            threshold = dict(start = 0,step = 0.2), # TFCE cluster method
            seed=12345, 
            n_jobs=1, 
            out_type='mask' # 确保 utils 能处理 mask 类型的 clusters
        )
        
        # 5. 计算绘图所需的均值和标准误
        scores_mean = np.mean(scores, axis=0)
        scores_sem = np.std(scores, axis=0) / np.sqrt(scores.shape[0])
        
        # 6. 绘图 (传入完整的 cluster_stats tuple)
        save_path = os.path.join(figures_dir, f'{subject}_{trial}_{task_name}_{label_name}_sliding.png')
        
        utils.plot_with_cluster_highlight(
            times=times,
            scores_mean=scores_mean,
            scores_sem=scores_sem,
            permutation_result=cluster_stats,  # <--- 传完整元组
            title=f"{task_name}: {label_name} ({subject})",
            save_path=save_path,
            chance_level=0.5
        )
        
        return scores

    # ============================================================
    # 执行分析
    # ============================================================
    
    # Task 1 Accuracy
    s_t1_acc = run_sliding_analysis(X_t1, y_acc, times_t1, "Task1_Resp", "Accuracy")
    # Task 1 Confidence
    s_t1_conf = run_sliding_analysis(X_t1, y_conf, times_t1, "Task1_Resp", "Confidence")
    
    # Task 2 Accuracy
    s_t2_acc = run_sliding_analysis(X_t2, y_acc, times_t2, "Task2_Stim", "Accuracy")
    # Task 2 Confidence
    s_t2_conf = run_sliding_analysis(X_t2, y_conf, times_t2, "Task2_Stim", "Confidence")
    # 保存
    print("Saving to:", os.path.abspath(results_dir))
    np.savez(
        os.path.join(results_dir, f'{subject}_{trial}_sliding_scores.npz'),
        scores_t1_acc=s_t1_acc,
        scores_t1_conf=s_t1_conf,
        scores_t2_acc=s_t2_acc,
        scores_t2_conf=s_t2_conf,
        times_t1=times_t1,
        times_t2=times_t2
    )
    print("\n完成！")