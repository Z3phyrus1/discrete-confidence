# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 11:56:00 2026

@author: WANGLIANGFU
"""


import os
import glob
import numpy as np
import utils

##############################
# 路径配置
##############################
working_dir = os.path.join('..', '..', '..', 'results', 'decoding_idsess')
figures_dir    = os.path.join('..', '..', '..', 'figures', 'decoding_idsess_mean')
os.makedirs(figures_dir, exist_ok=True)

##############################
# 汇总容器
##############################
t1_acc_list, t1_conf_list = [], []
t2_acc_list, t2_conf_list = [], []
times_t1, times_t2 = None, None

##############################
# 扫描文件
##############################
pattern = os.path.join(working_dir, '**', '*_sliding_scores.npz')
files = glob.glob(pattern, recursive=True)
print(f"找到 {len(files)} 个 npz 文件")

for fp in files:
    data = np.load(fp, allow_pickle=True)
    if times_t1 is None and 'times_t1' in data: times_t1 = data['times_t1']
    if times_t2 is None and 'times_t2' in data: times_t2 = data['times_t2']
    if 'scores_t1_acc' in data:  t1_acc_list.append(data['scores_t1_acc'])
    if 'scores_t1_conf' in data: t1_conf_list.append(data['scores_t1_conf'])
    if 'scores_t2_acc' in data:  t2_acc_list.append(data['scores_t2_acc'])
    if 'scores_t2_conf' in data: t2_conf_list.append(data['scores_t2_conf'])
    data.close()

def summarize_and_plot(arr_list, times, title_stub, fname_stub):

    if not arr_list or times is None:
        print(f"[WARN] {title_stub} 没有有效数据，跳过")
        return
    arr = np.array(arr_list)  # 期望形状: (n_files, n_splits, n_times) 或 (n_files, n_times)
    if arr.ndim == 3:
        # 合并 file 和 split -> (n_obs, n_times)
        arr_flat = arr.reshape(-1, arr.shape[-1])
    elif arr.ndim == 2:
        # 只有 (n_files, n_times)
        arr_flat = arr
    else:
        raise ValueError(f"Unexpected shape: {arr.shape}")

    # 均值/SEM
    mean = arr_flat.mean(axis=0)
    sem  = arr_flat.std(axis=0, ddof=1) / np.sqrt(arr_flat.shape[0])

    from mne.stats import permutation_cluster_1samp_test
    cluster_stats = permutation_cluster_1samp_test(
        arr_flat - 0.5,
        tail=1,
        n_permutations=1000,
        seed=12345,
        n_jobs=-1,
        out_type='mask',
        verbose=False
    )

    # 绘图
    save_path = os.path.join(figures_dir, f"{fname_stub}.png")
    utils.plot_with_cluster_highlight(
        times=times,
        scores_mean=mean,
        scores_sem=sem,
        permutation_result=cluster_stats,
        title=f"{title_stub} (n_files={len(arr_list)})",
        save_path=save_path,
        chance_level=0.5
    )
    print(f"[SAVE] {save_path}")

summarize_and_plot(t1_acc_list,  times_t1, "Task1 Resp - Accuracy",    "group_t1_acc")
summarize_and_plot(t1_conf_list, times_t1, "Task1 Resp - Confidence",  "group_t1_conf")
summarize_and_plot(t2_acc_list,  times_t2, "Task2 Stim - Accuracy",    "group_t2_acc")
summarize_and_plot(t2_conf_list, times_t2, "Task2 Stim - Confidence",  "group_t2_conf")

print("汇总完成。")