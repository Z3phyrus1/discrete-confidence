# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 09:52:54 2025

@author: WANGLIANGFU
"""
# -*- coding: utf-8 -*-


import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_1samp_test

working_dir = r'D:/xiangmu/results/decoding_idsess'        # 存放 npz 的根目录
figures_dir      = r'D:/xiangmu/figures/decoding_idsess_mean'   # 输出图目录
os.makedirs(figures_dir, exist_ok=True)

N_PERM = 500   # 置换次数
ALPHA  = 0.05   # 显著性阈值


def load_npz(pattern):
    files = glob.glob(pattern, recursive=True)
    acc_list, conf_list = [], []
    t_train, t_test = None, None
    missing_conf = []
    for fp in files:
        data = np.load(fp, allow_pickle=True)
        print(f"{os.path.basename(fp)}: {list(data.keys())}")  # 加入这行看每个文件的键
        if 'scores_gen_acc' in data:
            acc_list.append(data['scores_gen_acc'])
        if 'scores_gen_conf' in data:
            conf_list.append(data['scores_gen_conf'])
        else:
            missing_conf.append(fp)
        if t_train is None and 'times_train' in data:
            t_train = data['times_train']
        if t_test is None and 'times_test' in data:
            t_test = data['times_test']
        data.close()
    if missing_conf:
        print("\n以下文件缺少'scores_gen_conf'：")
        for f in missing_conf:
            print(f)
    return acc_list, conf_list, t_train, t_test, files

def cluster_perm(mat_stack, n_perm=N_PERM):

    X = mat_stack - 0.5
    T_obs, clusters, pvals, _ = permutation_cluster_1samp_test(
        X, n_permutations=n_perm, tail=1, out_type='mask', n_jobs=-1, verbose=False
    )
    sig_mask = np.zeros_like(T_obs, dtype=bool)
    for c, p in zip(clusters, pvals):
        if p <= ALPHA:
            sig_mask |= c
    mean_mat = mat_stack.mean(axis=0)
    return mean_mat, pvals, sig_mask

def plot_mean_with_contour(mean_mat, sig_mask, times_train, times_test,
                           title, save_path, vmin=0.4, vmax=0.6):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mean_mat, origin='lower', interpolation='lanczos',
                   extent=[times_test[0], times_test[-1], times_train[0], times_train[-1]],
                   aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    if sig_mask.any():
        ax.contour(times_test, times_train, sig_mask.astype(int),
                   levels=[0.5], colors='k', linewidths=1.0, alpha=0.9)
    cs = ax.contour(times_test, times_train, mean_mat,
                    levels=[0.5], colors='k', linewidths=0.6, linestyles='--')
    ax.clabel(cs, fmt="%.2f", fontsize=8)

    ax.axhline(0, color='k', lw=1)
    ax.axvline(0, color='k', lw=1)
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='AUC')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVE] {save_path}")

if __name__ == "__main__":
    pattern = os.path.join(working_dir, '**', '*generalization_results.npz')
    acc_list, conf_list, times_train, times_test, files = load_npz(pattern)
    print(f"找到 {len(files)} 个 npz 文件")

    if acc_list:
        acc_stack = np.array(acc_list)  # (n, n_train, n_test)
        mean_acc, p_acc, sig_acc = cluster_perm(acc_stack, n_perm=N_PERM)
        plot_mean_with_contour(mean_acc, sig_acc, times_train, times_test,
                               title=f'Mean Acc (n={len(acc_list)})',
                               save_path=os.path.join(figures_dir, 'mean_gen_acc.png'),
                               vmin=0.4, vmax=0.6)
    else:
        print("[WARN] 没有 scores_gen_acc")

    if conf_list:
        conf_stack = np.array(conf_list)
        mean_conf, p_conf, sig_conf = cluster_perm(conf_stack, n_perm=N_PERM)
        plot_mean_with_contour(mean_conf, sig_conf, times_train, times_test,
                               title=f'Mean Conf (n={len(conf_list)}) ',
                               save_path=os.path.join(figures_dir, 'mean_gen_conf.png'),
                               vmin=0.4, vmax=0.6)
    else:
        print("[WARN] 没有 scores_gen_conf")