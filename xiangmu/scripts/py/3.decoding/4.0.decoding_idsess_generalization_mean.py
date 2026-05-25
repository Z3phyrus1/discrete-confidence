# -*- coding: utf-8 -*-

import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_1samp_test

working_dir = os.path.join('..', '..', '..', 'results', 'decoding')
figures_dir = os.path.join('..', '..', '..', 'figures', 'decoding_mean')
os.makedirs(figures_dir, exist_ok=True)

N_PERM = 500
ALPHA = 0.05


def _safe_get(data, key):
    if key not in data:
        return None
    x = data[key]
    if isinstance(x, np.ndarray) and x.dtype == object and x.shape == ():
        x = x.item()
    if x is None:
        return None
    x = np.asarray(x)
    if x.size == 0:
        return None
    return x


def _same_1d(a, b, tol=1e-12):
    return (len(a) == len(b)) and np.allclose(a, b, atol=tol, rtol=0)


def _subject_from_path(fp):
    m = re.search(r'(sub\d+)', fp)
    return m.group(1) if m else None


def load_grouped_npz(pattern):
    files = sorted(glob.glob(pattern, recursive=True))
    print(f"找到 {len(files)} 个 npz 文件")

    by_sub_acc = {}
    by_sub_conf = {}
    t_train_ref, t_test_ref = None, None

    for fp in files:
        sub = _subject_from_path(fp)
        if sub is None:
            print(f"[WARN] 无法识别被试: {fp}")
            continue

        data = np.load(fp, allow_pickle=True)

        t_train = _safe_get(data, 'times_train')
        t_test = _safe_get(data, 'times_test')
        s_acc = _safe_get(data, 'scores_gen_acc')
        s_conf = _safe_get(data, 'scores_gen_conf')

        data.close()

        # times 必须存在
        if t_train is None or t_test is None:
            print(f"[WARN] 缺少 times，跳过: {fp}")
            continue

        # 建立参考时间轴
        if t_train_ref is None:
            t_train_ref = t_train
        if t_test_ref is None:
            t_test_ref = t_test

        # 检查时间轴一致
        if (not _same_1d(t_train_ref, t_train)) or (not _same_1d(t_test_ref, t_test)):
            print(f"[WARN] 时间轴不一致，跳过: {fp}")
            continue

        # shape 检查（应为 n_train x n_test）
        expected_shape = (len(t_train_ref), len(t_test_ref))

        if s_acc is not None:
            if s_acc.shape == expected_shape:
                by_sub_acc.setdefault(sub, []).append(s_acc)
            else:
                print(f"[WARN] ACC shape不符 {s_acc.shape} != {expected_shape}: {fp}")

        if s_conf is not None:
            if s_conf.shape == expected_shape:
                by_sub_conf.setdefault(sub, []).append(s_conf)
            else:
                print(f"[WARN] CONF shape不符 {s_conf.shape} != {expected_shape}: {fp}")

    return by_sub_acc, by_sub_conf, t_train_ref, t_test_ref


def build_subject_stack(by_sub_dict):
    """
    先做被试内平均 trial，再返回 (n_sub, n_train, n_test)
    """
    mats = []
    used_subs = []
    for sub in sorted(by_sub_dict.keys()):
        arr = np.array(by_sub_dict[sub])  # (n_trials, n_train, n_test)
        if arr.ndim != 3 or arr.shape[0] == 0:
            continue
        mats.append(arr.mean(axis=0))
        used_subs.append(sub)

    if len(mats) == 0:
        return None, []
    return np.stack(mats, axis=0), used_subs


def cluster_perm(mat_stack, n_perm=N_PERM):
    # mat_stack: (n_sub, n_train, n_test)
    X = mat_stack - 0.5
    T_obs, clusters, pvals, _ = permutation_cluster_1samp_test(
        X, n_permutations=n_perm, tail=1, out_type='mask', n_jobs=1, verbose=False
    )
    sig_mask = np.zeros_like(T_obs, dtype=bool)
    for c, p in zip(clusters, pvals):
        if p <= ALPHA:
            sig_mask |= c

    mean_mat = mat_stack.mean(axis=0)
    return mean_mat, pvals, sig_mask


def plot_mean_with_contour(mean_mat, sig_mask, times_train, times_test, title, save_path, vmin=0.4, vmax=0.6):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        mean_mat, origin='lower', interpolation='lanczos',
        extent=[times_test[0], times_test[-1], times_train[0], times_train[-1]],
        aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax
    )

    if sig_mask.any():
        ax.contour(times_test, times_train, sig_mask.astype(int), levels=[0.5], colors='k', linewidths=1.0, alpha=0.9)

    cs = ax.contour(times_test, times_train, mean_mat, levels=[0.5], colors='k', linewidths=0.6, linestyles='--')
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
    pattern = os.path.join(working_dir, '**', '*_generalization_results.npz')
    by_sub_acc, by_sub_conf, t_train, t_test = load_grouped_npz(pattern)

    if t_train is None or t_test is None:
        raise RuntimeError("未读到有效 times_train/times_test")

    # ACC
    acc_stack, acc_subs = build_subject_stack(by_sub_acc)
    if acc_stack is None or acc_stack.shape[0] < 2:
        print(f"[WARN] ACC 被试数不足: {0 if acc_stack is None else acc_stack.shape[0]}")
    else:
        print(f"[INFO] ACC used subjects ({len(acc_subs)}): {acc_subs}")
        mean_acc, p_acc, sig_acc = cluster_perm(acc_stack, n_perm=N_PERM)
        plot_mean_with_contour(
            mean_acc, sig_acc, t_train, t_test,
            title=f'Mean Gen ACC (subject-level, n_sub={acc_stack.shape[0]})',
            save_path=os.path.join(figures_dir, 'mean_gen_acc.png'),
            vmin=0.4, vmax=0.6
        )

    # CONF
    conf_stack, conf_subs = build_subject_stack(by_sub_conf)
    if conf_stack is None or conf_stack.shape[0] < 2:
        print(f"[WARN] CONF 被试数不足: {0 if conf_stack is None else conf_stack.shape[0]}")
    else:
        print(f"[INFO] CONF used subjects ({len(conf_subs)}): {conf_subs}")
        mean_conf, p_conf, sig_conf = cluster_perm(conf_stack, n_perm=N_PERM)
        plot_mean_with_contour(
            mean_conf, sig_conf, t_train, t_test,
            title=f'Mean Gen CONF (subject-level, n_sub={conf_stack.shape[0]})',
            save_path=os.path.join(figures_dir, 'mean_gen_conf.png'),
            vmin=0.4, vmax=0.6
        )

    print("汇总完成。")