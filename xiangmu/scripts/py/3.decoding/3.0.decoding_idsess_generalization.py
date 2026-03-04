# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 19:23:06 2025
Refactored for Single Trial Cross-Task Generalization
"""

import os
from glob import glob
import numpy as np
import pandas as pd
import re
from io import StringIO
from shutil import copyfile

copyfile('../common_settings.py', 'common_settings.py')
copyfile('../utils.py', 'utils.py')
import common_settings as CS
import utils
import mne
from mne.decoding import GeneralizingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__": 
    subject = 'sub10'
    trial   = '3'
    sub_id_int = int(re.search(r'\d+', subject).group())
    working_dir = os.path.join('../../..', 'data', 'clean_EEG', subject)
    print(f"Subject: {subject} (ID={sub_id_int}), Trial: {trial}")
    
    fit_par_path = os.path.join('../../..', 'data', 'behavior', 'idsess', 'fit_par.txt')
    with open(fit_par_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned = ''.join([ln.replace('"', '') for ln in lines])
    df_fit = pd.read_table(
        StringIO(cleaned),
        delim_whitespace=True,
        engine='python',
        dtype=str,
        keep_default_na=False
    )

    df_fit = df_fit.applymap(lambda x: x.strip())
    df_fit.columns = df_fit.columns.str.strip()

    target_id = f"{sub_id_int}_{int(trial):02d}"  # 例如 1_02
    row = df_fit[(df_fit['id'] == target_id) & (df_fit['model'] == 'sub2')]
    if row.empty:
        # 打印调试信息后强制停止
        print(f"[DEBUG] rows with id={target_id}:\n", df_fit[df_fit['id'] == target_id][['model','w','id']])
        raise ValueError(f"fit_par.txt 中未找到 ID={target_id} 且 model=sub2 的行")
    w_val = float(row['w'].iloc[0])
    print(f"[OK] W (sub2) = {w_val} for ID={target_id}")

    # ============================================================
    # 2) EEG files
    # ============================================================
    task1_stim_files = sorted(glob(os.path.join(working_dir, f'clean_epochs_task1_ICA_*{trial}_sess*.fif')))
    task1_resp_files = sorted(glob(os.path.join(working_dir, f'clean_epochs_response1_ICA_*{trial}_sess*.fif')))
    task2_stim_files = sorted(glob(os.path.join(working_dir, f'clean_epochs_task2_ICA_*{trial}_sess*.fif')))
    if len(task1_resp_files) == 0:
        raise FileNotFoundError(f"No Task1 Response files for trial {trial}")

    # ============================================================
    # 3) Behavior data
    # ============================================================
    behavior_path = os.path.join('../../..', 'data', 'behavior', 'id', 'data_wide_wmPred.txt')
    df_pred = pd.read_csv(behavior_path, sep=';')
    df_sub = df_pred[df_pred['id'] == sub_id_int].copy()
    beh_aligned = df_sub[df_sub['seq'].astype(str) == str(trial)].reset_index(drop=True)
    print(f"Behavior rows: {len(beh_aligned)}")

    # ============================================================
    # 4) Paths
    # ============================================================
    results_dir = os.path.join('../../..', 'results', 'decoding_idsess', subject, trial)
    figures_dir = os.path.join('../../..', 'figures', 'decoding_idsess', subject, trial)
    for d in [results_dir, figures_dir]:
        os.makedirs(d, exist_ok=True)

    # ============================================================
    # 5) Load EEG & preprocess
    # ============================================================
    print("Loading EEG...")
    # Task1 Stim (for acc labels)
    ep_t1_stim = mne.concatenate_epochs([mne.read_epochs(f, verbose=False) for f in task1_stim_files])

    # Task1 Resp (train)
    ep_t1_r_list = [mne.read_epochs(f, verbose=False) for f in task1_resp_files]
    for ep in ep_t1_r_list:
        ep.apply_baseline(CS.baseline_response)
    ep_t1_r = mne.concatenate_epochs(ep_t1_r_list).crop(-0.5, CS.tmax_response).resample(500)
    X_train, times_train = ep_t1_r.get_data(copy=False), ep_t1_r.times

    # Task2 Stim (test)
    ep_t2_s_list = [mne.read_epochs(f, verbose=False) for f in task2_stim_files]
    for ep in ep_t2_s_list:
        ep.apply_baseline(CS.baseline_task2)
    ep_t2_s = mne.concatenate_epochs(ep_t2_s_list).crop(CS.tmin_task2, CS.tmax_task2).resample(500)
    X_test, times_test = ep_t2_s.get_data(copy=False), ep_t2_s.times

    # Align lengths
    n_min = min(len(ep_t1_stim), len(ep_t1_r), len(ep_t2_s), len(beh_aligned))
    X_train = X_train[:n_min]
    X_test = X_test[:n_min]
    ep_t1_stim = ep_t1_stim[:n_min]
    ep_t1_r = ep_t1_r[:n_min]
    beh_aligned = beh_aligned.iloc[:n_min].reset_index(drop=True)
    print(f"Aligned samples: {n_min}")

    # ============================================================
    # 6) Labels
    # ============================================================
    y_train_acc = utils.compute_accuracy_from_epochs(ep_t1_stim, ep_t1_r)
    y_test_acc = beh_aligned['acc2'].astype(int).values
    y_train_conf = (beh_aligned['abs1'] > w_val).astype(int).values
    y_test_conf  = (beh_aligned['abs2'] > w_val).astype(int).values

    # ============================================================
    # 7) Decode
    # ============================================================
    clf = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', class_weight='balanced'))

    # Accuracy
    print("Running Gen: Acc (T1Resp -> T2Stim)...")
    gen_acc = GeneralizingEstimator(clf, scoring='roc_auc', n_jobs=-1, verbose=True)
    gen_acc.fit(X_train, y_train_acc)
    scores_acc = gen_acc.score(X_test, y_test_acc)
    utils.plot_temporal_generalization(
        scores=scores_acc, times_train=times_train, times_test=times_test,
        title=f'Acc Gen ({subject} Tr{trial})',
        save_path=os.path.join(figures_dir, f'{subject}_{trial}_gen_acc_cross_task.png'),
        vmin=0.4, vmax=0.6
    )

    # Confidence
    print("Running Gen: Conf (T1Resp -> T2Stim)...")
    gen_conf = GeneralizingEstimator(clf, scoring='roc_auc', n_jobs=-1, verbose=True)
    gen_conf.fit(X_train, y_train_conf)
    scores_conf = gen_conf.score(X_test, y_test_conf)
    utils.plot_temporal_generalization(
        scores=scores_conf, times_train=times_train, times_test=times_test,
        title=f'Conf Gen ({subject} Tr{trial})',
        save_path=os.path.join(figures_dir, f'{subject}_{trial}_gen_conf_cross_task.png'),
        vmin=0.4, vmax=0.6
    )

    # ============================================================
    # 8) Save
    # ============================================================
    save_dict = {
        'scores_gen_acc': scores_acc,
        'scores_gen_conf': scores_conf,
        'times_train': times_train,
        'times_test': times_test
    }
    np.savez(os.path.join(results_dir, f'{subject}_generalization_results.npz'), **save_dict)
    print("Done.")