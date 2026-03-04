# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 16:30:44 2025

@author:  WANGLIANGFU
"""

import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from shutil import copyfile

copyfile('../common_settings.py', 'common_settings.py')
copyfile('../utils.py', 'utils.py')
import common_settings as CS
import utils as utils
import mne
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from mne.decoding import (
    CSP,
    GeneralizingEstimator,
    LinearModel,
    Scaler,
    SlidingEstimator,
    Vectorizer,
    cross_val_multiscore,
    get_coef,
)




if __name__ == "__main__":  
    subject = 'sub09'
    sub_id_int = int(re.search(r'\d+', subject).group())
    
    working_dir = os.path.join('../..', 'data', 'clean_EEG', subject)
    print(f"被试文件夹:  {working_dir}")
    
    task1_response_files = sorted(glob(os.path.join(working_dir, 'clean_epochs_response1_ICA_*.fif')))
    print(f"找到 {len(task1_response_files)} 个 Response1 数据文件")
    
    task2_files = sorted(glob(os.path.join(working_dir, 'clean_epochs_task2_ICA_*.fif')))
    print(f"找到 {len(task2_files)} 个 Task2 数据文件")
    
    task1_files = sorted(glob(os.path.join(working_dir, 'clean_epochs_task1_ICA_*.fif')))
    print(f"找到 {len(task1_files)} 个 Task1 数据文件")

    behavior_path = os.path.join('../..', 'data', 'behavior', 'data_wide_wmPred.txt')
    fit_par_path = os.path.join('../..', 'data', 'behavior', 'fit_par.txt')
    df_pred = pd.read_csv(behavior_path, sep=';')
    w_val = utils.get_w_threshold(fit_par_path, sub_id_int)
    
    if w_val is not None: 
        print(f"被试 {subject} 阈值 w = {w_val}")
        if 'id_sess' in df_pred.columns:
            print(f"  id_sess 值: {df_pred[df_pred['id']==sub_id_int]['id_sess'].unique()}")
    else:
        print(f"[WARN] 无法获取 W 阈值，自信度解码将被跳过")

    results_dir_base = os.path.join('../..', 'results', 'decoding_id', subject)
    figures_dir_base = os.path.join('../..', 'figures', 'decoding_id', subject)
    for f in [results_dir_base, figures_dir_base]:
        if not os.path.exists(f):
            os.makedirs(f)

    ############################
    ### Task1 Response ####
    ############################
    epochs_task1_response_list = []
    y_confidence_list = []
    
    print(f"开始处理 {len(task1_response_files)} 个 Response1 文件...")
    
    # 准备行为数据（保持原始顺序）
    df_sub = df_pred[df_pred['id'] == sub_id_int].copy()
    print(f"该被试共有 {len(df_sub)} 条行为数据")
    
    # 只取在文件中原顺序的最后出现的4个id_sess
    all_id_sess = df_sub['id_sess'].unique()
    print(f"所有 id_sess（原始顺序）: {all_id_sess}")
    
    # 取最后 4 个
    if len(all_id_sess) >= 4:
        id_sess_to_use = all_id_sess[-4:]
    else: 
        id_sess_to_use = all_id_sess
    
    print(f"将使用的 id_sess（按获取顺序）: {id_sess_to_use}")

    for i, filename in enumerate(task1_response_files):
        print(f"\n[{i+1}] 处理文件: {os.path.basename(filename)}")
        epochs = mne.read_epochs(filename, verbose=False)
        epochs.apply_baseline(CS.baseline_response)
        epochs_task1_response_list.append(epochs)
        
        print(f"    -> Epochs 数量: {len(epochs)}")
        
        if w_val is not None:
            basename = os.path.basename(filename)
            m = re.search(r'sub\d+_(.+?)-epo\.fif', basename)
            
            if m:
                file_id = m.group(1)
                print(f"    -> 文件标识符: {file_id}")
                
                log_name = f"{subject}_{file_id}_response1_drop_log.csv"
                log_path = os.path.join(working_dir, log_name)
                
                if os.path.exists(log_path):
                    df_log = pd.read_csv(log_path)
                    n_orig = len(df_log)
                    print(f"    -> 原始 Trial 总数: {n_orig}")

                    if i < len(id_sess_to_use):
                        target_id_sess = id_sess_to_use[i]
                        print(f"    -> 对应的 id_sess: {target_id_sess}")
                        
                        # 从行为数据中获取对应的切片
                        beh_data_for_file = df_sub[df_sub['id_sess'] == target_id_sess].copy().reset_index(drop=True)
                        print(f"    -> 匹配到行为数据:  {len(beh_data_for_file)} 条")
                        
                        if len(beh_data_for_file) == n_orig:
                            # 计算自信度
                            high_conf_all = (beh_data_for_file['abs1'] > w_val).astype(int).values
                            print(f"    -> 计算自信度:  {np.sum(high_conf_all)}/{len(high_conf_all)} 个高自信")
                            
                            # 使用 epochs.selection
                            selection = epochs.selection
                            y_labels = high_conf_all[selection]
                            
                            print(f"    -> 最终标签数:  {len(y_labels)}, Epochs 数: {len(epochs)}")
                            
                            if len(y_labels) == len(epochs):
                                y_confidence_list.append(y_labels)
                                print(f"    -> ✓ 成功匹配标签")
                            else: 
                                print(f"    -> ✗ 标签数不匹配!")
                        else: 
                            print(f"    -> ✗ 行为数据数量不匹配！期望 {n_orig}，实际 {len(beh_data_for_file)}")
                    else:
                        print(f"    -> ✗ 文件索引超出 id_sess 范围")
                else:
                    print(f"    -> ✗ 日志文件不存在")
            else:
                print(f"    -> ✗ 无法提取标识符")
    
    print(f"\n最终:  成功匹配 {len(y_confidence_list)}/{len(epochs_task1_response_list)} 个文件的标签")
    
    # 拼接 Epochs
    epochs_task1_response = mne.concatenate_epochs(epochs_task1_response_list)
    epochs_task1_response = epochs_task1_response.crop(-0.5, CS.tmax_response)
    epochs_task1_response.resample(100)
    
    X_task1 = epochs_task1_response.get_data(copy=False)
    y_task1_lr = epochs_task1_response.events[:, 2]
    
    # 拼接自信度标签
    y_task1_conf = None
    if len(y_confidence_list) == len(epochs_task1_response_list):
        y_task1_conf = np.concatenate(y_confidence_list)
    
    # 训练左右分类器
    clf_lr = utils.make_generalizing_estimator()
    clf_lr.fit(X_task1, y_task1_lr)
    
    # Task1 Stimulus
    epochs_task1_list = []
    for filename in task1_files:
        epochs = mne.read_epochs(filename, verbose=False)
        epochs_task1_list.append(epochs)
    epochs_task1 = mne.concatenate_epochs(epochs_task1_list)
    
    # 计算 正确/错误
    y_task1_acc = utils.compute_accuracy_from_epochs(epochs_task1, epochs_task1_response)
    
    # 训练正确/错误分类器
    clf_acc = utils.make_generalizing_estimator()
    clf_acc.fit(X_task1, y_task1_acc)
    
    # Task1 正确/错误 时间解码
    clf_td = utils.make_sliding_estimator()
    cv = StratifiedShuffleSplit(**CS.cv_args)
    scores_task1_acc = cross_val_multiscore(clf_td, X_task1, y_task1_acc, n_jobs=-1, verbose=1)
    
    # 置换检验
    permutation_t1_acc = mne.stats.permutation_cluster_1samp_test(scores_task1_acc - 0.5, tail=1, seed=12345, n_jobs=-1, out_type='mask')
    
    #使用带聚类高亮的绘图函数
    scores_task1_acc_mean = np.mean(scores_task1_acc, axis=0)
    scores_task1_acc_sem = np.std(scores_task1_acc, axis=0) / np.sqrt(scores_task1_acc.shape[0])
    times = epochs_task1_response.times
    
    utils.plot_with_cluster_highlight(
        times=times,
        scores_mean=scores_task1_acc_mean,
        scores_sem=scores_task1_acc_sem,
        permutation_result=permutation_t1_acc,
        title=f"Task1: Correct vs Incorrect Temporal Decoding ({subject})",
        save_path=os.path.join(figures_dir_base, f'{subject}_task1_acc_temporal_with_highlight.png'),
        chance_level=0.5
    )

    ############################
    ### 自信度解码 #####
    ############################
    if y_task1_conf is not None: 
        print("\n正在进行自信度解码...")
        scores_task1_conf = cross_val_multiscore(clf_td, X_task1, y_task1_conf, n_jobs=-1, verbose=1)
        
        # 置换检验
        permutation_t1_conf = mne.stats.permutation_cluster_1samp_test(scores_task1_conf - 0.5, tail=1, seed=12345, n_jobs=-1, out_type='mask')
        
        # 新增：使用带聚类高亮的绘图函数
        scores_task1_conf_mean = np.mean(scores_task1_conf, axis=0)
        scores_task1_conf_sem = np.std(scores_task1_conf, axis=0) / np.sqrt(scores_task1_conf.shape[0])
        
        utils.plot_with_cluster_highlight(
            times=times,
            scores_mean=scores_task1_conf_mean,
            scores_sem=scores_task1_conf_sem,
            permutation_result=permutation_t1_conf,
            title=f"Task1: High vs Low Confidence Temporal Decoding ({subject})",
            save_path=os.path.join(figures_dir_base, f'{subject}_task1_conf_temporal_with_highlight.png'),
            chance_level=0.5
        )
    else:
        print('自信度解码失败')

    # ============================================================
    # 保存结果
    # ============================================================
    print("\n" + "="*60)
    print("保存结果")
    print("="*60)
    

    ############################
    ### Task2 处理 ############
    ############################
    epochs_task2_list = []
    for filename in task2_files:  
        epochs = mne.read_epochs(filename, verbose=False)
        epochs_task2_list.append(epochs)
    epochs_task2 = mne.concatenate_epochs(epochs_task2_list)
    epochs_task2 = epochs_task2.crop(-0.5, CS.tmax_response)
    epochs_task2.resample(100)
    times_task2 = epochs_task2.times
    X_task2 = epochs_task2.get_data(copy=True)
    y_task2_lr = np.array([1 if e % 10 == 1 else 2 for e in epochs_task2.events[:, 2]])
           
    scores_task2_lr = clf_lr.score(X_task2, y_task2_lr)
    
    scores_task2_acc = clf_acc.score(X_task2, y_task1_acc)

    times_train = epochs_task1_response.times
    if y_task1_conf is not None: 
        print("\n正在进行自信度解码...")
        scores_task2_conf = cross_val_multiscore(clf_td, X_task2, y_task1_conf, n_jobs=-1, verbose=1)
        
        # 置换检验
        permutation_t2_conf = mne.stats.permutation_cluster_1samp_test(scores_task2_conf - 0.5, tail=1, seed=12345, n_jobs=-1, out_type='mask')
        
        # 新增：使用带聚类高亮的绘图函数
        scores_task2_conf_mean = np.mean(scores_task2_conf, axis=0)
        scores_task2_conf_sem = np.std(scores_task2_conf, axis=0) / np.sqrt(scores_task2_conf.shape[0])
        
        utils.plot_with_cluster_highlight(
            times=times,
            scores_mean=scores_task2_conf_mean,
            scores_sem=scores_task2_conf_sem,
            permutation_result=permutation_t2_conf,
            title=f"Task2 : High vs Low Confidence Temporal Decoding ({subject})",
            save_path=os.path.join(figures_dir_base, f'{subject}_task2_conf_temporal_with_highlight.png'),
            chance_level=0.5
        )
    else:
        print('自信度解码失败')
    # # 左右泛化图
    # utils.plot_temporal_generalization(
    #     times_train=times_train,
    #     times_test=times_task2,
    #     scores=scores_task2_lr,
    #     title=f'Task1_Response -> Task2 (Left/Right) - {subject}',
    #     save_path=os.path.join(figures_dir_base, f'{subject}_task1resp_to_task2_lr.png'),
    #     vmin=0.3, vmax=0.7
    # )

    # # 正确/错误泛化图
    # if scores_task2_acc is not None:
    #     utils.plot_temporal_generalization(
    #         times_train=times_train,
    #         times_test=times_task2,
    #         scores=scores_task2_acc,
    #         title=f'Task1_Response -> Task2 (Correct/Error) - {subject}',
    #         save_path=os.path.join(figures_dir_base, f'{subject}_task1resp_to_task2_acc.png'),
    #         vmin=0.3, vmax=0.7
    #     )

    # 保存结果（不保存置换检验结果）
    save_dict = {
        'scores_task1_acc': scores_task1_acc,
        'scores_task2_lr': scores_task2_lr,
        'times_train': times_train,
        'times_task2': times_task2,
    }
    
    if scores_task2_acc is not None:
        save_dict['scores_task2_acc'] = scores_task2_acc
    
    if y_task1_conf is not None:
        save_dict['scores_task1_conf'] = scores_task1_conf
    
    np.savez(os.path.join(results_dir_base, f'{subject}_cross_task_decoding.npz'), **save_dict)
    print(f"[INFO] 结果已保存到 {results_dir_base}")
    print("[INFO] 所有处理完成！")