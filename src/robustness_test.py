
import random
import os
import itertools

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from config_loader import config
from executer import Executer
from visualizer import Visualizer, CombinedVisualizer
from combined_visualizer_extensions import add_outlier_info

# 設定
SEEDS = list(range(10))
MAX_SENTENCES = 20
EPOCHS = 10
EPS = 0.7
MIN_SAMPLES = 3

# 結果格納
all_labels = []  # shape: [seed][stage][sample]
all_flags = []  # shape: [seed][stage][sample]

for seed in SEEDS:
    # 1) シード固定
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 2) executer + Visualizer.save_cluster_info で各ステージのデータ生成
    file_name = config["file_name"]
    col_names = config["col_names"]

    # 各カラムペアごとに処理
    for i in range(len(col_names) - 1):
        col_from = col_names[i]
        col_to = col_names[i + 1]
        executer = Executer(file_name, col_from, col_to)
        in_enc, out_enc, assoc, inp_t, out_t, in_f, out_f = executer.train(
            max_sentence_number=MAX_SENTENCES,
            epochs=EPOCHS
        )
        # AttentionEvaluator 経由で attention_values を取得し save_cluster_info
        ae = Visualizer(inp_t, out_t, in_f, out_f,
                        executer.get_sentence_lists(),
                        [in_enc.get_attention(
                            inp_t), out_enc.get_attention(out_t)],
                        device=executer.which_device(),
                        col_name_from=col_from,
                        col_name_to=col_to)
        ae.save_cluster_info(eps=EPS, min_samples=MIN_SAMPLES)

    # 3) merge → outlier 検知
    merged = CombinedVisualizer.merge_and_recluster(
        eps=EPS, min_samples=MIN_SAMPLES)
    merged = add_outlier_info(
        merged,
        method="percentile",
        percentile=95,
        lof_n_neighbors=20,
        epsilon=1e-6
    )

    # 4) 結果抽出
    labels_per_stage = [info["clusters"] for info in merged]
    flags_per_stage = [info["outlier_flags"] for info in merged]

    all_labels.append(labels_per_stage)
    all_flags.append(flags_per_stage)

# 5) ペアワイズで ARI と Jaccard を計算・表示
n_stages = len(all_labels[0])
ari_stats = {i: [] for i in range(n_stages)}
jaccard_stats = {i: [] for i in range(n_stages)}

for (i, labels_i), (j, labels_j) in itertools.combinations(enumerate(all_labels), 2):
    flags_i = all_flags[i]
    flags_j = all_flags[j]
    for stage in range(n_stages):
        # ARI
        ari = adjusted_rand_score(labels_i[stage], labels_j[stage])
        ari_stats[stage].append(ari)
        # Jaccard (Outlier 重複率)
        si = {idx for idx, f in enumerate(flags_i[stage]) if f}
        sj = {idx for idx, f in enumerate(flags_j[stage]) if f}
        union = si | sj
        jaccard = len(si & sj) / len(union) if union else 1.0
        jaccard_stats[stage].append(jaccard)

print("=== Robustness Evaluation ===")
for stage in range(n_stages):
    mean_ari = np.mean(ari_stats[stage])
    std_ari = np.std(ari_stats[stage])
    mean_jac = np.mean(jaccard_stats[stage])
    std_jac = np.std(jaccard_stats[stage])
    print(f"Stage {stage}: ARI = {mean_ari:.3f} ± {std_ari:.3f}, "
          f"Outlier Jaccard = {mean_jac:.3f} ± {std_jac:.3f}")
