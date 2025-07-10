from outlier_detector import OutlierDetector

def add_outlier_info(
    merged_infos,
    method="percentile",
    percentile=95,
    lof_n_neighbors=20,
    epsilon=1e-6
):
    """
    merge_and_recluster の返り値 merged_infos に対し、
    各ステージごとに OutlierDetector を適用し、
    'outlier_flags', 'outlier_scores' を追加して返す。
    """
    enhanced = []
    for info in merged_infos:
        emb = info["emb"]               # np.ndarray [n_samples, dim]
        labels = info["clusters"]       # List[int]
        od = OutlierDetector(
            embeddings=emb,
            labels=labels,
            method=method,
            percentile=percentile,
            lof_n_neighbors=lof_n_neighbors,
            epsilon=epsilon
        )
        flags, scores = od.detect()
        info["outlier_flags"] = flags.tolist()
        info["outlier_scores"] = scores.tolist()
        enhanced.append(info)
    return enhanced
