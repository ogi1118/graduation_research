import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import LocalOutlierFactor


class OutlierDetector:
    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        method: str = "percentile",    # "percentile" or "lof"
        percentile: int = 95,          # 上位何％を逸脱とみなすか
        lof_n_neighbors: int = 20,     # LOF を使う場合の近傍数
        epsilon: float = 1e-6          # 動的重み計算の ε
    ):
        """
        embeddings: shape [n_samples, dim]
        labels:     shape [n_samples], 各サンプルのクラスタラベル
        """
        self.emb = embeddings
        self.labels = np.array(labels)
        self.method = method
        self.percentile = percentile
        self.lof_n = lof_n_neighbors
        self.eps = epsilon

    def compute_medoid_and_ssw(self):
        medoids = {}
        ssws = {}
        unique = set(self.labels)
        for cid in unique:
            if cid == -1:
                continue  # ノイズはスキップ
            idx = np.where(self.labels == cid)[0]
            vecs = self.emb[idx]
            dmat = pairwise_distances(vecs)
            sumd = dmat.sum(axis=1)
            medoid_idx = idx[np.argmin(sumd)]
            medoid = self.emb[medoid_idx]
            ssw = ((vecs - medoid) ** 2).sum()
            medoids[cid] = medoid
            ssws[cid] = ssw
        return medoids, ssws

    def detect(self):
        # 1) メドイドとクラスタ内 SSE を取得
        medoids, ssws = self.compute_medoid_and_ssw()
        max_ssw = max(ssws.values()) if ssws else 0.0

        # 2) 埋め込みとメドイドを正規化（コサイン類似度算出のため）
        emb = self.emb
        # 各ベクトルを行ごとに L2 正規化
        emb_norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        # メドイドも正規化
        medoid_norm = {
            cid: m / np.linalg.norm(m)
            for cid, m in medoids.items()
        }

        # 3) スコア計算：w*(1−cosine_sim)
        scores = np.zeros(len(emb))
        for i, cid in enumerate(self.labels):
            if cid == -1 or cid not in medoid_norm:
                scores[i] = 0.0
            else:
                # 重み
                w = max_ssw / (ssws[cid] + self.eps)
                # コサイン類似度
                sim = float(emb_norm[i].dot(medoid_norm[cid]))
                # 「離れているほど大きい」スコア
                scores[i] = w * (1.0 - sim)

        # 4) 閾値判定
        if self.method == "percentile":
            thresh = np.percentile(scores, self.percentile)
            flags = scores > thresh
        else:
            lof = LocalOutlierFactor(n_neighbors=self.lof_n, novelty=True)
            lof.fit(emb)
            flags = lof.predict(emb) < 0

        return flags, scores
