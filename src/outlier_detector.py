import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import LocalOutlierFactor


class OutlierDetector:
    def __init__(
        self,
        embeddings: np.ndarray,
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
        self.method = method
        self.percentile = percentile
        self.lof_n = lof_n_neighbors
        self.eps = epsilon

    def compute_medoid_and_ssw(self):
        # カラム全体で一度だけメドイドを計算
        dmat = pairwise_distances(self.emb)
        sumd = dmat.sum(axis=1)
        medoid_idx = np.argmin(sumd)
        medoid = self.emb[medoid_idx]
        # 全サンプルに対する SSE（Sum of Squared Errors）
        ssw = ((self.emb - medoid) ** 2).sum()
        return medoid, ssw

    def detect(self):
        # 1) カラム全体のメドイドと SSE を取得
        medoid, ssw = self.compute_medoid_and_ssw()
        max_ssw = ssw  # カラム全体では単一値

        # 2) コサイン類似度用に L2 正規化
        emb_norm = self.emb / np.linalg.norm(self.emb, axis=1, keepdims=True)
        medoid_norm = medoid / np.linalg.norm(medoid)

        # 3) 重み付き逸脱スコア：w * (1 − cosine_sim)
        w = max_ssw / (ssw + self.eps)
        sim = emb_norm.dot(medoid_norm)              # shape [n_samples]
        scores = w * (1.0 - sim)

        # 4) 閾値判定 or LOF
        if self.method == "percentile":
            thresh = np.percentile(scores, self.percentile)
            flags = scores > thresh
        else:
            lof = LocalOutlierFactor(n_neighbors=self.lof_n, novelty=True)
            lof.fit(self.emb)
            flags = lof.predict(self.emb) < 0

        return flags, scores
