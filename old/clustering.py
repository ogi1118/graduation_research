import sklearn
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.manifold import TSNE
from token_element import TokenElement
import numpy as np
import matplotlib.pyplot as plt
from preprocess import Preprocessor
import pandas as pd
import seaborn as sns

# pip install -U spacy
# python -m spacy download en_core_web_sm

class ClusterFinder:
    def __init__(self, model_file_name, pca=0.8) -> None:
        prep = Preprocessor(model_file_name)
        self.token_list = prep.get_token_list()
        self.vector_array_pca = prep.apply_pca(pca)
        print(self.vector_array_pca.shape)
        self.cluster_labels = None
        self.vector_2d = None

    def get_vector_array_pca(self):
        return self.vector_array_pca
    
    def get_token_list(self):
        return self.token_list
    
    def exec_dbscan(self,eps=2.0, min_samples=2):
        db = DBSCAN(metric="euclidean", eps=eps, min_samples=min_samples) # metric="cosine"
        db.fit(self.vector_array_pca)
        self.cluster_labels = db.labels_
        
    def exec_tsne_dbscan(self,eps=2.0, min_samples=2):
        tsne = TSNE(n_components=2, metric="euclidean", perplexity=10) # metric="cosine",
        vector_2d = tsne.fit_transform(self.vector_array_pca)
        db = DBSCAN(metric="euclidean", eps=eps, min_samples=min_samples) # metric="cosine"
        db.fit(vector_2d)
        self.cluster_labels = db.labels_
        self.vector_2d = vector_2d
    
    def exec_tsne_optics(self, min_samples=2):
        tsne = TSNE(n_components=2, metric="euclidean", perplexity=10) # metric="cosine",
        vector_2d = tsne.fit_transform(self.vector_array_pca)
        opt = OPTICS(min_samples=min_samples) # metric="cosine"
        opt.fit(vector_2d)
        self.cluster_labels = opt.labels_
        self.vector_2d = vector_2d

    def exec_optics_tsne(self, min_samples=2):
        opt = OPTICS(min_samples=min_samples, metric="cosine")
        opt.fit(self.vector_array_pca)
        tsne = TSNE(n_components=2, metric="cosine", perplexity=10)
        vector_2d = tsne.fit_transform(self.vector_array_pca)
        self.cluster_labels = opt.labels_
        self.vector_2d = vector_2d

    def print_cluster_labels(self):
        print(self.cluster_labels)


    def print_token_with_labels(self):
        cluster_dic = {}
        for token, cluster_label in zip(self.token_list, self.cluster_labels):
            token_string = token.get_token_string()
            print(token_string+" : "+str(cluster_label))
            cluster_dic.setdefault(cluster_label, []).append(token_string)
        return cluster_dic

    def visualize(self):
        vector_array = self.vector_array_pca
        tsne = TSNE(n_components=2, metric="euclidean", perplexity=10) # metric="cosine",
        vector_2d = tsne.fit_transform(vector_array)
        plt.scatter(vector_2d[:,0], vector_2d[:,1], c=self.cluster_labels+1, marker=".", label=self.cluster_labels)
        # plt.legend()
        plt.show()

    def visualize_by_seaborn(self):
        vector_2d = self.vector_2d
        xyc_dic = {}
        xyc_dic["X"] = []
        xyc_dic["Y"] = []
        xyc_dic["C"] = []
        for v, label in zip(vector_2d, self.cluster_labels):
            xyc_dic["X"].append(v[0])
            xyc_dic["Y"].append(v[1])
            xyc_dic["C"].append(label)
        data = pd.DataFrame(xyc_dic)
        palette = sns.color_palette("tab20")
        sns.scatterplot(x="X", y="Y", hue="C", data=data, palette=palette, legend="full")
        # plt.legend()
        plt.show()


if __name__ == '__main__':
    cluster_finder = ClusterFinder("tokens_gpbl_explored.pickle", pca=0.8) #pca 0.99 
    # cluster_finder = ClusterFinder("tokens_gpbl.pickle", pca=0.8) #pca 0.99 
    # cluster_finder.exec_tsne_dbscan(eps=7, min_samples=10) # eps=6, min_samples=10
    # cluster_finder.exec_tsne_optics(min_samples=15) # min_samples=20
    cluster_finder.exec_optics_tsne(min_samples=10) # 10
    cluster_dic = cluster_finder.print_token_with_labels()
    for key in cluster_dic.keys():
        print("key:{}, size:{}, cluster:{}".format(key, len(cluster_dic[key]), cluster_dic[key]))
    # cluster_finder.visualize()
    cluster_finder.visualize_by_seaborn()

# hq4qKtKLc2LnosrW7pSz
