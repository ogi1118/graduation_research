import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from token_element import TokenElement
import pickle
import numpy as np
import matplotlib.pyplot as plt

# pip install -U spacy
# python -m spacy download en_core_web_sm

class Preprocessor:
    def __init__(self, model_file_name) -> None:
        home_dir = "/home/masaomi/gpbl_log/source/"
        self.token_list = None
        vector_list = []
        with open(home_dir+model_file_name, 'rb') as f:
            self.token_list = pickle.load(f)
        if self.token_list is not None:
            for token in self.token_list:
                vector_list.append(list(token.get_vector()))
        self.vector_array = np.array(vector_list)

    def get_vector_array(self):
        return self.vector_array
    
    def get_token_list(self):
        return self.token_list
    
    def apply_pca(self, n_components):
        pca = PCA(n_components=n_components)
        pca.fit(self.vector_array)
        print(pca.score(self.vector_array))
        return pca.transform(self.vector_array)

    def apply_tsne(self):
        tsne = TSNE(n_components=2, metric="cosine", perplexity=10)
        print(self.vector_array.shape)
        return tsne.fit_transform(self.vector_array)
    
    # def test(self):
    #     tsne = TSNE(n_components=2, perplexity=5)
    #     array = np.array([[0.5,1,0,0,0],[1,1,0,0,0],[1,0,0,0,0],[0,0,0,1,1],[0,0,0,0,1],[1,1.2,0,0,0],[1,0.1,0,0,0],[0,0,0,1.2,1],[0,0,0,0.1,1]])
    #     print(array.shape)
    #     print(tsne.fit_transform(array))      


if __name__ == '__main__':
    # prep = Preprocessor("tokens_gpbl.pickle")
    prep = Preprocessor("tokens_gpbl_explored.pickle")
    token_list = prep.get_token_list()
    new_vector_array_pca = prep.apply_pca(0.8)
    print(new_vector_array_pca.shape)
    new_vector_array_tsne = prep.apply_tsne()
    print(new_vector_array_tsne.shape)
    plt.scatter(new_vector_array_tsne[:,0], new_vector_array_tsne[:,1])
    plt.show()
