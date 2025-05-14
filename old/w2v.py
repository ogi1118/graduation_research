from gensim.models import Word2Vec
import gensim.downloader as api
import pandas as pd
# import nltk
# from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
from token_element import TokenElement
import pickle

# pip install -U spacy
# python -m spacy download en_core_web_sm

class Word2Vec:
    def __init__(self, csv_file_name) -> None:
        home_dir = "/home/masaomi/gpbl_log/source/"
        # self.w2v_model = models.KeyedVectors.load_word2vec_format(home_dir+'GoogleNews-vectors-negative300.bin', binary=True)
        self.w2v_model = api.load('word2vec-google-news-300')
        self.df = pd.read_csv(csv_file_name, sep=";")
        # self.lemm = WordNetLemmatizer()
        self.spacy_model = spacy.load("en_core_web_sm")

    #def install_nltk(self):
    #    nltk.download('wordnet')
        # nltk.download()

    def get_stemmed_tokens_by_column(self, column_name):
        stemmed_tokens = []
        for row in self.df[column_name]:
            sentenses = sent_tokenize(row)
            #doc = self.spacy_model(row)
            #sentenses = doc.sents
            for sentense in sentenses:
                # tokens = word_tokenize(sentense)
                # for token in tokens:
                #     stemmed_token = self.lemm.lemmatize(token)
                #     if stemmed_token != "." or stemmed_token != ",":  
                #         stemmed_tokens.append(stemmed_token)
                tokens = self.spacy_model(sentense)
                for token in tokens:
                    if not ((token.pos_ in ("PUNCT", "NUM", "SYM")) or (token.is_stop)):
                        stemmed_token = token.lemma_
                        print(stemmed_token, flush=True)
                        if stemmed_token != "." and stemmed_token != ",":  
                            stemmed_tokens.append(stemmed_token)
        return stemmed_tokens
    
    def get_vectors(self, column_name):
        token_list = []
        stemmed_tokens = self.get_stemmed_tokens_by_column(column_name)
        for token in stemmed_tokens:
            try:
                emb_vector = self.w2v_model[token]
                te = TokenElement(token, emb_vector)
                token_list.append(te)
            except KeyError:
                pass
        return token_list
    
    def save(self, col_name, model_file_name):
        token_list = self.get_vectors(col_name)
        with open(model_file_name, 'wb') as f:
            pickle.dump(token_list, f)


if __name__ == '__main__':
    w2v = Word2Vec("dataset20240126.csv")
    # w2v.save("identified_problems_or_needs", "tokens_gpbl.pickle")
    w2v.save("data_or_information_to_be_further_explored", "tokens_gpbl_explored.pickle")
    print("done.")