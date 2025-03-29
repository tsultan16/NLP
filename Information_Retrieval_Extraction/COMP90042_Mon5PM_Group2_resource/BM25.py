"""
    A BM25 Retreiver Implementation

    Author: Tanzid Sultan 
"""

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import lil_matrix, csr_matrix
from collections import defaultdict
from unidecode import unidecode
import numpy as np
from tqdm import tqdm
import math, random

# using scipy sparse array instead of dictionary for storing TFIDF
class BM25_retriever():
    def __init__(self, k=1.25, b=0.75, remove_stopwords=False, apply_stemming=False):
        self.k = k
        self.b = b
        self.remove_stopwords = remove_stopwords
        self.apply_stemming = apply_stemming
        self.tokenizer = RegexpTokenizer(r'\w+') 
        if remove_stopwords:
            self.stopwords = stopwords.words('english')
        if apply_stemming:
            self.stemmer = PorterStemmer()      


    def train(self, documents):
        # tokenizing documents
        print("Tokenizing documents...")
        documents = [self.tokenize(doc) for doc in tqdm(documents, total=len(documents))]
        self.vocab = list(set([w for doc in documents for w in doc]))
        self.word2idx = {w:i for i,w in enumerate(self.vocab)}
        self.TFIDF, self.inverted_index, self.doc_tfidf_norms = self.create_inverted_index(documents)
        
    def tokenize(self, sent, stemming=True):
        # Replace accented letters with regular letters
        sent = unidecode(sent)
        # tokenize into words, remove punctuation
        words = self.tokenizer.tokenize(sent.lower())
        # remove stopwords
        if self.remove_stopwords:
            words = [w for w in words if w not in self.stopwords]
        # apply stemming to each word
        if self.apply_stemming and stemming:    
            words = [self.stemmer.stem(w) for w in words]
        return words

    def create_inverted_index(self, documents):
        # compute term frequency and document frequencies
        TF, term_docs = self.compute_TF_weighted(documents)

        # create inverted index
        N = len(documents)
        TFIDF = lil_matrix((len(self.vocab), len(documents)))
        inverted_index = defaultdict(list)
        print(f"Computing TFIDF and creating inverted index...")
        for w, docs in tqdm(term_docs.items(), total=len(term_docs)):
            for d in sorted(list(docs)):
                TFIDF[self.word2idx[w], d] = TF[self.word2idx[w], d] * math.log10(N/len(docs))
                inverted_index[self.word2idx[w]].append(d)

        # compute document TFIDF vector norms
        print(f"Computing TFIDF vector norms...")
        doc_tfidf_norms = np.zeros(N)
        for d, words in tqdm(enumerate(documents), total=len(documents)):
            for w in words:
                doc_tfidf_norms[d] = doc_tfidf_norms[d] +  TFIDF[self.word2idx[w], d]**2
            doc_tfidf_norms[d] = math.sqrt(doc_tfidf_norms[d])

        return TFIDF, inverted_index, doc_tfidf_norms  
          
    # weighted TF for BM25
    def compute_TF_weighted(self, documents):
        term_docs = defaultdict(set)
        doc_length = defaultdict(float)
        Dtotal = 0
        print(f"Computing TF...")
        TF = lil_matrix((len(self.vocab), len(documents)))
        for d, words in tqdm(enumerate(documents), total=len(documents)):
            for w in words:
                TF[self.word2idx[w], d] += 1
                term_docs[w].add(d)
            doc_length[d] = len(words)
            Dtotal += len(words)
        Davg = Dtotal / len(documents)

        # compute BM25 weighted term frequencies
        TF_weighted = lil_matrix((len(self.vocab), len(documents)))
        for w, docs in term_docs.items():
            for d in docs:
                tf = TF[self.word2idx[w],d]
                TF_weighted[self.word2idx[w],d] = (tf * (self.k + 1)) / (tf + self.k * (1 - self.b + self.b * (doc_length[d]/Davg)))
        return TF_weighted, term_docs
    
    # @profile 
    def retrieve_docs(self, query, topk=1):
        # for retrieval, we will remove stopwords from the query
        query_words = self.tokenize(query.lower())

        # remove out of vocab words
        query_words = [w for w in query_words if w in self.vocab]
        if query_words == []:
            return [], []

        # get all documents which contain words from query
        docs = list(set(doc for w in query_words for doc in self.inverted_index[self.word2idx[w]]))
        # score all these documents
        word_indices = np.array([self.word2idx[w] for w in query_words]).reshape(-1)

        # calculate document scores
        scores = self.TFIDF[word_indices[:, None], docs].sum(axis=0) / self.doc_tfidf_norms[docs]
        scores = np.squeeze(np.array(scores))
        sorted_indices = np.argsort(scores)[::-1]
        best_indices = sorted_indices[:topk].tolist()
        topk_scores = scores[best_indices].tolist()
        topk_doc_indices = [docs[idx] for idx in best_indices]
        del scores
        del docs
        return topk_doc_indices, topk_scores


    def get_tfidf_scores(self, query):
        query_words_unstemmed = self.tokenize(query.lower(), stemming=False)
        # apply stemming separately
        query_words = [self.stemmer.stem(w) for w in query_words_unstemmed]
        # create a dictionary to store the TF-IDF score for each word in the query
        tfidf_scores = {}
        for i , word in enumerate(query_words):
            if word in self.vocab:
                tfidf_scores[query_words_unstemmed[i]] = self.TFIDF[self.word2idx[word], :].mean()
            else:
                tfidf_scores[query_words_unstemmed[i]] = 0
        return tfidf_scores

    def get_tf_scores(self, query):
        query_words_unstemmed = self.tokenize(query.lower(), stemming=False)
        # apply stemming separately
        query_words = [self.stemmer.stem(w) for w in query_words_unstemmed]
        # create a dictionary to store the TF score for each word in the query
        tf_scores = {}
        for i , word in enumerate(query_words):
            if word in self.vocab:
                tf_scores[query_words_unstemmed[i]] = self.TF[self.word2idx[word], :].sum()
            else:
                tf_scores[query_words_unstemmed[i]] = 0
        return tf_scores    


# @profile 
def eval(claims_dataset, passage_ids, retreiver, k_values=[10]):
    precision = [0] * len(k_values)
    recall = [0] * len(k_values)
    f1 = [0] * len(k_values)
    for claim in tqdm(claims_dataset, total=len(claims_dataset)):
        query = claim["claim_text"]
        gold_evidence_list = claim["evidences"]
        topk_doc_indices, best_scores = retreiver.retrieve_docs(query, topk=max(k_values))
        if topk_doc_indices == []:
            continue
        topk_evidence_ids = [passage_ids[idx] for idx in topk_doc_indices]

        for i, k in enumerate(k_values):
            # evaluation (precision, recall, F1)
            intersection = set(topk_evidence_ids[:k]).intersection(gold_evidence_list)
            p = len(intersection) / len(topk_evidence_ids[:k])
            r = len(intersection) / len(gold_evidence_list)
            precision[i] += p
            recall[i] += r
            f1[i] += (2*p*r/(p + r)) if (p+r) > 0 else 0


    avg_precision = [p/len(claims_dataset) for p in precision]
    avg_recall = [r/len(claims_dataset) for r in recall]
    avg_f1 = [f/len(claims_dataset) for f in f1]

    #print(f"k = {k} --> Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average F1: {avg_f1}")

    return avg_precision, avg_recall, avg_f1