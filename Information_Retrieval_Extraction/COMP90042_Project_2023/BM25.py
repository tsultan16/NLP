from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import lil_matrix, csr_matrix
from collections import defaultdict
from unidecode import unidecode
import numpy as np
from tqdm import tqdm
import math, random
from memory_profiler import profile

# using scipy sparse array instead of dictionary for storing TFIDF
class IR_System():
    def __init__(self, k = 1.25, b = 0.75, remove_stopwords=False, apply_stemming=False):
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
        
    def tokenize(self, sent):
        # Replace accented letters with regular letters
        sent = unidecode(sent)
        # tokenize into words, remove punctuation
        words = self.tokenizer.tokenize(sent.lower())
        # remove stopwords
        if self.remove_stopwords:
            words = [w for w in words if w not in self.stopwords]
        # apply stemming to each word
        if self.apply_stemming:    
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


# @profile 
def eval(train_claims, val_claims, passage_ids, retreiver, k=10, num_claims=None):
    precision_tot = 0
    recall_tot = 0
    f1_tot = 0
    num_computed = 0
    for claims_dataset in (train_claims, val_claims):
        for claim in tqdm(claims_dataset, total=len(claims_dataset)):
            query = claim["claim_text"]
            gold_evidence_list = claim["evidences"]
            topk_doc_indices, best_scores = retreiver.retrieve_docs(query, topk=10)
            if topk_doc_indices == []:
                continue
            topk_evidence_ids = [passage_ids[idx] for idx in topk_doc_indices]
            # evaluation (precision, recall, F1)
            intersection = set(topk_evidence_ids).intersection(gold_evidence_list)
            precision = len(intersection) / len(topk_evidence_ids)
            recall = len(intersection) / len(gold_evidence_list)
            precision_tot += precision
            recall_tot += recall
            f1_tot += (2*precision*recall/(precision + recall)) if (precision + recall) > 0 else 0 
            num_computed += 1
            if num_claims is not None and num_computed == num_claims:
                break
        if num_claims is not None and num_computed == num_claims:
            break

    avg_precision = precision_tot / num_computed
    avg_recall = recall_tot / num_computed
    avg_f1 = f1_tot / num_computed

    print(f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average F1: {avg_f1}")

    return avg_precision, avg_recall, avg_f1