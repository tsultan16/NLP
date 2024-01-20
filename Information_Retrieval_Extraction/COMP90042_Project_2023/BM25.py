from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict
from unidecode import unidecode
import numpy as np
from tqdm import tqdm
from memory_profiler import profile

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
        self.documents = documents
        self.TFIDF, self.inverted_index, self.doc_tfidf_norms = self.create_inverted_index()
        
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

    def create_inverted_index(self):
        N = len(self.documents)
        TFIDF = defaultdict(float)
        inverted_index = defaultdict(list)

        # compute term frequency and document frequencies
        TF, term_docs = self.compute_TF_weighted()

        # create inverted index
        print(f"Computing TFIDF and creating inverted index...")
        for w, docs in tqdm(term_docs.items(), total=len(term_docs)):
            for d in sorted(list(docs)):
                tfidf = TF[(w,d)] * math.log10(N/len(docs))
                inverted_index[w].append(d)
                TFIDF[(w,d)] = tfidf

        # compute document TFIDF vector norms
        print(f"Computing TFIDF vector norms...")
        doc_tfidf_norms = [0] * N
        for d, doc in tqdm(enumerate(self.documents), total=len(self.documents)):
            words = self.tokenize(doc)
            for w in words:
                doc_tfidf_norms[d] = doc_tfidf_norms[d] +  TFIDF[(w,d)]**2
            doc_tfidf_norms[d] = math.sqrt(doc_tfidf_norms[d])

        return TFIDF, inverted_index, doc_tfidf_norms  
          
    # weighted TF for BM25
    def compute_TF_weighted(self):
        TF = defaultdict(int)
        term_docs = defaultdict(set)
        doc_length = defaultdict(float)
        Dtotal = 0
        print(f"Computing TFIDF...")
        for d, doc in tqdm(enumerate(self.documents), total=len(self.documents)):
            words = self.tokenize(doc)
            for w in words:
                TF[(w, d)] += 1
                term_docs[w].add(d)
            doc_length[d] = len(words)
            Dtotal += len(words)
        Davg = Dtotal / len(self.documents)

        # compute BM25 weighted term frequencies
        TF_weighted = defaultdict(float)
        for (w,d), tf in TF.items():
            TF_weighted[(w,d)] = (tf * (self.k + 1)) / (tf + self.k * (1 - self.b + self.b * (doc_length[d]/Davg)))
        return TF_weighted, term_docs
    
    @profile 
    def retrieve_docs(self, query, topk=1):
        # for retrieval, we will remove stopwords from the query
        query_words = self.tokenize(query.lower())
        #print(f"query words: {query_words}")
        # get all documents which contain words from query
        docs = []
        for w in query_words:
            docs.extend(d for d in self.inverted_index[w]) 
        # remove duplicates
        docs = list(set(docs))
        #print(f"docs: {docs}")    
        # score all these documents
        
        """
        scores = np.zeros(len(docs))
        for i in range(len(docs)):
            d = docs[i]
            for w in query_words:
                scores[i] += self.TFIDF[(w,d)]
            scores[i] = scores[i] / self.doc_tfidf_norms[d]        
        #print(f"scores: {scores}")  

        sorted_indices = np.argsort(scores)[::-1]
        best_indices = sorted_indices[:topk].tolist()
        best_scores = scores[best_indices].tolist()
        topk_doc_indices = [docs[idx] for idx in best_indices]
        del scores  # Explicitly delete the scores variable to free up memory
        """

        # score all these documents
        topk_scores = []
        topk_doc_indices = []
        for d in docs:
            score = 0
            for w in query_words:
                score += self.TFIDF[(w,d)]
            score = score / self.doc_tfidf_norms[d]
            # maintain a list of largest topk scores seen so far and corresponding indices
            if len(topk_scores) < topk or score > min(topk_scores):
                if len(topk_scores) == topk:
                    min_idx = topk_scores.index(min(topk_scores))
                    topk_scores.pop(min_idx)
                    topk_doc_indices.pop(min_idx)
                topk_scores.append(score)
                topk_doc_indices.append(d)

        # sort top-k scores and their corresponding document indices
        sorted_indices = np.argsort(topk_scores)[::-1]
        topk_scores = [topk_scores[idx] for idx in sorted_indices]
        topk_doc_indices = [topk_doc_indices[idx] for idx in sorted_indices]
        
        return topk_doc_indices, topk_scores
