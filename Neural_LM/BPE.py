import collections
import re
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import string
from multiprocessing import Pool


"""
    Simple implementation of a Byte Pair Encoding tokenizer. Note that we don't use lowercase folding and punctuations (comma, period, question mark, etc.) are treated as individual words. The encode and decode input must be a sentence string. The learner input is a list of strings.
"""

class BPE():

    def __init__(self, max_vocab_size=100, eow_token="_", eos_token=None):
        self.eow_token = eow_token
        self.eos_token = eos_token
        self.max_vocab_size = max_vocab_size
        self.default_init_vocab = list(string.ascii_letters) 
        self.vocab = None
        self.merged_pairs = None
        self.subword2idx = None
        self.word_tokens = {}

    # splits corpus into words, returns a dictionary of word frequencies
    def init_vocab_words(self, corpus):
        vocab = set(self.default_init_vocab + [self.eow_token])
        if self.eos_token is not None:
                vocab.add(self.eos_token) 
        # get word frequencies
        word_freq = collections.defaultdict(int)
        num_words = 0
        for sentence in corpus:
            # split sentence across whitespaces and make all characters lower case (we use NLTK word_tokenize fucntion because
            # we want to keep punctuations as separate words)
            words = word_tokenize(sentence.strip()) 
            if self.eos_token is not None:
                words.append(self.eos_token) 
            num_words += len(words)    
            for word in words:
                for c in word:
                    vocab.add(c)
                # store the word key as a string containing tokens separated by whitespace
                word_key = " ".join(list(word+self.eow_token))
                word_freq[word_key] += 1

        print(f"Number of words in corpus: {num_words}")
        return vocab, word_freq

    # find all unique pairs of adjacent tokens and their counts
    def get_pairs(self, word_freq):
        pairs = collections.Counter()
        for word, freq in word_freq.items():
            tokens = word.split()
            for pair in zip(tokens[:-1], tokens[1:]):
                pairs[pair] += freq 
        return pairs

    # merge a pair of tokens (token learner)
    def merge_pair_learner(self, pair, word_freq):
        word_freq_new = collections.defaultdict(int)
        for word, freq in word_freq.items():
            # merge all occurances of the pair in every word
            tokens = word.split()
            i = 0
            while i < len(tokens)-1:
                if (tokens[i], tokens[i+1]) == pair:
                    tokens[i:i+2] = ["".join(pair)]
                else:    
                    i += 1
            word_new = " ".join(tokens) 
            word_freq_new[word_new] = freq

        return word_freq_new


    # learns a BPE subword vocabulary and merge rules from a given training corpus
    def learn(self, corpus):
        # get inital vocab, word and token pair frequencies
        vocab, word_freq = self.init_vocab_words(corpus)
        # performs mergers to learn the vocabulary
        merged_pairs = []        
        pbar = tqdm(total=self.max_vocab_size, desc="Building vocab. Num tokens added --> ")
        for _ in range(self.max_vocab_size):
            # get counts of all adjacent token pairs
            pairs = self.get_pairs(word_freq)
            if not pairs:
                break
            # get most frequent pair
            most_freq_pair = pairs.most_common(1)[0][0]
            # apply merger
            word_freq = self.merge_pair_learner(most_freq_pair, word_freq)
            # add merged token to vocab
            vocab.add("".join(most_freq_pair))
            merged_pairs.append(most_freq_pair)
            pbar.update(1)
            
        self.vocab = sorted(list(vocab))
        self.merged_pairs = merged_pairs
        self.subword2idx = {t:i for i,t in enumerate(self.vocab)}
        print("Done building vocab!")

    def precompute_word_tokens(self, corpus):    
        # precompute tokenized words
        word_tokens = collections.defaultdict(list)
        pbar = tqdm(total=len(corpus), desc="Precomputing word tokenizations for corpus words--> ")
        for sentence in corpus:
            sentence_words = set(word_tokenize(sentence.strip()))
            for word in sentence_words:
                if word not in word_tokens:
                    word_tokens[word] = self.tokenize_word(word)   
            pbar.update(1)
        self.word_tokens = word_tokens
        print("Done precomputing subword tokens for all words in corpus.")

    # merge a pair of tokens (token segmenter)
    def merge_pair_segmenter(self, pair, tokens):
        # merge all occurances of the pair in every word
        split_tokens = tokens.split()
        i = 0
        while i < len(split_tokens)-1:
            if (split_tokens[i], split_tokens[i+1]) == pair:
                split_tokens[i:i+2] = ["".join(pair)]
            else:    
                i += 1
        tokens_new = " ".join(split_tokens) 

        return tokens_new

    # encode a word into a list of BPE tokens
    def tokenize_word(self, word):
        # split word into individual characters separated by white spaces, also insert eow token
        word_tokens = " ".join(list(word)) + " "+ self.eow_token
        # now replace all occurances of pairs with the merged pair
        for pair in self.merged_pairs:
            word_tokens =  self.merge_pair_segmenter(pair, word_tokens)
            if len(word_tokens.split()) == 1:
                break
        word_tokens = word_tokens.split()
        return word_tokens

    # tokenize a corpus/sentence
    def tokenize_sentence(self, sentence):
        # split corpus into words
        corpus_words = word_tokenize(sentence.strip())
        if self.eos_token is not None:
                corpus_words.append(self.eos_token) 
        # now replace all occurances of pairs with the merged pair
        corpus_tokens = []
        for word in corpus_words:
            if word in self.word_tokens:
                tokens = self.word_tokens[word]
            else:    
                tokens = self.tokenize_word(word)
            corpus_tokens.extend(tokens)    
        return corpus_tokens    
    
    # tokenizes multiple sentences in parallel
    def tokenize_sentences(self, sentences, num_procs=4):
        with Pool(num_procs) as p:
            tokenized_sentences = p.map(self.tokenize_sentence, sentences)
        return tokenized_sentences    

    # convert sentence to list of integer indices of subword tokens
    def encode(self, sentences):
        # convert sentence to subword tokens
        tokenized_sentences = self.tokenize_sentences(sentences)
        # convert subwords tokens to indices
        tokenized = []
        for tokens in tokenized_sentences:
            tokens = [self.subword2idx[t] for t in tokens]
        tokenized.append(tokens)
        return tokenized
    
    # decoding token sequence back to a sentence is easy, just merge tokens and put a space after every token that ends with the "_" character
    def decode(self, sentences_idx):
        sentences_subwords = []
        for tokens_idx in sentences_idx:
            subwords = [self.vocab[idx] for idx in tokens_idx]
        sentences_subwords.append(subwords)
        
        decoded_sentences = []
        for subwords in sentences_subwords:
            decoded_sentence = ""
            for token in subwords:
                if token.endswith(self.eow_token):
                    decoded_sentence = decoded_sentence + token.strip(self.eow_token) + " "
                else:
                    decoded_sentence = decoded_sentence + token    
            decoded_sentence = decoded_sentence.rstrip()   
            if self.eos_token:
                decoded_sentence = re.sub(self.eos_token, "\n", decoded_sentence)       
            decoded_sentences.append(decoded_sentence)     
        return decoded_sentences