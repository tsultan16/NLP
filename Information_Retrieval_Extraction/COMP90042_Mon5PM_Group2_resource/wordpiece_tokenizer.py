"""
    A WordPiece Tokenizer Implementation

    Author: Tanzid Sultan    
"""

from collections import defaultdict, Counter
import string, re
import unicodedata
from tqdm import tqdm
import random 
random.seed(1234)
from nltk.tokenize import word_tokenize
from joblib import Parallel, delayed
from math import ceil


class WordPieceTokenizer():
    def __init__(self, cleaning=False, lowercase=True):
        self.cleaning = cleaning
        self.lowercase = lowercase
        self.vocab = []
        self.word2int = {}
        self.int2word = {}
        # special tokens
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.invalid_chars = ('*', '~', '_', '^', '`', '+', '\\', '[', ']', '<', '>') # invalid character lexicon
        self.max_vocab_size = None

    @property
    def vocab_size(self):
        return len(self.vocab)    

    def mask_token_id(self):
        return self.word2int[self.mask_token]

    def pad_token_id(self):
        return self.word2int[self.pad_token]

    def cls_token_id(self):
        return self.word2int[self.cls_token]

    def unk_token_id(self):
        return self.word2int[self.unk_token]

    def sep_token_id(self):
        return self.word2int[self.sep_token]

    def vocab_size(self):
        return len(self.vocab)
    
    def clean_sentence(self, s):
        if self.cleaning:
            # apply lowercase folding
            if self.lowercase:
                s = s.lower()
            # removes all control characters and invalid characters, replaces multiple adjacent whitespace with single whitespace
            s = "".join(ch for ch in s if unicodedata.category(ch)[0] != 'C' and ch not in self.invalid_chars) 
            s = " ".join(s.split())    
        return s

    # computes pair scores for all pairs of splits in the affected words
    def compute_pair_scores(self, splits, affected_words, pair_scores, word_freqs):
        letter_freqs = defaultdict(int)
        pair_freqs = defaultdict(int)
        for word in affected_words:
            freq = word_freqs[word]
            split = splits[word]
            # if word only contains one split
            if len(split) == 1:
                letter_freqs[split[0]] += freq
                continue

            # count up every individual split and adjacent pair of splits 
            for i in range(len(split)-1):
                pair = (split[i], split[i+1])
                letter_freqs[split[i]] += freq
                pair_freqs[pair] += freq

            letter_freqs[split[-1]] += freq

        # update pair scores for affected pairs
        for pair, freq in pair_freqs.items():
            pair_scores[pair] = freq/(letter_freqs[pair[0]]*letter_freqs[pair[1]])

        return pair_scores

    # merges a pair of characters in all splits of affected words
    def merge_pair(self, c1, c2, splits):
        merged = c1 + c2.lstrip('#')
        affected_words = set()
        for word, split in splits.items():
            if c1 not in split or c2 not in split:
                continue
            i = 0
            while i < len(split):
                if i < len(split) - 1 and split[i] == c1 and split[i+1] == c2:
                    split[i:i+2] = [merged]
                    affected_words.add(word)
                i += 1
        return affected_words 


    # generates wordpiece vocabulary of subwords from a given corpus
    # the input corpus is a list of sentences
    def generate_vocab(self, corpus, max_vocab_size, num_augmented_words=0):
        self.max_vocab_size = max_vocab_size

        # pretokenize the corpus into words (useing NLTK words tokenizer which treats punctuation as separate tokens)
        print(f"Pretokenizing corpus into words and computing unigram counts...")
        corpus_tokens = [word_tokenize(self.clean_sentence(sentence)) for sentence in corpus] 
        # get unigram counts
        word_freqs = Counter(word for word_list in corpus_tokens for word in word_list)
                
        # initialize WordPiece vocabulary
        print(f"Generating WordPiece vocabulary with max_vocab_size={max_vocab_size}...")
        self.vocab = set()
        for word in word_freqs.keys():
            self.vocab.add(word[0])
            self.vocab.update('##' + letter for letter in word[1:])

        # now add special tokens
        self.vocab.update([self.pad_token, self.cls_token, self.unk_token, self.mask_token, self.sep_token])

        # generate splits
        print("Generating splits...")
        splits = {word: [c if i==0 else f"##{c}" for i,c in enumerate(word)] for word in word_freqs.keys()}

        # initialize dictionary to store pair scores
        pair_scores = defaultdict(int)

        # compute initial pair scores
        print("Computing initial pair scores...")
        pair_scores = self.compute_pair_scores(splits, splits.keys(), pair_scores, word_freqs)

       # generate the subword vocabulary
        pbar = tqdm(total=max_vocab_size, desc="Building vocab. Current vocab_size --> ")
        pbar.update(len(self.vocab))

        while len(self.vocab) < max_vocab_size:
            # get pair with largest score 
            max_score_pair = max(pair_scores.items(), key=lambda pair: pair[1])
            # add new subword to vocabulary
            subword = max_score_pair[0][0] + max_score_pair[0][1].lstrip('#')
            self.vocab.add(subword)
            # update splits 
            affected_words = self.merge_pair(*max_score_pair[0], splits)
            # remove scores of pairs that include either of the characters in the merged pair
            pair_scores = {pair: score for pair, score in pair_scores.items() if (pair[0] != max_score_pair[0][0] and pair[0] != max_score_pair[0][1] and pair[1] != max_score_pair[0][0] and pair[1] != max_score_pair[0][1])}
            # compute new pair scores
            pair_scores = self.compute_pair_scores(splits, affected_words, pair_scores, word_freqs)
            pbar.update(1)    

        self.vocab = sorted(list(self.vocab))

        if num_augmented_words > 0:
            # get most frequent words in corpus
            most_frequent_words = word_freqs.most_common(num_augmented_words)
            # add most frequent words to vocabulary
            for word in most_frequent_words:
                if word[0] not in self.vocab:
                    self.vocab.append(word[0])

        self.word2int = {word:i for i,word in enumerate(self.vocab)}
        self.int2word = {i:word for i,word in enumerate(self.vocab)}


    def encode(self, sentences, return_subwords=False, verbose=False, num_procs=1):
            # Calculate the number of sentences per process
            sentences_per_proc = ceil(len(sentences) / num_procs)
            # Split sentences into chunks
            chunks = [sentences[i:i+sentences_per_proc] for i in range(0, len(sentences), sentences_per_proc)]
            # Encode each chunk in parallel
            encoded_chunks = Parallel(n_jobs=num_procs)(delayed(self.encode_chunk)(chunk, return_subwords) for chunk in chunks)
            # If return_subwords is True, encoded_chunks is a list of tuples. We need to separate the encoded sentences and subwords lists.
            if return_subwords:
                encoded_sentences, subwords_lists = zip(*[item for sublist in encoded_chunks for item in sublist])
                return encoded_sentences, subwords_lists
            else:
                # Flatten the list of chunks into a single list of encoded sentences
                encoded_sentences = [sentence for chunk in encoded_chunks for sentence in chunk]
                return encoded_sentences


    def encode_chunk(self, chunk, return_subwords):
        return [self.tokenize_sentence(sentence, return_subwords=return_subwords) for sentence in chunk]


    def tokenize_sentence(self, sentence, return_subwords=False):
        # first clean the sentence
        sentence = self.clean_sentence(sentence)
        # split the sentence into words (using NLTK word tokenizer) 
        words = word_tokenize(sentence)
        # tokenize the words into subwords
        subword_tokens = []
        for word in words:
            subword_tokens = subword_tokens + self.tokenize_word(word)
        # convert subwords to indices
        indices = [self.word2int[t] for t in subword_tokens]
        if return_subwords:
            return indices, subword_tokens

        return indices

    def tokenize_word(self, word):
        tokens = []
        if word in self.vocab:
            return [word]
        
        while len(word) > 0:
            i = len(word)    
            # find longest mactching subword in vocabulary
            while i > 0 and word[:i] not in self.vocab:
                i -= 1
            if i == 0:
                # no match found
                return [self.unk_token]
            # found longest subword
            tokens.append(word[:i])
            # iterate on remaining part of the word
            word = word[i:]
            # add prefix
            if len(word) > 0:
                word = f"##{word}"
        return tokens          


    def decode(self, idx_token_lists, num_procs=1):
        # Calculate the number of token lists per process
        token_lists_per_proc = ceil(len(idx_token_lists) / num_procs)
        # Split token_lists into chunks
        chunks = [idx_token_lists[i:i+token_lists_per_proc] for i in range(0, len(idx_token_lists), token_lists_per_proc)]
        # Decode each chunk in parallel
        decoded_chunks = Parallel(n_jobs=num_procs)(delayed(self.decode_chunk)(chunk) for chunk in chunks)
        # Flatten the list of chunks into a single list of decoded sentences
        decoded_sentences = [sentence for chunk in decoded_chunks for sentence in chunk]
        return decoded_sentences

    def decode_chunk(self, chunk):
        return [self.decode_idx_tokens(idx_tokens) for idx_tokens in chunk]

    def decode_idx_tokens(self, idx_tokens):
        # first convert indices to subword tokens
        subwords = self.idx_to_subwords(idx_tokens)
        # merge subwords
        i = 0
        while i < len(subwords)-1:
            a = subwords[i]
            b = subwords[i+1]
            if len(b) == 1:
                i += 1  
                continue
            if b[:2]=="##":
                subwords = subwords[:i] + [a+b.lstrip('#')] + subwords[i+2:]
            else:       
                i += 1    
        s = " ".join(subwords)
        return s

    def idx_to_subwords(self, idx):
        return [self.int2word[ix] for ix in idx]