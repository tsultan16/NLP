"""
    Library of various utility functions

"""
import json
import re
from unidecode import unidecode

def load_dataset(): 
    # load the evidence passages
    with open("data/evidence.json", "r") as train_file:
        knowledge_source = json.load(train_file)         

    # load the training data insttances
    with open("data/train-claims.json", "r") as train_file:
        train_data = json.load(train_file)

    # load the validation data instances
    with open("data/dev-claims.json", "r") as dev_file:
        val_data = json.load(dev_file)    

    return knowledge_source, train_data, val_data


# a sentence cleaner class, can add more cleaning functions as needed
class SentenceCleaner():
    def __init__(self):
        pass

    def remove_repeated_non_alphanumeric(self, s):
        pattern = re.compile(r'([^\w\s])\1+')
        return pattern.sub('', s)

    def no_alphanumeric(self, s):
        return not any(c.isalnum() for c in s)

    def clean(self, s):
        
        # convert unicode characters to equivalent asc-ii
        s = unidecode(s)
        # Remove URLs
        s = re.sub(r'http\S+|www.\S+', '', s)
        # remove sequences of repeated non-alphanumeric characters
        s = self.remove_repeated_non_alphanumeric(s)
        # remove sentences that do not contain any alphabets and numbers
        if self.no_alphanumeric(s):
            return ''

        return s    

    def clean_dataset(self, knowledge_source, train_data, val_data):
        # clean all sentences in the knowledge source
        knowledge_source_clean = {}
        for i, ev in knowledge_source.items():
            clean_passage = self.clean(ev)
            # skip empty cleaned passages
            if len(clean_passage.split(" ")) == 0:
                continue
            knowledge_source_clean[i] = clean_passage

        knowledge_source = knowledge_source_clean

        # clean all claim sentences
        for i, claim in train_data.items():
            train_data[i]['claim_text'] = self.clean(claim['claim_text'])
        for i, claim in val_data.items():
            val_data[i]['claim_text'] = self.clean(claim['claim_text'])

        return knowledge_source, train_data, val_data
    

