"""
    Util functions: document store loading, cleaning, etc.

"""
import json


def load_data(clean=False):
    # load the evidence passages
    with open("project-data/evidence.json", "r") as train_file:
        document_store = json.load(train_file)         
    print(f"Number of evidence passages: {len(document_store)}")

    # load the training data insttances
    with open("project-data/train-claims.json", "r") as train_file:
        train_data = json.load(train_file)
    print(f"Number of training instances: {len(train_data)}")

    # load the validation data instances
    with open("project-data/dev-claims.json", "r") as dev_file:
        val_data = json.load(dev_file)    
    print(f"Number of validation instances: {len(val_data)}")

    if clean: 
        # we remove duplicate values from the document_store dictionary (we arbitrarily keep the first one)
        seen = set()
        document_store_no_duplicates = {}
        for key, value in document_store.items():
            if value not in seen:
                document_store_no_duplicates[key] = value
                seen.add(value)

        # remove all "bad" documents from the document store, except those that occur in claim gold evidence lists, we will define "bad" documents as ones that have less than 50 characters
        claim_evidence_list = [claim['evidences'] for claim in train_data.values()]
        claim_evidence_list = claim_evidence_list + [claim['evidences'] for claim in val_data.values()]
        claim_evidence_list = list(set([evidence for evidence_list in claim_evidence_list for evidence in evidence_list]))

        document_store = {i: evidence_text for i, evidence_text in document_store_no_duplicates.items() if len(evidence_text) >= 30 or i in claim_evidence_list}
        print(f"Number of evidence passages remaining after cleaning: {len(document_store)}")
    
    return document_store, train_data, val_data    