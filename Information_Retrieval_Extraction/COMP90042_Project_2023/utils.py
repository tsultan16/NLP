"""
    Util functions: document store loading, cleaning, etc.

"""
import json
import torch
import pickle
import random

# loads and cleans the claims dataset
def load_data(clean=False, clean_threshold=20):
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
        # remove all "bad" documents from the document store, except those that occur in claim gold evidence lists, we will define "bad" documents as ones that have less than clean_threshold number of characters
        claim_evidence_list = [claim['evidences'] for claim in train_data.values()]
        claim_evidence_list = claim_evidence_list + [claim['evidences'] for claim in val_data.values()]
        claim_evidence_list = list(set([evidence for evidence_list in claim_evidence_list for evidence in evidence_list]))

        document_store = {i: evidence_text for i, evidence_text in document_store.items() if (len(evidence_text) >= clean_threshold) or (i in claim_evidence_list)}
        print(f"Number of evidence passages remaining after cleaning: {len(document_store)}")
    
    return document_store, train_data, val_data    


# loads pre-trained DPR passage embeddings and passage_ids
def load_dpr_passage_embeddings():
    # load embeddings from file
    evidence_passage_embeds = torch.load("dpr_embeddings/evidence_passage_simple_dpr_embeds.pt")
    # now load the passage_ids list from the pickle file
    with open("dpr_embeddings/passage_ids.pkl", "rb") as f: 
        passage_ids = pickle.load(f)
    return evidence_passage_embeds, passage_ids


def tokenize_passage(passage_text, tokenizer, block_size):
        positive_encoding = tokenizer.encode_plus(passage_text, add_special_tokens=False, return_offsets_mapping=False, return_attention_mask=False, return_token_type_ids=False)
        passage_idx = positive_encoding['input_ids']
        if len(passage_idx) > (block_size-2):
            start_pos = random.randint(0, max(0,len(passage_idx) - (block_size-2)))
            passage_idx = passage_idx[start_pos:start_pos+block_size-2]

        passage_idx = [tokenizer.cls_token_id] + passage_idx + [tokenizer.sep_token_id]    
        passage_idx = passage_idx + [tokenizer.pad_token_id]*(block_size-len(passage_idx))

        if len(passage_idx) > block_size:
            raise Exception(f"Sequence length {len(passage_idx)} is longer than max_length {block_size}!")

        passage_attn_mask  = [1 if idx != tokenizer.pad_token_id else 0 for idx in passage_idx]
        passage_idx = torch.tensor(passage_idx)
        passage_attn_mask = torch.tensor(passage_attn_mask)
        return passage_idx, passage_attn_mask


def tokenize_claim(claim_text, tokenizer, block_size):
        claim_encoding = tokenizer.encode_plus(claim_text, add_special_tokens=False, return_offsets_mapping=False, return_attention_mask=False, return_token_type_ids=False)
        claim_idx = claim_encoding['input_ids']
        claim_idx = [tokenizer.cls_token_id] + claim_idx + [tokenizer.sep_token_id]    
        claim_idx = claim_idx + [tokenizer.pad_token_id]*(block_size-len(claim_idx))

        if len(claim_idx) > block_size:
            raise Exception(f"Sequence length {len(claim_idx)} is longer than max_length {block_size}!")

        claim_attn_mask  = [1 if idx != tokenizer.pad_token_id else 0 for idx in claim_idx]
        claim_idx = torch.tensor(claim_idx)
        claim_attn_mask = torch.tensor(claim_attn_mask)
        return claim_idx, claim_attn_mask


def find_topk_evidence_dpr(dpr_model, tokenizer, claim_text, evidence_passage_embeds, passage_ids, block_size, k=5, device="cuda"):
    # tokenize claim text
    claim_idx, claim_attn_mask = tokenize_claim(claim_text, tokenizer, block_size)
    claim_idx = claim_idx.unsqueeze(0).to(device)
    claim_attn_mask = claim_attn_mask.unsqueeze(0).to(device)
    # get BERT embedding of claim
    claim_embedding = dpr_model.encode_queries(claim_idx, claim_attn_mask)
    # find topk passages 
    scores = torch.mm(evidence_passage_embeds, claim_embedding.T)
    topk_scores, topk_ids = torch.topk(scores.squeeze(1), k=k)
    topk_scores = topk_scores.squeeze().tolist()
    topk_ids = topk_ids.squeeze().tolist()
    # get passage ids
    topk_passage_ids = [passage_ids[i] for i in topk_ids]
    return topk_passage_ids, topk_scores

