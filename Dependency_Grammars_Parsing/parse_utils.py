"""
    Neural Dependency Parser Utils
"""


import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import psutil


# function for reading CoNLL parse files
def read_conllu(file_path):
    """
    Read a CoNLL-U file and return a list of sentences, where each sentence is a list of dictionaries, one for each token.
    """
    with open(file_path, 'r') as f:
        sentences = f.read().strip().split('\n\n')
        examples = []
        for sentence in sentences:
            token_dicts = []
            for line in sentence.split('\n'):
                if line[0] == '#':
                    continue    
                token_dict = list(zip(['id', 'form', 'lemma', 'upostag', 'xpostag' , 'feats', 'head', 'deprel', 'deps', 'misc'], line.split('\t')))
                # only keep form, xpostag, head, and deprel
                token_dicts.append(dict([token_dict[1], token_dict[4], token_dict[6], token_dict[7]]))
            examples.append(token_dicts)
        return examples
    

# function for extracting all the tokens and labelled head-dependency relations from the data
def get_tokens_relations(data_instance):
    """
    Extract all the labeled dependency relations from the data.
    """
    tokens = []
    relations = []
    for token_id, token in enumerate(data_instance):
        head_id = int(token['head'])
        if head_id == 0:
            head = 'ROOT'
        else:
            head = data_instance[head_id - 1]['form']
        dependent = token['form']
        tokens.append((dependent, token_id+1))
        relation = token['deprel']
        relations.append(((head, head_id), (dependent, token_id+1), relation))
    return tokens, relations


# training oracle returns the state-action pairs from every step of the parsing process
# the state only consists of the top two words on the stack and top word on the buffer
def training_oracle(data_instance, return_states=False, max_iters=100, verbose=False):
    # get the tokens and relations for the refenrence parse 
    tokens, Rp = get_tokens_relations(data_instance)
    sentence_words = [t[0] for t in tokens]
    if verbose: 
        print(f"Sentence: {sentence_words}")
        print(f"Reference parse: {Rp}")

    head_dep = [(r[0], r[1]) for r in Rp]

    # intialize the stack and buffer
    stack = [('ROOT', 0), tokens[0]]
    buffer = tokens[1:]
    Rc = []
    states = None
    if return_states:
        states = [([('ROOT', 0)], tokens[0])]
    actions = ['SHIFT']
    labels = ['null']
    # parse the sentence to get the sequence of states and actions
    niters = 0
    
    if verbose: 
        print(f"\nStack: {stack}")
        print(f"Buffer: {buffer}")    

    while (buffer or len(stack) > 1) and niters < max_iters:
        # get top two elements of stack
        S1 = stack[-1]
        S2 = stack[-2] 
        niters += 1

        if return_states:
            if len(buffer) > 0:
                states.append((stack[-2:] , buffer[0]))
            else:
                states.append((stack[-2:], None))

        # check if LEFTARC possible
        if (S1, S2) in head_dep:
            # remove second element of stack
            stack.pop(-2)
            rel = Rp[head_dep.index((S1, S2))]
            Rc.append(rel)
            next_action = 'LEFTARC' 
            next_label = rel[2]
            arc = (S1, S2, rel[2])

        # check if RIGHTARC possible
        elif (S2, S1) in head_dep:
            # get all head-dependent relations with S1 as head
            S1_rels = [r for r in Rp if r[0] == S1]
            # check if all dependents of S1 are in Rc
            if all([r in Rc for r in S1_rels]):
                stack.pop(-1)
                rel = Rp[head_dep.index((S2, S1))]
                Rc.append(rel)
                next_action = 'RIGHTARC' 
                next_label = rel[2]
                arc = (S2, S1, rel[2])
            else:
                if len(buffer)==0:
                    if verbose: print(f"Error! Parse failed, no valid action available!")
                    return None, None, None, None
                stack.append(buffer.pop(0))
                next_action = 'SHIFT'
                next_label = 'null'
                arc = None

        # otherwise SHIFT    
        else:
            if len(buffer)==0:
                    if verbose: print(f"Error! Parse failed, no valid action available!")
                    return None, None, None, None
            stack.append(buffer.pop(0))
            next_action = 'SHIFT'
            next_label = 'null'
            arc = None

        actions.append(next_action)
        labels.append(next_label)
        if verbose:
            print(f"Action: {next_action}, Arc: {arc}")
            print(f"\nStack: {stack}")
            print(f"Buffer: {buffer}")
            print(f"Rc: {Rc}")      

    # make sure Rc and Rp are consistent
    assert all([r in Rc for r in Rp]) and len(Rc)==len(Rp), "Rc not consistent with Rp"

    if niters == max_iters:
        print("Maximum number of iterations reached!")  

    return states, actions, labels, sentence_words, Rc    


class DependencyParseDataset(Dataset):
    
    def __init__(self, sentences, state_action_label, action2idx, label2idx, block_size=256):
        self.sentences = sentences
        self.state_action_label = state_action_label
        self.action2idx = action2idx
        self.label2idx = label2idx
        self.block_size = block_size
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    def __len__(self):
        return len(self.sentences)
    
    def tokenize_sentence(self, sentence):
        input_encoding = self.tokenizer.encode_plus(sentence, is_split_into_words=True, return_offsets_mapping=False, padding=False, truncation=False, add_special_tokens=True)
        input_idx = input_encoding['input_ids']
        word_ids = input_encoding.word_ids()
        if len(input_idx) > self.block_size:
            raise ValueError(f"Tokenized sentence is too long: {len(input_idx)}. Truncation unsupported.")
        # add padding 
        input_idx = input_idx + [self.tokenizer.pad_token_id] * (self.block_size - len(input_idx))    
        # create attention mask 
        input_attn_mask = [1 if idx != self.tokenizer.pad_token_id else 0 for idx in input_idx]
        # convert to tensors
        input_idx = torch.tensor(input_idx)
        input_attn_mask = torch.tensor(input_attn_mask) 

        return input_idx, input_attn_mask, word_ids

    def tokenize_state(self, states, word_ids):
        state_idx = []
        for stack_words, buffer_word in states:
            state_words_idx = [self.tokenizer.pad_token_id] * 3  # missing words are filled with PAD token
            for i in range(len(stack_words)):
                if stack_words[i][0] == 'ROOT':
                    state_words_idx[i] = self.tokenizer.cls_token_id  # ROOT is represented by CLS token
                else:
                    state_words_idx[i] = word_ids.index(stack_words[i][1]-1)
            
            if buffer_word is not None:
                state_words_idx[2] = word_ids.index(buffer_word[1]-1)
            
            state_idx.append(state_words_idx)
        return state_idx    

    def __getitem__(self, idx):
        # get sentence 
        sentence = self.sentences[idx]
        # get states, actions, and labels
        states, actions, labels = self.state_action_label[idx]
        assert len(states) == len(actions) == len(labels), "Lengths of states, actions, and labels do not match."
        # tokenize the sentence
        input_idx, input_attn_mask, word_ids = self.tokenize_sentence(sentence)
        # map state words to index of first subword token
        state_idx = self.tokenize_state(states, word_ids)
        # map actions and labels to indices
        action_idx = [self.action2idx[a] for a in actions]
        label_idx = [self.label2idx[l] for l in labels]    

        return input_idx, input_attn_mask, state_idx, action_idx, label_idx   


def collate_fn(batch):
    # Separate the tensors and the dictionaries
    input_idxs, input_attn_masks, state_idx, action_idx, label_idx = zip(*batch)

    # Default collate the tensors
    input_idxs = torch.stack(input_idxs)
    input_attn_masks = torch.stack(input_attn_masks)

    return input_idxs, input_attn_masks, state_idx, action_idx, label_idx 


class BERT_ORACLE(torch.nn.Module):
    
    def __init__(self, num_actions, num_labels, num_features=3, unlabeled_arcs=True, dropout_rate=0.1, mlp_hidden_size=128):
        super().__init__()
        self.unlabeled_arcs = unlabeled_arcs
        # load pretrained BERT model
        self.bert_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = torch.nn.Dropout(dropout_rate)
        # define action classifier head (2 layer MLP)
        self.classifier_head_action = torch.nn.Sequential(
            torch.nn.Linear(num_features * self.bert_encoder.config.hidden_size, mlp_hidden_size),
            torch.nn.LayerNorm(mlp_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_size, num_actions)
        )
        if not self.unlabeled_arcs:
            # define arc-label classifier head (2 layer MLP)
            self.classifier_head_label = torch.nn.Sequential(
                torch.nn.Linear(num_features * self.bert_encoder.config.hidden_size, mlp_hidden_size),
                torch.nn.LayerNorm(mlp_hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(mlp_hidden_size, num_labels)
            )

        # make sure BERT parameters are trainable
        for param in self.bert_encoder.parameters():
            param.requires_grad = True

    def get_features(self, bert_output, state_idx, batch_idx):
        features_states = []
        for i in range(len(state_idx[batch_idx])):
            # get BERT embeddings for the top two words on the stack and the top word on the buffer
            # (the embedding of a word is being represented by the embedding of it's first subword token)
            stack1 = bert_output[batch_idx, state_idx[batch_idx][i][0], :] # shape: (hidden_size,)
            stack2 = bert_output[batch_idx, state_idx[batch_idx][i][1], :] # shape: (hidden_size,)
            buffer = bert_output[batch_idx, state_idx[batch_idx][i][2], :] # shape: (hidden_size,)
            # concatenate the embeddings
            features = torch.cat([stack1, stack2, buffer], dim=0) # shape: (3*hidden_size,)
            features_states.append(features) 
        # stack up the features for all states into a single tensor
        features = torch.stack(features_states) # shape: (num_states, 3*hidden_size)   
        return features

    def get_bert_encoding(self, input_idx, input_attn_mask):
        bert_output = self.bert_encoder(input_idx, attention_mask=input_attn_mask)
        bert_output = self.dropout(bert_output.last_hidden_state) # shape: (batch_size, block_size, hidden_size)
        return bert_output

    def forward(self, input_idx, input_attn_mask, state_idx, target_action_idx=None, target_label_idx=None):
        # compute BERT embeddings for input tokens
        bert_output = self.get_bert_encoding(input_idx, input_attn_mask)

        loss = 0.0
        batch_action_logits = []
        batch_label_logits = []
        # iterate over each sentence in the batch
        for batch_idx in range(len(input_idx)):  
            # get the features for all parse states
            features = self.get_features(bert_output, state_idx, batch_idx) # shape: (num_states, 3*hidden_size)
            # compute action logits and cross-entropy loss
            action_logits = self.classifier_head_action(features) # shape: (num_states, num_actions)
            batch_action_logits.append(action_logits)
            if target_action_idx is not None:
                action_targets = torch.tensor(target_action_idx[batch_idx], dtype=torch.long, device=input_idx.device)
                loss += F.cross_entropy(action_logits, action_targets)
            if not self.unlabeled_arcs:
                # compute arc-label logits and cross-entropy loss 
                label_logits = self.classifier_head_label(features) # shape: (num_states, num_labels)
                batch_label_logits.append(label_logits)
                if target_label_idx is not None:
                    label_targets = torch.tensor(target_label_idx[batch_idx], dtype=torch.long, device=input_idx.device)
                    loss += F.cross_entropy(label_logits, label_targets)

        # average loss over the batch
        loss = loss/len(input_idx)    

        return loss, batch_action_logits, batch_label_logits    

    @torch.no_grad()
    def predict(self, bert_output, state_idx):
        batch_action_logits = []
        batch_label_logits = []
        # iterate over each sentence in the batch
        for batch_idx in range(len(state_idx)):  
            # get the features for all parse states
            features = self.get_features(bert_output, state_idx, batch_idx) # shape: (num_states, 3*hidden_size)
            # compute action logits and cross-entropy loss
            action_logits = self.classifier_head_action(features) # shape: (num_states, num_actions)
            batch_action_logits.append(action_logits)
            if not self.unlabeled_arcs:
                # compute arc-label logits and cross-entropy loss 
                label_logits = self.classifier_head_label(features) # shape: (num_states, num_labels)
                batch_label_logits.append(label_logits)
   
        return batch_action_logits, batch_label_logits  


# training loop
def train(model, optimizer, train_dataloader, val_dataloader, scheduler=None, device="cpu", num_epochs=10, val_every=100, save_every=None, log_metrics=None):
    avg_loss = 0
    model.train()
    # reset gradients
    optimizer.zero_grad()
    for epoch in range(num_epochs):
        train_uas = 0
        train_las = 0
        val_loss = 0
        val_uas = 0
        val_las = 0
        num_instances = 0
        pbar = tqdm(train_dataloader, desc="Epochs")
        for step, batch in enumerate(pbar):
            input_idx, input_attn_mask, state_idx, target_action_idx, target_label_idx = batch
            # move tensors to device
            input_idx, input_attn_mask = input_idx.to(device), input_attn_mask.to(device)
            # forward pass
            loss, batch_action_logits, batch_label_logits = model(input_idx, input_attn_mask, state_idx, target_action_idx, target_label_idx)
            # reset gradients
            optimizer.zero_grad()
            # backward pass
            loss.backward()
            # optimizer step
            optimizer.step()

            if scheduler is not None:
                    scheduler.step()

            avg_loss = 0.9* avg_loss + 0.1*loss.item()

            # compute unlabeled and labeled attachment scores
            for batch_idx in range(len(input_idx)):
                action_logits = batch_action_logits[batch_idx]
                action_idx = target_action_idx[batch_idx]
                if not model.unlabeled_arcs:
                    label_logits = batch_label_logits[batch_idx]
                    label_idx = target_label_idx[batch_idx]
                # compute UAS and LAS
                sentence_uas = 0
                sentence_las = 0
                for i in range(len(action_idx)):
                    if action_idx[i] == torch.argmax(action_logits[i]):
                        sentence_uas += 1
                        if not model.unlabeled_arcs:
                            if label_idx[i] == torch.argmax(label_logits[i]):
                                sentence_las += 1                
                sentence_uas = sentence_uas/len(action_idx)
                train_uas += sentence_uas
                if not model.unlabeled_arcs:
                    sentence_las = sentence_las/len(action_idx)
                    train_las += sentence_las
                num_instances += 1    

            if val_every is not None:
                if (step+1)%val_every == 0:
                    # compute validation loss
                    val_loss, val_uas, val_las = validation(model, val_dataloader, device=device)
                    pbar.set_description(f"Epoch {epoch + 1}, EMA Train Loss: {avg_loss:.3f}, Train UAS, LAS: ({train_uas/num_instances: .3f}, {train_las/num_instances: .3f}), Val Loss: {val_loss: .3f}, Val UAS, LAS: ({val_uas: .3f}, {val_las: .3f})")  

            pbar.set_description(f"Epoch {epoch + 1}, EMA Train Loss: {avg_loss:.3f}, Train UAS, LAS: ({train_uas/num_instances: .3f}, {train_las/num_instances: .3f}), Val Loss: {val_loss: .3f}, Val UAS, LAS: ({val_uas: .3f}, {val_las: .3f})")

            if log_metrics:
                metrics = {"Batch loss":loss.item(), "Moving Avg Loss":avg_loss, "Train UAS":train_uas/num_instances, "Train LAS":train_las/num_instances,"Val Loss": val_loss, "Val UAS":val_uas, "Val LAS":val_las}   
                log_metrics(metrics)

        if save_every is not None:
            if (epoch+1) % save_every == 0:
                save_model_checkpoint(model, optimizer, epoch, avg_loss)


def validation(model, dataloader, device="cpu"):
    model.eval()
    val_losses = torch.zeros(len(dataloader))
    with torch.no_grad():
        val_uas = 0
        val_las = 0
        num_instances = 0
        for step,batch in enumerate(dataloader):
            input_idx, input_attn_mask, state_idx, target_action_idx, target_label_idx = batch
            input_idx, input_attn_mask = input_idx.to(device), input_attn_mask.to(device)
            loss, batch_action_logits, batch_label_logits = model(input_idx, input_attn_mask, state_idx, target_action_idx, target_label_idx)
            
            # compute unlabeled and labeled attachment scores
            for batch_idx in range(len(input_idx)):
                action_logits = batch_action_logits[batch_idx]
                action_idx = target_action_idx[batch_idx]
                if not model.unlabeled_arcs:
                    label_logits = batch_label_logits[batch_idx]
                    label_idx = target_label_idx[batch_idx]
                # compute UAS and LAS
                sentence_uas = 0
                sentence_las = 0
                for i in range(len(action_idx)):
                    if action_idx[i] == torch.argmax(action_logits[i]):
                        sentence_uas += 1
                        if not model.unlabeled_arcs:
                            if label_idx[i] == torch.argmax(label_logits[i]):
                                sentence_las += 1                
                sentence_uas = sentence_uas/len(action_idx)
                val_uas += sentence_uas
                if not model.unlabeled_arcs:
                    sentence_las = sentence_las/len(action_idx)
                    val_las += sentence_las
                num_instances += 1  

            val_losses[step] = loss.item()
    model.train()
    val_loss = val_losses.mean().item()
    val_uas = val_uas/num_instances
    val_las = val_las/num_instances
    return val_loss, val_uas, val_las


def save_model_checkpoint(model, optimizer, epoch=None, loss=None, filename='BERT_TRANSITION_PARSER_checkpoint.pth'):
    # Save the model and optimizer state_dict
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    # Save the checkpoint to a file
    torch.save(checkpoint, filename)
    print(f"Saved model checkpoint!")


def load_model_checkpoint(model, optimizer=None,  filename='BERT_TRANSITION_PARSER_checkpoint.pth'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded model from checkpoint!")
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.train()
        return model, optimizer          
    else:
        return model        