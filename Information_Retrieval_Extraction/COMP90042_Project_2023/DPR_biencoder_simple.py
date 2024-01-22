""" 
    DPR Bi-Encoder Model Implementation
"""

import torch
from transformers import DistilBertModel
import torch.nn.functional as F
from tqdm import tqdm

# uses in-batch negatives
class BERTBiEncoder(torch.nn.Module):
    def __init__(self, dropout_rate=0.1, out_of_batch_negs=False):
        super().__init__()
        # load pretrained BERT model
        self.query_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.passage_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.out_of_batch_negs = out_of_batch_negs

        for param in self.query_encoder.parameters():
            param.requires_grad = True
        for param in self.passage_encoder.parameters():
            param.requires_grad = True

    def forward(self, query_idx, query_attn_mask, pos_idx, pos_attn_mask, neg_idx, neg_attn_mask):
        # compute BERT encodings, extract the `[CLS]` encoding (first element of the sequence), apply dropout        
        query_output = self.query_encoder(query_idx, attention_mask=query_attn_mask)
        query_enc = self.dropout(query_output.last_hidden_state[:, 0]) # shape: (batch_size, hidden_size)
        pos_output = self.passage_encoder(pos_idx, attention_mask=pos_attn_mask)
        pos_enc = self.dropout(pos_output.last_hidden_state[:,0]) # shape: (batch_size, hidden_size)
        
        if not self.out_of_batch_negs:
            neg_output = self.passage_encoder(neg_idx, attention_mask=neg_attn_mask)
            neg_enc = self.dropout(neg_output.last_hidden_state[:,0]) # shape: (batch_size, hidden_size)
            # compute similarity score matrix for query and positives
            scores_QP = torch.mm(query_enc, pos_enc.transpose(0, 1)) # shape: (batch_size, batch_size)
            # compute similarity score matrix for query and negatives
            scores_QN = torch.mm(query_enc, neg_enc.transpose(0, 1)) # shape: (batch_size, batch_size)
            # concatenate the positive and negative scores
            scores = torch.cat([scores_QP, scores_QN], dim=1) # shape: (batch_size, 2*batch_size)
            # compute cross-entropy loss
            loss = F.cross_entropy(scores, torch.arange(scores.shape[0]).to(scores.device))
        else:
            neg_output = self.passage_encoder(neg_idx, attention_mask=neg_attn_mask)
            neg_enc = self.dropout(neg_output.last_hidden_state[:,0]) # shape: (batch_size*num_negatives, hidden_size)
            # reshape negative passages back to (batch_size, num_negatives, hidden_size)
            neg_enc = neg_enc.view(query_enc.shape[0], -1, neg_enc.shape[-1])
            # compute similarity scores between each query in batch with its positive and negative passages
            scores_QP = torch.bmm(query_enc.unsqueeze(1), pos_enc.unsqueeze(-1)).squeeze(-1) # shape: (batch_size, 1, hidden_size) x (batch_size, hidden_size, 1) = (batch_size, 1, 1) -> (batch_size, 1)
            scores_QN = torch.bmm(query_enc.unsqueeze(1), neg_enc.transpose(1, 2)).squeeze(1) # shape: (batch_size, 1, hidden_size) x (batch_size, hidden_size, num_negatives) = (batch_size, 1, num_negatives) -> (batch_size, num_negatives)
            # concatenate the positive and negative scores
            scores = torch.cat([scores_QP, scores_QN], dim=1) # shape: (batch_size, 1+num_negatives)
            # compute cross-entropy loss
            loss = F.cross_entropy(scores, torch.zeros(scores.shape[0]).to(scores.device).long())

        return scores, loss
    
    @ torch.no_grad()
    def encode_queries(self, query_idx, query_attn_mask):
        self.eval()
        query_output = self.query_encoder(query_idx, attention_mask=query_attn_mask)
        query_enc = self.dropout(query_output.last_hidden_state[:, 0])
        return query_enc

    @ torch.no_grad()
    def encode_passages(self, passage_idx, passage_attn_mask):
        self.eval()
        passage_output = self.passage_encoder(passage_idx, attention_mask=passage_attn_mask)
        passage_enc = self.dropout(passage_output.last_hidden_state[:,0]) # shape: (batch_size, hidden_size)
        return passage_enc



# training loop
def train(model, optimizer, train_dataloader, val_dataloader, scheduler=None, device="cpu", num_epochs=10, accumulation_steps=1, val_every=100, save_every=None, log_metrics=None):
    avg_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0
    model.train()
    # reset gradients
    optimizer.zero_grad()
    for epoch in range(num_epochs):
        num_correct = 0
        num_total = 0
        pbar = tqdm(train_dataloader, desc="Train Epochs")
        for i, batch in enumerate(pbar):
            query_idx, query_attn_mask, pos_idx, pos_attn_mask, neg_idx, neg_attn_mask = batch
            # move batch to device
            query_idx, query_attn_mask, pos_idx, pos_attn_mask, neg_idx, neg_attn_mask = query_idx.to(device), query_attn_mask.to(device), pos_idx.to(device), pos_attn_mask.to(device), neg_idx.to(device), neg_attn_mask.to(device)
            # forward pass
            scores, loss = model(query_idx, query_attn_mask, pos_idx, pos_attn_mask, neg_idx, neg_attn_mask)
            # backward pass
            loss.backward()
            # apply gradient step 
            if (i+1) % accumulation_steps == 0:
                # optimizer step
                optimizer.step()
                # reset gradients
                optimizer.zero_grad()
   
            avg_loss = 0.9* avg_loss + 0.1*loss.item()
            B, _ = query_idx.shape
            y_pred = scores.argmax(dim=-1).view(-1) # shape (B,)
            targets = torch.arange(B).to(device) # shape (B,)
            num_correct += (y_pred.eq(targets.view(-1))).sum().item()      
            num_total += B
            train_acc = num_correct / num_total        

            if val_every is not None:
                if (i+1)%val_every == 0:
                    # compute validation loss
                    val_loss, val_acc = validation(model, val_dataloader, device=device)
                    pbar.set_description(f"Epoch {epoch + 1}, EMA Train Loss: {avg_loss:.3f}, Train Accuracy: {train_acc: .3f}, Val Loss: {val_loss: .3f}, Val Accuracy: {val_acc: .3f}")  

            pbar.set_description(f"Epoch {epoch + 1}, EMA Train Loss: {avg_loss:.3f}, Train Accuracy: {train_acc: .3f}, Val Loss: {val_loss: .3f}, Val Accuracy: {val_acc: .3f}")  

            if log_metrics:
                metrics = {"Batch loss" : loss.item(), "Moving Avg Loss" : avg_loss, "Val Loss": val_loss}
                log_metrics(metrics)

        # run optimizer step for remainder batches
        if len(train_dataloader) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        if save_every is not None:
            if (epoch+1) % save_every == 0:
                save_dpr_model_checkpoint(model, optimizer, epoch, avg_loss)


def validation(model, val_dataloader, device="cpu"):
    model.eval()
    val_losses = torch.zeros(len(val_dataloader))
    with torch.no_grad():
        num_correct = 0
        num_total = 0
        pbar = tqdm(val_dataloader, desc="Val Epochs")
        for i,batch in enumerate(pbar):
            query_idx, query_attn_mask, pos_idx, pos_attn_mask, neg_idx, neg_attn_mask = batch
            query_idx, query_attn_mask, pos_idx, pos_attn_mask, neg_idx, neg_attn_mask = query_idx.to(device), query_attn_mask.to(device), pos_idx.to(device), pos_attn_mask.to(device), neg_idx.to(device), neg_attn_mask.to(device)
            scores, loss = model(query_idx, query_attn_mask, pos_idx, pos_attn_mask, neg_idx, neg_attn_mask)
            B, _ = query_idx.shape
            y_pred = scores.argmax(dim=-1).view(-1) # shape (B,)
            targets = torch.arange(B).to(device) # shape (B,)
            num_correct += (y_pred.eq(targets.view(-1))).sum().item()      
            num_total += B
            val_losses[i] = loss.item()
    model.train()
    val_loss = val_losses.mean().item()
    val_accuracy = num_correct / num_total
    return val_loss, val_accuracy

def save_dpr_model_checkpoint(model, optimizer, epoch=None, loss=None, filename=None):
    # Save the model and optimizer state_dict
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    # Save the checkpoint to a file
    if filename:
        torch.save(checkpoint, filename)
    else:
        torch.save(checkpoint, 'dpr_checkpoint.pth')
    print(f"Saved model checkpoint!")


def load_dpr_model_checkpoint(model, optimizer=None, filename=None):
    if filename:
        checkpoint = torch.load(filename)
    else:
        checkpoint = torch.load('dpr_checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded model from checkpoint!")
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.train()
        return model, optimizer          
    else:
        return model