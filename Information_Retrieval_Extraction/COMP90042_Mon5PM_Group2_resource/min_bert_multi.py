"""
    A minimal BERT Model Implementation

    Author: Tanzid Sultan 
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
import math
torch.manual_seed(1234)

# use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu' 


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, total_head_size, num_heads, dropout_rate):
        super().__init__()

        assert total_head_size % num_heads == 0, "head_size needs to be integer multiple of num_heads"

        self.embedding_dim = embedding_dim
        self.total_head_size = total_head_size 
        self.head_size = total_head_size // num_heads 
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # define parameters
        self.key = nn.Linear(embedding_dim, self.total_head_size, bias=False)
        self.query = nn.Linear(embedding_dim, self.total_head_size, bias=False)
        self.value = nn.Linear(embedding_dim, self.total_head_size, bias=False)
        self.attn_dropout = nn.Dropout(dropout_rate)

        # we also need to apply a linear projection to make the output residual the same dimension as the input
        self.proj = nn.Linear(total_head_size, embedding_dim) 
        self.output_dropout = nn.Dropout(dropout_rate)


    # define forward pass, input shape: (B,T,C) where B=batch size, T=block_size, C=embedding_dim
    # the attn_mask is a mask that can be used for masking out the attention weights for padding tokens 
    def forward(self, x, attn_mask):
        B, T, C = x.shape
        #print(f"B = {B}, T={T}, C={C}")
        k = self.key(x) # (B,T,H) where H is the total_head_size
        q = self.query(x) # (B,T,H)
        v = self.value(x) # (B,T,H)

        # reshape (B,T,H) --> (B,T,n,h), where n=num_heads and h=head_size and H=n*h
        k = k.view(B,T,self.num_heads,self.head_size) 
        q = q.view(B,T,self.num_heads,self.head_size) 
        v = v.view(B,T,self.num_heads,self.head_size) 

        # now we transpose so that the num_heads is the second dimension followed by T,h
        # this allows us to batch matrix mutliply for all heads simulataneously to compute their attention weights
        # (B,T,n,h) --> (B,n,T,h) 
        k = k.transpose(1,2) 
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        """
        # compute attention scores manually (slower)
        W = q @ k.transpose(-2,-1)  / math.sqrt(self.head_size) # (B,n,T,T)
        attn_mask = attn_mask.view(B,1,1,T)        
        #print(f"W shape= {W.shape}, attn_mask shape = {attn_mask.shape}")
        W = W.masked_fill(attn_mask == 0, float('-inf')) 
        W = F.softmax(W, dim=-1)
        # apply dropout to attention weights
        W = self.attn_dropout(W)
        out = W @ v # (B,n,T,h)
        """

        # reshape attn_mask from (B, T) to (B, 1, 1, T)
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        # use pytorch built-in function for faster computation of attention scores (set the 'is_causal' parameter for applying causal masking)
        out = F.scaled_dot_product_attention(q,k,v,attn_mask=attn_mask.bool(),dropout_p=self.dropout_rate if self.training else 0,is_causal=False)

        # we can transpose the output from (B,n,T,h) --> (B,T,n,h)
        # since the last two dimensions of the transposed tensor are non-contiguous, we apply 
        # contiguous() which return a contiguous tensor
        out = out.transpose(1,2).contiguous()

        # finally we collapse the last two dimensions to get the concatenated output, (B,T,n,h) --> (B,T,n*h) 
        out = out.view(B,T,self.total_head_size)

        # now we project the concatenated output so that it has the same dimensions as the multihead attention layer input
        # (we need to add it with the input because of the residual connection, so need to be same size) 
        out = self.proj(out) # (B,T,C) 

        # apply dropout
        out = self.output_dropout(out)

        return out
    

# a simple mlp 
class FeedForward(nn.Module):
    def __init__(self, embedding_dim, dropout_rate):
        super().__init__()
        # we add extra computations by growing out the feed-forward hidden size by a factor of 4
        # we also add an extra linear layer at the end to project the residual back to same dimensions as input
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4*embedding_dim),  
            nn.GELU(),
            nn.Linear(4*embedding_dim, embedding_dim), 
            nn.Dropout(dropout_rate)
        )
    
    # in the forward pass, concatenate the outputs from all the attention heads
    def forward(self, x):
        return self.net(x)
    

# transformer encoder block with residual connection and layer norm
# Note: the original transformer uses post layer norms, here we use pre layer norms, i.e. layer norm is applied at the input
# instead of the output, this typically leads to better results in terms of training convergence speed and gradient scaling 
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, head_size, num_heads, dropout_rate):
        super().__init__()
        self.sa = MultiHeadAttention(embedding_dim, head_size, num_heads, dropout_rate) # multi-head attention layer 
        self.ff = FeedForward(embedding_dim, dropout_rate)   # feed-forward layer
        self.ln1 = nn.LayerNorm(embedding_dim) # layer norm at input of multi-head attention
        self.ln2 = nn.LayerNorm(embedding_dim) # layer norm at input of feed-forward

    # in the forward pass, concatenate the outputs from all the attention heads
    def forward(self, x, attn_mask):
        # residual connection between input and multi-head attention output (also note that we're doing a pre-layer norm, i.e. layer norm at the input of the multi-head attention)
        x = x + self.sa(self.ln1(x), attn_mask)
        # residual connection between multi-head attention output and feed-forward output (also note that we're doing a pre-layer norm, i.e. layer norm at the input of the feed-forward)
        x = x + self.ff(self.ln2(x)) 
        return x
    

# BERT model with multiple transformer blocks 
class BERTModel(nn.Module):
    def __init__(self, vocab_size, block_size, embedding_dim, head_size, num_heads, num_layers, pad_token_id, sent_entailment=True, claim_classification=True, dropout_rate=0.2, device='cpu'):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size        # block_size is just the input sequence length
        self.embedding_dim = embedding_dim
        self.head_size = head_size
        self.hum_heads = num_heads
        self.num_layers = num_layers
        self.sent_entailment = sent_entailment
        self.claim_classification = claim_classification

        '''
        Define model parameters
        '''
        # token embedding layer 
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id) # shape: (vocab_size,C)
        # position embedding layer
        self.pos_embedding = nn.Embedding(block_size, embedding_dim) # shape: (T,C)
        # segment embedding layer (disabled for now)
        self.segment_embedding = nn.Embedding(2, embedding_dim)

        # stack of transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(embedding_dim, head_size, num_heads, dropout_rate) for _ in range(num_layers)])

        # pooling transformation of CLS token (for downstream tasks requiring full sentence hidden representation)
        #self.pooling_linear = nn.Linear(embedding_dim, embedding_dim) # shape: (C,C)
        #self.pooling_activation_fn = nn.Tanh()

        # MLM output layer
        self.ln = nn.LayerNorm(embedding_dim)
        self.output_MLM = nn.Linear(embedding_dim, vocab_size)
        self.dropout = torch.nn.Dropout(dropout_rate)

        if sent_entailment:
            # add a linear layer for sentence entailment task
            self.entailment_linear = nn.Linear(embedding_dim, 2)

        if claim_classification:
            # add a linear layer for claim classification task
            self.claim_linear = nn.Linear(embedding_dim, 4)    

        # store position indices inside a buffer for fast access when computing position embeddings
        position_idx = torch.arange(block_size, device=device).unsqueeze(0)
        self.register_buffer('position_idx', position_idx)


        # forward pass takes in a batch of input token sequences idx of shape (B,T) and corresponding targets of shape (B,T)
    def forward(self, idx, attn_mask, segment_idx=None, return_cls=False):
        B, T = idx.shape
        # get token embeddings
        token_embeds = self.token_embedding(idx) # (B,T,C)
        # add positional encoding
        pos_embeds = self.pos_embedding(self.position_idx[:,:T]) # (T,C) 
        x = token_embeds + pos_embeds # (B,T,C)
        
        # add sentence segment embedding (disabled for now)
        if segment_idx is not None:
            # segment_idx is an integer tensor of shape (B,T) and has 0's corresponding to first sentence tokens and 1's for second sentence 
            segment_embeds = self.segment_embedding(segment_idx.long()) 
            x = x + segment_embeds # (B,T,C)

        # pass through transformer blocks to get encoding
        for block in self.blocks:
            x = block(x, attn_mask) # (B,T,C)
    
        # apply final layers norm (maybe not necessary?)
        x = self.dropout(self.ln(x))
        cls_encoding = x[:,0] # (B,C)

        # get CLS token encoding and apply pooling transform
        if return_cls:
            #pooled_cls_encoding = self.pooling_activation_fn(self.pooling_linear(cls_encoding)) # (B,C)
            return cls_encoding

        # compute output logits
        MLM_logits = self.output_MLM(x) # (B,T,V)

        if self.sent_entailment:
            entailment_logits = self.entailment_linear(cls_encoding)        
            if self.claim_classification:
                claim_class_logits = self.claim_linear(cls_encoding)
                return MLM_logits, entailment_logits, claim_class_logits
            else:
                return MLM_logits, entailment_logits

        return MLM_logits 


def train(model, num_epochs, train_dataloader, val_dataloader, optimizer, grad_accumulation_steps=1, scheduler=None, val_every=None, save_every=1, device='cpu', log_metrics=None, mixed_precision=False, checkpoint_name='BERT_checkpoint', include_claim_loss=True, claim_loss_weight = 0.5):
    
    if mixed_precision:
        # initialize a gradient scaler
        scaler = GradScaler()

    smoothed_loss, val_loss, val_accuracy_MLM, val_accuracy_entailment, val_accuracy_claim = 0, 0, 0, 0, 0
    val_entailment_precision, val_entailment_recall = 0, 0
    smoothed_loss_MLM, smoothed_loss_entailment, smoothed_loss_claim = 0, 0, 0
    model.train()
    optimizer.zero_grad()
    num_correct_MLM = 0
    num_total_MLM = 0
    num_correct_entailment = 0
    num_total_entailment = 0
    num_correct_claim = 0
    num_total_claim = 0
    entailment_num_pos = 1
    entailment_num_true_pos = 1
    entailment_num_pred_pos = 1
    for epoch in range(num_epochs):
        pbar = tqdm(train_dataloader, desc="Epochs")
        for step, batch in enumerate(pbar):
            xb, yb_MLM, yb_entailment, yb_claim, attn_mask, segment_ids = batch['masked_input'], batch['MLM_label'], batch['entailment_label'], batch['claim_label'], batch['attention_mask'] , batch['segment_ids']
            # move tensors to device
            xb, yb_MLM, yb_entailment, yb_claim, attn_mask, segment_ids = xb.to(device), yb_MLM.to(device), yb_entailment.to(device), yb_claim.to(device), attn_mask.to(device), segment_ids.to(device)
            B,T = xb.shape
            yb_MLM = yb_MLM.view(B*T) # reshape labels to (B*T)

            # forward pass
            if mixed_precision:
                with autocast():
                    MLM_logits, entailment_logits, claim_class_logits = model(xb, attn_mask, segment_ids)
                    MLM_logits = MLM_logits.view(B*T,-1) # flatten sequences across batch
                    # compute cross entropy loss for MLM task
                    loss_MLM = F.cross_entropy(MLM_logits, yb_MLM, ignore_index=-100)
                    # compute cross entropy loss for sentence entailment task
                    loss_entailment = F.cross_entropy(entailment_logits, yb_entailment)
                    # compute cross entropy loss for claim classification task
                    loss_claim = F.cross_entropy(claim_class_logits, yb_claim)
                    if include_claim_loss:                       
                        total_loss = loss_MLM + (1-claim_loss_weight)*loss_entailment + claim_loss_weight*loss_claim
                    else:
                        total_loss = loss_MLM + loss_entailment

                    # backward pass with scaled gradients
                    scaler.scale(total_loss).backward()
                    if (step+1) % grad_accumulation_steps == 0:
                        # optimizer step with unscaled gradients
                        scaler.step(optimizer)
                        # update scaler for next iteration
                        scaler.update()
                        optimizer.zero_grad()

            else:       
                MLM_logits, entailment_logits, claim_class_logits = model(xb, attn_mask, segment_ids)
                MLM_logits = MLM_logits.view(B*T,-1) # flatten sequences across batch
                # compute cross entropy loss for MLM task
                loss_MLM = F.cross_entropy(MLM_logits, yb_MLM, ignore_index=-100)
                # compute cross entropy loss for sentence entailment task
                loss_entailment = F.cross_entropy(entailment_logits, yb_entailment)
                # compute cross entropy loss for claim classification task
                loss_claim = F.cross_entropy(claim_class_logits, yb_claim)
                if include_claim_loss:
                    total_loss = loss_MLM + (1-claim_loss_weight)*loss_entailment + claim_loss_weight*loss_claim
                else:
                    total_loss = loss_MLM + loss_entailment

                # backward pass
                total_loss.backward()
                if (step+1) % grad_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            # compute accuracy for masked token predictions
            mask = (yb_MLM != -100)
            y_pred_MLM = MLM_logits.argmax(dim=-1)
            num_correct_MLM += torch.eq(y_pred_MLM[mask], yb_MLM[mask]).sum().item()
            num_total_MLM += mask.sum().item()
            train_accuracy_MLM = num_correct_MLM/num_total_MLM

            # compute accuracy for sentence entailment task
            entailment_pred = entailment_logits.argmax(dim=-1)
            num_correct_entailment += torch.eq(entailment_pred, yb_entailment).sum().item()
            num_total_entailment += B
            train_accuracy_entailment = num_correct_entailment/num_total_entailment

            # compute precision and recall for sentence entailment
            entailment_num_pos += (yb_entailment == 1).sum().item()
            entailment_num_true_pos += ((entailment_pred == 1) & (yb_entailment == 1)).sum().item()
            entailment_num_pred_pos += (entailment_pred == 1).sum().item()
            train_entailment_precision = entailment_num_true_pos / entailment_num_pred_pos
            train_entailment_recall = entailment_num_true_pos / entailment_num_pos

            # compute accuracy for claim classification task
            claim_pred = claim_class_logits.argmax(dim=-1)
            num_correct_claim += torch.eq(claim_pred, yb_claim).sum().item()
            num_total_claim += B
            train_accuracy_claim = num_correct_claim/num_total_claim
            
            if val_every is not None:
                if (step+1)%val_every == 0:
                    val_loss, val_accuracy_MLM, val_accuracy_entailment, val_accuracy_claim, val_entailment_precision, val_entailment_recall = validation(model, val_dataloader, device)

            # compute exponential moving average loss and training accuracy
            smoothed_loss = 0.9 * smoothed_loss + 0.1 * total_loss.item()
            smoothed_loss_MLM = 0.9 * smoothed_loss_MLM + 0.1 * loss_MLM.item()
            smoothed_loss_entailment = 0.9 * smoothed_loss_entailment + 0.1 * loss_entailment.item()
            smoothed_loss_claim = 0.9 * smoothed_loss_claim + 0.1 * loss_claim.item()
            pbar.set_description(f"Epoch {epoch + 1}, Train Loss(Total, MLM, Entailment, Claim): ({smoothed_loss:.3f}, {smoothed_loss_MLM:.3f}, {smoothed_loss_entailment:.3f}, {smoothed_loss_claim:.3f}), Train Accuracy (MLM, Entailment, Claim): ({train_accuracy_MLM:.3f}, {train_accuracy_entailment:.3f}, {train_accuracy_claim:.3f}), Val Loss: {val_loss:.3f}, Val Accuracy (MLM, Entailment, Claim): ({val_accuracy_MLM:.3f}, {val_accuracy_entailment:.3f}, {val_accuracy_claim:.3f})")   

            if log_metrics:
                metrics = {"Batch Total Loss":total_loss.item(), "Train Total Loss":smoothed_loss, "Train MLM Loss": smoothed_loss_MLM, "Train Entailment Loss": smoothed_loss_entailment, "Train Claim Loss": smoothed_loss_claim, "Train Accuracy MLM": train_accuracy_MLM, "Train Accuracy Entailment": train_accuracy_entailment, "Train Accuracy Claim": train_accuracy_claim, "Train Entailment Precision":train_entailment_precision, "Train Entailment Recall": train_entailment_recall, "Val Loss": val_loss, "Val Accuracy MLM":val_accuracy_MLM, "Val Accuracy Entailment":val_accuracy_entailment, "Val Accuracy Claim":val_accuracy_claim, "Val Entailment Precision":val_entailment_precision, "Val Entailment Recall": val_entailment_recall}
                log_metrics(metrics)

        # lr scheduler step       
        if scheduler is not None:
            scheduler.step()

        # save checkpoint 
        if save_every is not None: 
            if (epoch+1)%save_every == 0:
                save_bert_model_checkpoint(model, optimizer, name=checkpoint_name, loss=smoothed_loss, epoch=epoch+1)
            
    print(f"Training done!")


# compute loss and accuracy on validation set
def validation(model, val_dataloader, device="cpu"):
    model.eval()
    val_losses = torch.zeros(len(val_dataloader))
    with torch.no_grad():
        num_correct_MLM = 0
        num_total_MLM = 0
        num_correct_entailment = 0
        num_total_entailment = 0
        num_correct_claim = 0
        num_total_claim = 0
        entailment_num_pos = 1
        entailment_num_true_pos = 1
        entailment_num_pred_pos = 1
        for step, batch in enumerate(val_dataloader):
            xb, yb_MLM, yb_entailment, yb_claim, attn_mask, segment_ids = batch['masked_input'], batch['MLM_label'], batch['entailment_label'], batch['claim_label'], batch['attention_mask'] , batch['segment_ids']
            # move tensors to device
            xb, yb_MLM, yb_entailment, yb_claim, attn_mask, segment_ids = xb.to(device), yb_MLM.to(device), yb_entailment.to(device), yb_claim.to(device), attn_mask.to(device), segment_ids.to(device)
            B,T = xb.shape
            yb_MLM = yb_MLM.view(B*T) # reshape labels to (B*T)
           
            MLM_logits, entailment_logits, claim_class_logits = model(xb, attn_mask, segment_ids)
            MLM_logits = MLM_logits.view(B*T,-1) # flatten sequences across batch
            # compute cross entropy loss for MLM task
            loss_MLM = F.cross_entropy(MLM_logits, yb_MLM, ignore_index=-100)
            # compute cross entropy loss for sentence entailment task
            loss_entailment = F.cross_entropy(entailment_logits, yb_entailment)
            # compute cross entropy loss for claim classification task
            loss_claim = F.cross_entropy(claim_class_logits, yb_claim)
            total_loss = loss_MLM + loss_entailment + loss_claim
           
            val_losses[step] = total_loss.item()
            
            # compute accuracy for masked token predictions
            mask = (yb_MLM != -100)
            y_pred_MLM = MLM_logits.argmax(dim=-1)
            num_correct_MLM += torch.eq(y_pred_MLM[mask], yb_MLM[mask]).sum().item()
            num_total_MLM += mask.sum().item()

            # compute accuracy for sentence entailment task
            entailment_pred = entailment_logits.argmax(dim=-1)
            num_correct_entailment += torch.eq(entailment_pred, yb_entailment).sum().item()
            num_total_entailment += B

            # compute precision and recall for sentence entailment
            entailment_num_pos += (yb_entailment == 1).sum().item()
            entailment_num_true_pos += ((entailment_pred == 1) & (yb_entailment == 1)).sum().item()
            entailment_num_pred_pos += (entailment_pred == 1).sum().item()

            # compute accuracy for claim classification task
            claim_pred = claim_class_logits.argmax(dim=-1)
            num_correct_claim += torch.eq(claim_pred, yb_claim).sum().item()
            num_total_claim += B

    model.train()
    val_loss = val_losses.mean().item()
    val_accuracy_MLM = num_correct_MLM/num_total_MLM
    val_accuracy_entailment = num_correct_entailment/num_total_entailment
    val_accuracy_claim = num_correct_claim/num_total_claim
    val_entailment_precision = entailment_num_true_pos / entailment_num_pred_pos
    val_entailment_recall = entailment_num_true_pos / entailment_num_pos
    
    return val_loss, val_accuracy_MLM, val_accuracy_entailment, val_accuracy_claim, val_entailment_precision, val_entailment_recall


def save_bert_model_checkpoint(model, optimizer, name='BERT_checkpoint', epoch=None, loss=None):
    # Save the model and optimizer state_dict
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    # Save the checkpoint to a file
    torch.save(checkpoint, name+'.pth')
    print(f"Saved BERT model checkpoint!")


def load_bert_model_checkpoint(model, optimizer=None, name='BERT_checkpoint', device='cpu', strict=True):
    checkpoint = torch.load(name+'.pth')
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Loaded pretrained BERT model checkpoint at epoch {epoch} with loss {loss}")    
    if optimizer is not None:
        return model, optimizer, epoch, loss
    return model, epoch, loss