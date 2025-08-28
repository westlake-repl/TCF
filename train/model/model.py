
import torch
from torch import nn
from .modules import UserEncoder, TextEncoderFinetuning, TextEncoderAllFreeze, MLPLayers
from torch.nn.init import xavier_normal_, constant_


class SASRec(torch.nn.Module):

    def __init__(self, args, pop_prob_list, item_num, use_modal, text_model):
        super(SASRec, self).__init__()
        self.args = args
        self.use_modal = use_modal
        self.max_seq_len = args.max_seq_len
        self.pop_prob_list = torch.FloatTensor(pop_prob_list)

        self.user_encoder = UserEncoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)

        if self.use_modal:
            if text_model is not None:
                self.text_encoder = TextEncoderFinetuning(
                    text_model,
                    args.n_project_out_mlp_layers,
                    args.embedding_dim,
                    args.word_embedding_dim)
            else:
                self.text_encoder = TextEncoderAllFreeze(
                    text_model,
                    args.n_project_out_mlp_layers,
                    args.embedding_dim,
                    args.word_embedding_dim)
        else:
            self.id_embedding = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
        
        if args.loss_type == 'BCE':
            self.criterion = nn.BCEWithLogitsLoss()
        elif args.loss_type == 'IBCE':
            self.criterion = nn.CrossEntropyLoss()

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, sample_items_id, sample_items, log_mask, local_rank):
        if isinstance(self.criterion,  nn.CrossEntropyLoss):
            if self.use_modal:
                item_embs = self.text_encoder(sample_items)
            else:
                item_embs = self.id_embedding(sample_items)
                
            user_embs = self.user_encoder(
                item_embs.view(-1, self.max_seq_len + 1, self.args.embedding_dim)[:, :-1, :], 
                log_mask, local_rank)
            user_embs = user_embs.view(-1, self.args.embedding_dim)  # (bs*max_seq_len, ed)
            
            # IN-BATCH CROSS-ENTROPY LOSS 
            self.pop_prob_list = self.pop_prob_list.to(local_rank)
            pop_bias = torch.log(self.pop_prob_list[sample_items_id])
            scores = torch.matmul(user_embs, item_embs.T) 
            debias_scores = scores - pop_bias
            
            bs, seq_len = log_mask.size(0), log_mask.size(1)
            label = torch.arange(bs * (seq_len + 1)).reshape(bs, seq_len + 1)
            label = label[:, 1:].to(local_rank).view(-1)
            
            flatten_item_seq = sample_items_id
            user_history = torch.zeros(bs, seq_len + 2).type_as(sample_items_id)
            user_history[:, :-1] = sample_items_id.view(bs, -1)
            user_history = user_history.unsqueeze(-1).expand(-1, -1, len(flatten_item_seq))
            history_item_mask = (user_history == flatten_item_seq).any(dim=1)
            history_item_mask = history_item_mask.repeat_interleave(seq_len, dim=0)
            unused_item_mask = history_item_mask.scatter(1, label.view(-1, 1), False)
            debias_scores[unused_item_mask] = -float('inf')
            
            indices = torch.where(log_mask.view(-1) != 0)
            debias_scores = debias_scores.view(bs * seq_len, -1)
            
            loss = self.criterion(debias_scores[indices], label[indices])
        
        elif isinstance(self.criterion, nn.BCEWithLogitsLoss):
            pos_sample_items, neg_sample_items = sample_items
            if self.use_modal:
                pos_item_embs = self.text_encoder(pos_sample_items)
                neg_item_embs = self.text_encoder(neg_sample_items)
            else:
                pos_item_embs = self.id_embedding(pos_sample_items)
                neg_item_embs = self.id_embedding(neg_sample_items)

            pos_item_embs = pos_item_embs.reshape(-1, self.max_seq_len + 1, self.args.embedding_dim)
            neg_item_embs = neg_item_embs.reshape(-1, self.max_seq_len + 1, self.args.embedding_dim)
            
            input_item_embs = pos_item_embs[:, :-1, :]
            tgt_pos_item_embs = pos_item_embs[:, 1:, :]
            tgt_neg_item_embs = neg_item_embs[:, 1:, :]
            
            pred_embs = self.user_encoder(input_item_embs, log_mask, local_rank)
            pos_scores = (pred_embs * tgt_pos_item_embs).sum(dim=-1)
            neg_scores = (pred_embs * tgt_neg_item_embs).sum(dim=-1)
            
            indices = torch.where(log_mask != 0)
            pos_scores = pos_scores[indices].reshape(-1)
            neg_scores = neg_scores[indices].reshape(-1)
            pos_labels = torch.ones_like(pos_scores)
            neg_labels = torch.zeros_like(neg_scores)
            scores = torch.cat([pos_scores, neg_scores], dim=0)
            labels = torch.cat([pos_labels, neg_labels], dim=0)
            loss = self.criterion(scores, labels)
        
        return loss

class DSSM(torch.nn.Module):
    
    def __init__(self, args, pop_prob_list, user_num, item_num, use_modal, text_model):
        super(DSSM, self).__init__()
        self.args = args
        self.use_modal = use_modal
        self.pop_prob_list = torch.FloatTensor(pop_prob_list)
        self.n_mlp_layers = args.n_mlp_layers
        self.embedding_dim = args.embedding_dim
        self.text_model = text_model
        self.activation = nn.GELU()
        
        self.user_embedding = nn.Embedding(user_num + 1, self.embedding_dim, padding_idx=0)
        xavier_normal_(self.user_embedding.weight.data)
        self.user_encoder = MLPLayers(
            self.embedding_dim,
            self.embedding_dim,
            args.n_mlp_layers,
            self.activation)
        
        if self.use_modal:
            n_project_out_mlp_layers = args.n_project_out_mlp_layers if args.n_mlp_layers == 0 else 1
            if text_model is not None:
                self.text_encoder = TextEncoderFinetuning(
                    text_model,
                    n_project_out_mlp_layers,
                    args.embedding_dim,
                    args.word_embedding_dim)
            else:
                self.text_encoder = TextEncoderAllFreeze(
                    text_model,
                    n_project_out_mlp_layers,
                    args.embedding_dim,
                    args.word_embedding_dim)
        else:
            self.id_embedding = nn.Embedding(item_num + 1, self.embedding_dim, padding_idx=0)
        
        self.item_encoder = MLPLayers(
            self.embedding_dim, 
            self.embedding_dim,
            args.n_mlp_layers,
            self.activation)
            
        if args.loss_type == 'BCE':
            self.criterion = nn.BCEWithLogitsLoss()
        elif args.loss_type == 'IBCE':
            self.criterion = nn.CrossEntropyLoss()
            
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, user_ids, user_history, sample_items, local_rank):
        if isinstance(self.criterion, nn.CrossEntropyLoss):
            user_embedding = self.user_embedding(user_ids)
            user_feature = self.user_encoder(user_embedding)
            if self.use_modal: 
                if self.text_model is not None:
                    item_ids, attention_mask, inputs_hidden_state = sample_items
                    sample_items = (attention_mask, inputs_hidden_state)
                    item_embedding = self.text_encoder(sample_items)
                else:
                    item_ids, sample_items = sample_items
                    item_embedding = self.text_encoder(sample_items)
            else:
                item_ids = sample_items
                item_embedding = self.id_embedding(sample_items)    
            
            item_feature = self.item_encoder(item_embedding)
            
            # IN-BATCH CROSS-ENTROPY LOSS 
            bs = user_ids.size(0)
            self.pop_prob_list = self.pop_prob_list.to(local_rank)
            debias_logits = torch.log(self.pop_prob_list[item_ids])
            label = torch.arange(bs, dtype=torch.long).to(local_rank)
            logits = torch.matmul(user_feature, item_feature.T)  
            logits = logits - debias_logits
            user_history = user_history.unsqueeze(-1).expand(-1, -1, bs)
            mask = (user_history == item_ids).sum(1).bool()
            logits[mask] = -float('inf')
            loss = self.criterion(logits, label)
            
        elif isinstance(self.criterion, nn.BCEWithLogitsLoss):
            pos_sample_items, neg_sample_items = sample_items
            if self.use_modal: 
                pos_item_embedding = self.text_encoder(pos_sample_items)
                neg_item_embedding = self.text_encoder(neg_sample_items)
            else:
                pos_item_embedding = self.id_embedding(pos_sample_items) 
                neg_item_embedding = self.id_embedding(neg_sample_items) 
                
            pos_item_feature = self.item_encoder(pos_item_embedding)
            neg_item_feature = self.item_encoder(neg_item_embedding)
            
            user_embedding = self.user_embedding(user_ids)
            user_feature = self.user_encoder(user_embedding)
                
            pos_scores = (pos_item_feature * user_feature).sum(-1)
            neg_scores = (neg_item_feature * user_feature).sum(-1)
            scores = torch.cat([pos_scores, neg_scores], dim=0)
            labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)
            loss = self.criterion(scores, labels)
                        
        return loss       
            
            
        