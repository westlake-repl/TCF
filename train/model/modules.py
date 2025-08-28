import torch.nn as nn
import torch
from transformers import BertConfig, OPTConfig, T5Config

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_inner)
        self.w_2 = nn.Linear(d_inner, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.activate = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.dropout(self.w_2(self.activate(self.w_1(x))))
        return self.layer_norm(residual + x)


class SelfAttention(nn.Module):
    def __init__(self, temperature, dropout):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask):
        attn = torch.matmul(query, key.transpose(-2, -1)) / self.temperature
        attn = attn + mask
        p_attn = self.dropout(self.softmax(attn))
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.d_v = self.d_k

        self.w_Q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.w_K = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.w_V = nn.Linear(d_model, n_heads * self.d_v, bias=False)
        self.fc = nn.Linear(n_heads * self.d_v, d_model, bias=False)

        self.self_attention = SelfAttention(temperature=self.d_k ** 0.5, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, query, key, value, mask):
        sz_b, len_q, len_k, len_v = query.size(0), query.size(1), key.size(1), value.size(1)
        residual = query

        q = self.w_Q(query).view(sz_b, len_q, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_K(key).view(sz_b, len_k, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_V(value).view(sz_b, len_v, self.n_heads, self.d_v).transpose(1, 2)

        x, attn = self.self_attention(q, k, v, mask=mask)
        x = x.transpose(1, 2).contiguous().view(sz_b, len_q, self.d_model)
        x = self.dropout(self.fc(x))
        return self.layer_norm(residual + x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_inner, dropout):

        super().__init__()
        self.multi_head_attention = MultiHeadedAttention(n_heads=n_heads, d_model=d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_inner=d_inner, dropout=dropout)

    def forward(self, block_input, mask):
        output = self.multi_head_attention(block_input, block_input, block_input, mask)
        return self.feed_forward(output)


class TransformerEncoder(torch.nn.Module):
    def __init__(self, n_vocab, n_position, d_model, n_heads, dropout, n_layers):
        super(TransformerEncoder, self).__init__()
        # self.word_embedding = nn.Embedding(n_vocab + 1, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(n_position, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model=d_model, n_heads=n_heads, d_inner=d_model * 4, dropout=dropout
                              ) for _ in range(n_layers)])

    def forward(self, input_embs, log_mask, att_mask):
        position_ids = torch.arange(log_mask.size(1), dtype=torch.long, device=log_mask.device)
        position_ids = position_ids.unsqueeze(0).expand_as(log_mask)
        output = self.layer_norm(input_embs + self.position_embedding(position_ids))
        output = self.dropout(output)
        for transformer in self.transformer_blocks:
            output = transformer.forward(output, att_mask)
        return output


class MLPLayers(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_layers,
        activation):
        super(MLPLayers, self).__init__()
        self.num_layers = num_layers
        if self.num_layers > 1:
            self.layers = torch.nn.ModuleList()
            self.layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(input_dim, output_dim),
                    activation))
            for _ in range(1, self.num_layers):
                self.layers.append(torch.nn.Sequential(
                            nn.Dropout(0.1),
                            torch.nn.Linear(output_dim, output_dim),
                            activation))
        elif self.num_layers == 1:
            self.layers = torch.nn.ModuleList(
                [torch.nn.Sequential(
                 torch.nn.Linear(input_dim, output_dim),
                 activation)])
        else:
            self.layers = None

    def forward(self, inputs):
        if self.layers is None:
            return inputs
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


class UserEncoder(torch.nn.Module):
    def __init__(self, item_num, max_seq_len, item_dim, num_attention_heads, dropout, n_layers):
        super(UserEncoder, self).__init__()
        self.transformer_encoder = TransformerEncoder(n_vocab=item_num, n_position=max_seq_len,
                                                      d_model=item_dim, n_heads=num_attention_heads,
                                                      dropout=dropout, n_layers=n_layers)

    def forward(self, input_embs, log_mask, local_rank):
        att_mask = (log_mask != 0)
        att_mask = att_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        att_mask = torch.tril(att_mask.expand((-1, -1, log_mask.size(-1), -1))).to(local_rank)
        att_mask = torch.where(att_mask, 0., -1e9)
        return self.transformer_encoder(input_embs, log_mask, att_mask)


class TextEncoderFinetuning(torch.nn.Module):
    def __init__(self,
                 text_model,
                 n_project_mlp_layers,
                 item_embedding_dim,
                 word_embedding_dim):
        super(TextEncoderFinetuning, self).__init__()
        self.text_model = text_model
        self.activate = nn.GELU()
        
        self.mlp = MLPLayers(
            word_embedding_dim,
            item_embedding_dim,
            n_project_mlp_layers,
            self.activate)

    def forward(self, text):
        attention_mask, inputs_hidden_state = text
        
        if isinstance(self.text_model.config, OPTConfig):
            if not self.text_model.keep_embed_layer:
                last_hidden_state = self.text_model(attention_mask=attention_mask, inputs_hidden_state=inputs_hidden_state)[0]
            else:
                input_ids = inputs_hidden_state.view(attention_mask.shape)
                last_hidden_state = self.text_model(input_ids=input_ids, attention_mask=attention_mask)[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            mean_output = torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            output = mean_output
        elif isinstance(self.text_model.config, BertConfig):
            if not self.text_model.keep_embed_layer:
                last_hidden_state = self.text_model(attention_mask=attention_mask, inputs_hidden_state=inputs_hidden_state)[0]
            else:
                input_ids = inputs_hidden_state.view(attention_mask.shape)
                last_hidden_state = self.text_model(input_ids=input_ids, attention_mask=attention_mask)[0]
            
            if self.text_model.config.mlm_mean_pooling:
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                mean_output = torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                output = mean_output
            else:
                output = last_hidden_state[:, 0, :]
        elif isinstance(self.text_model.config, T5Config):
            if not self.text_model.keep_embed_layer:
                last_hidden_state = self.text_model(attention_mask=attention_mask, inputs_hidden_state=inputs_hidden_state)[0]
            else:
                input_ids = inputs_hidden_state.view(attention_mask.shape)
                last_hidden_state = self.text_model(input_ids=input_ids, attention_mask=attention_mask)[0]
            
            if self.text_model.config.mlm_mean_pooling:
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                mean_output = torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                output = mean_output
            else:
                output = last_hidden_state[:, 0, :]
        else:
            raise NotImplementedError
        
        output = self.mlp(output)
        return output


class TextEncoderAllFreeze(nn.Module):
    def __init__(self,
                 text_model,
                 n_project_mlp_layers,
                 item_embedding_dim,
                 word_embedding_dim):
        super(TextEncoderAllFreeze, self).__init__()
        self.text_model = text_model # text_model is None here
        self.activate = nn.GELU()
        
        self.mlp = MLPLayers(
            word_embedding_dim,
            item_embedding_dim,
            n_project_mlp_layers,
            self.activate)


    def forward(self, text):
        output = self.mlp(text)
        return output
