
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, BaseModelOutputWithPast
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import  (
    BertModel,
    BertEmbeddings,
    BertLayer,
    BertPooler,
    BaseModelOutputWithPastAndCrossAttentions,
    BERT_INPUTS_DOCSTRING,
    BERT_START_DOCSTRING,
    _CHECKPOINT_FOR_DOC,
    _CONFIG_FOR_DOC,
    _TOKENIZER_FOR_DOC
)

logger = logging.get_logger(__name__)

    
def check_keep_encoders_range(
    config: BertConfig,
    keep_embed_layer: bool,
    keep_encoders_range: Optional[Union[int, Tuple[int, Optional[int]]]]):
    """
    Validate and adjust the provided range of encoder layers to keep during optimization.

    This function takes in the model configuration, a boolean indicating whether to keep the
    embedding layer, and a range of encoder layers to keep. It checks if the provided range is
    valid and adjusts it if necessary. If no range is provided, the function returns None.

    Args:
        config (BertConfig): The model configuration object.
        keep_embed_layer (bool): Whether to keep the embedding layer during optimization.
        keep_encoders_range (Optional[Union[int, Tuple[int, Optional[int]]]]): The range of decoder
            layers to keep during optimization. The range is a left-closed, right-open interval [a, b).

    Raises:
        ValueError: If keep_encoders_range is not specified and keep_embed_layer is False, or if the
            provided range is invalid (e.g., range_end <= range_start).

    Returns:
        Optional[Tuple[int, int]]: A tuple (range_start, range_end), representing the adjusted range
            of encoder layers to keep, or None if keep_encoders_range is None.

    Example:
        config = BertConfig("bert-base-uncased")
        adjusted_range = check_keep_encoders_range(config, True, (2, 6))
    """
    if keep_encoders_range is None:
        if not keep_embed_layer:
            raise ValueError("keep_encoders_range must be specified if keep_embed_layer is False")
        return None
    elif isinstance(keep_encoders_range, int):
        range_start = keep_encoders_range
        range_end = range_start + 1
    else:
        assert len(keep_encoders_range) == 2
        range_start, range_end = keep_encoders_range

    assert isinstance(range_start, int)

    if range_start != 0 and keep_embed_layer:
        first_layer = - config.num_hidden_layers if range_start < 0 else 0
        if range_start != 0 and range_start != first_layer:
            raise ValueError(
                "You can't keep the embedding layer since you drop the first encoder layer"
            )

    if range_start < 0:
        range_start = config.num_hidden_layers + range_start

    if range_end is None:
        range_end = config.num_hidden_layers
    elif range_end <= 0:
        range_end = config.num_hidden_layers + range_end

    if range_end <= range_start:
        raise ValueError("range_end should be greater than range_start")

    assert range_start >= 0 
    assert range_end > range_start
    assert config.num_hidden_layers >= range_end
    return (range_start, range_end)


class PartialBertEncoder(nn.Module):
    def __init__(
        self, 
        config: BertConfig,
        keep_embed_layer: bool = True,
        keep_encoders_range: Optional[Tuple[int]] = (0, -1)):
        super().__init__()
        self.config = config
        self.keep_embed_layer = keep_embed_layer
        self.keep_encoders_range = keep_encoders_range
             
        # self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.layer = nn.ModuleList()
        for i in range(self.keep_encoders_range[0], self.keep_encoders_range[1]):
            self.layer.add_module(str(i), BertLayer(config))
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class PartialBertModel(BertModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, 
                 config: BertConfig,
                 keep_embed_layer: bool = True,
                 keep_encoders_range: Optional[Tuple[int]] = (0, -1),
                 add_pooling_layer=False):
        super(BertModel, self).__init__(config)
        self.config = config
        keep_encoders_range = check_keep_encoders_range(config, keep_embed_layer, keep_encoders_range)
        self.keep_embed_layer = keep_embed_layer
        self.keep_encoders_range = keep_encoders_range
        
        self.embeddings = BertEmbeddings(config) if self.keep_embed_layer else None
        if self.keep_encoders_range is not None:
            self.encoder = PartialBertEncoder(config, keep_embed_layer, keep_encoders_range) 
            self.has_last_layer = hasattr(self.encoder.layer, str(self.config.num_hidden_layers - 1)) 
            
            if self.has_last_layer:
                self.pooler = BertPooler(config) if add_pooling_layer else None
            else:
                if add_pooling_layer:
                    raise ValueError("You can't add pooler layer since you drop the top encoder layer.")
        
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        inputs_hidden_state: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False
            
        if self.keep_embed_layer:
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                input_shape = input_ids.size()
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")
            
            batch_size, seq_length = input_shape
            device = input_ids.device if input_ids is not None else inputs_embeds.device

            # past_key_values_length
            past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

            if attention_mask is None:
                attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        
            if token_type_ids is None:
                if hasattr(self.embeddings, "token_type_ids"):
                    buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                    token_type_ids = buffered_token_type_ids_expanded
                else:
                    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
      
            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )
            former_layer_output = embedding_output
            
            if self.keep_encoders_range is None:
                return BaseModelOutputWithPast(last_hidden_state=former_layer_output)
        else: 
            if inputs_hidden_state is not None:
               input_shape = inputs_hidden_state.size()[:-1]
            else:
                raise ValueError("You have to specify inputs_hidden_state") 
            if attention_mask is None:
                raise ValueError("You have to specify attention_mask when inputs_hidden_state is not None")

            device = inputs_hidden_state.device 

            # past_key_values_length
            past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            
            former_layer_output = inputs_hidden_state

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, len(self.encoder.layer))

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        encoder_outputs = self.encoder(
            former_layer_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        
        if self.has_last_layer:
            sequence_output = encoder_outputs[0]
            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

            if not return_dict:
                return (sequence_output, pooled_output) + encoder_outputs[1:]

            return BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                past_key_values=encoder_outputs.past_key_values,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
                cross_attentions=encoder_outputs.cross_attentions,
            )
        else:
            return encoder_outputs
      
if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModel, logging
    logging.set_verbosity_error()
    import torch
    import os

    # test loading single decoder layer
    a, b = None, None
    
    opt = PartialBertModel.from_pretrained(
        pretrained_model_name_or_path="bert-base-uncased",
        keep_embed_layer=False,
        keep_encoders_range=2)
    for name, param in opt.named_parameters():
        if name=="encoder.layer.2.attention.output.dense.weight":
            a = param
            
    opt = AutoModel.from_pretrained("bert-base-uncased")
    for name, param in opt.named_parameters():
        if name=="encoder.layer.2.attention.output.dense.weight":
            b = param
            
    assert int((a != b).sum()) == 0
    
    # test layer-wise loading and inferencing
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vanilla_model = AutoModel.from_pretrained("bert-base-uncased")
    for params in vanilla_model.parameters():
        params.requires_grad = False

    data = tokenizer('hello world')
    input_ids = torch.tensor(data['input_ids']).unsqueeze(0)
    attn_mask = torch.tensor(data['attention_mask']).unsqueeze(0)

    vanilla_model.eval()
    with torch.no_grad():
        og_output = vanilla_model(input_ids=input_ids, attention_mask=attn_mask)

    partial_models = [PartialBertModel.from_pretrained("bert-base-uncased", keep_embed_layer=True, keep_encoders_range=None)]
    for i in range(0, 12):
        partial_models.append(PartialBertModel.from_pretrained("bert-base-uncased", keep_embed_layer=False, keep_encoders_range=i))

    for model in partial_models:
        for params in model.parameters():
            params.requires_grad = False

    last_hidden_state = None
    for i in range(len(partial_models)):
        model = partial_models[i]
        model.eval()
        with torch.no_grad():
            if i == 0:
                output = model(input_ids=input_ids, attention_mask=attn_mask)
                torch.save(output.last_hidden_state, 'temp.pt')
                last_hidden_state = torch.load('temp.pt')
            else:
                output = model(inputs_hidden_state=last_hidden_state, attention_mask=attn_mask)
                torch.save(output.last_hidden_state, 'temp.pt')
                last_hidden_state = torch.load('temp.pt')
        os.remove('temp.pt')
    assert int((output.last_hidden_state != og_output.last_hidden_state).sum()) == 0
