# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch T5 model."""

import os
import copy
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    logging,
)
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import (
    T5LayerNorm,
    T5Block,
    T5Stack,
    T5EncoderModel,
)

logger = logging.get_logger(__name__)

try:
    from apex.normalization import FusedRMSNorm

    T5LayerNorm = FusedRMSNorm  # noqa

    logger.info("Discovered apex.normalization.FusedRMSNorm - will use it instead of T5LayerNorm")
except ImportError:
    # using the normal T5LayerNorm
    pass
except Exception:
    logger.warning("discovered apex but it failed to load, falling back to T5LayerNorm")
    pass

ALL_LAYERNORM_LAYERS.append(T5LayerNorm)


def check_keep_hidden_layers_range(
    config: T5Config,
    keep_embed_layer: bool,
    keep_hidden_layers_range: Optional[Union[int, Tuple[int, Optional[int]]]]):
    """
    Validate and adjust the provided range of encoder/decoder layers to keep during optimization.

    This function takes in the model configuration, a boolean indicating whether to keep the
    embedding layer, and a range of encoder/decoder layers to keep. It checks if the provided range is
    valid and adjusts it if necessary. If no range is provided, the function returns None.

    Args:
        config (T5Config): The model configuration object.
        keep_embed_layer (bool): Whether to keep the embedding layer during optimization.
        keep_hidden_layers_range (Optional[Union[int, Tuple[int, Optional[int]]]]): The range of decoder
            layers to keep during optimization. The range is a left-closed, right-open interval [a, b).

    Raises:
        ValueError: If keep_hidden_layers_range is not specified and keep_embed_layer is False, or if the
            provided range is invalid (e.g., range_end <= range_start).

    Returns:
        Optional[Tuple[int, int]]: A tuple (range_start, range_end), representing the adjusted range
            of encoder/decoder layers to keep, or None if keep_hidden_layers_range is None.

    """
    if keep_hidden_layers_range is None:
        if not keep_embed_layer:
            raise ValueError("keep_hidden_layers_range must be specified if keep_embed_layer is False")
        return None
    elif isinstance(keep_hidden_layers_range, int):
        range_start = keep_hidden_layers_range
        range_end = range_start + 1
    else:
        assert len(keep_hidden_layers_range) == 2
        range_start, range_end = keep_hidden_layers_range

    assert isinstance(range_start, int)

    if range_start != 0 and keep_embed_layer:
        first_layer = - config.num_layers if range_start < 0 else 0
        if range_start != 0 and range_start != first_layer:
            raise ValueError(
                "You can't keep the embedding layer since you drop the first encoder layer"
            )

    if range_start < 0:
        range_start = config.num_layers + range_start

    if range_end is None:
        range_end = config.num_layers
    elif range_end <= 0:
        range_end = config.num_layers + range_end

    if range_end <= range_start:
        raise ValueError("range_end should be greater than range_start")

    assert range_start >= 0 
    assert range_end > range_start
    assert config.num_layers >= range_end
    return (range_start, range_end)


class BaseModelOutputWithPositionBias(BaseModelOutput):
    def __init__(
        self,
        last_hidden_state,
        past_key_values,
        position_bias,
        encoder_decoder_position_bias):
        self.last_hidden_state = last_hidden_state
        self.past_key_values = past_key_values
        self.position_bias = position_bias
        self.encoder_decoder_position_bias = encoder_decoder_position_bias


class PartialT5Stack(T5Stack):
    def __init__(
        self,
        config,
        embed_tokens=None,
        keep_embed_layer=True,
        keep_hidden_layers_range=None,
        ):
        super(T5Stack, self).__init__(config)
        
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self.keep_embed_layer = False if self.is_decoder else keep_embed_layer
        self.keep_hidden_layers_range = keep_hidden_layers_range
        
        if self.keep_hidden_layers_range is not None:
            self.block = nn.ModuleList()
            start, end = self.keep_hidden_layers_range
            if start > 0:
                # will be and should be removed after the relative attention bias
                # is copied to the start block after loaded from the pretrained model
                self.block.add_module("0", T5Block(config, has_relative_attention_bias=True))
                
            for i in range(start, end):
                self.block.add_module(str(i), T5Block(config, has_relative_attention_bias=bool(i == start)))

            if keep_hidden_layers_range[1] == config.num_layers:
                self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
                self.dropout = nn.Dropout(config.dropout_rate)
            else:
                self.final_layer_norm = None
                self.dropout = None
        else:
            self.final_layer_norm = None
            self.dropout = None
            
        self.post_init()
    
    
    def parallelize(self):
        raise NotImplementedError("PartialT5Stack does not support parallelization")
    
    def deparallelize(self):
        raise NotImplementedError("PartialT5Stack does not support parallelization")
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        inputs_hidden_state=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        position_bias=None,
        encoder_decoder_position_bias=None,
        return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.keep_embed_layer:
            if input_ids is not None and inputs_embeds is not None:
                err_msg_prefix = "decoder_" if self.is_decoder else ""
                raise ValueError(
                    f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
                )
            elif input_ids is not None:
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            else:
                err_msg_prefix = "decoder_" if self.is_decoder else ""
                raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

            if inputs_embeds is None:
                assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
                inputs_embeds = self.embed_tokens(input_ids)

            batch_size, seq_length = input_shape

            # required mask seq length can be calculated via length of past
            mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

            if use_cache is True:
                assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

            if attention_mask is None:
                attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

            # hidden_states = self.dropout(inputs_embeds)
            hidden_states = inputs_embeds
            if self.keep_hidden_layers_range is None:
                return BaseModelOutputWithPastAndCrossAttentions(
                    last_hidden_state=hidden_states,
                    past_key_values=None,
                    hidden_states=(hidden_states,),
                    attentions=(),
                    cross_attentions=None,
                )
        else:
            if inputs_hidden_state is not None:
               input_shape = inputs_hidden_state.size()[:-1]
            else:
                raise ValueError("You have to specify inputs_hidden_state") 
            if attention_mask is None:
                raise ValueError("You have to specify attention_mask when inputs_hidden_state is not None")

            hidden_states = inputs_hidden_state
            
        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        
        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None

        start = self.keep_hidden_layers_range[0]
        end = self.keep_hidden_layers_range[1]
        index = range(start, end)
        for i, layer_module, past_key_value in zip(index, self.block, past_key_values):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = self.dropout(hidden_states)
        else:
            return BaseModelOutputWithPositionBias(
                last_hidden_state=hidden_states,
                past_key_values=present_key_value_states,
                position_bias=position_bias,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
            )

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class PartialT5EncoderModel(T5EncoderModel):
    def __init__(
        self,
        config: T5Config,
        keep_embed_layer: bool,
        keep_hidden_layers_range: Optional[Union[int, Tuple[int, Optional[int]]]] = (0, -1),
        ):
        super(T5EncoderModel, self).__init__(config)
        self.keep_hidden_layers_range = \
            check_keep_hidden_layers_range(config, keep_embed_layer, keep_hidden_layers_range)
        self.keep_embed_layer = keep_embed_layer
        
        if keep_embed_layer:
            self.shared = nn.Embedding(config.vocab_size, config.d_model)
        else:
            self.shared = None

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = PartialT5Stack(
            encoder_config, 
            self.shared,
            keep_embed_layer=keep_embed_layer,
            keep_hidden_layers_range=self.keep_hidden_layers_range)

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        res = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        # if two inputs are provided, they are model and loading_info
        if type(res) == tuple:
            model, _ = res
        else:
            model = res
        if model.keep_hidden_layers_range is not None:
            start, end = model.keep_hidden_layers_range
            if start > 0:
                position_module = None, None
                for n, m in model.named_modules():
                    if n.startswith('encoder.block.0') and n.endswith('relative_attention_bias'):
                        position_module = m
                    elif n.endswith('relative_attention_bias'):
                        m.weight = nn.Parameter(position_module.weight.clone())
                        if getattr(m, "bias", None) is not None:
                            m.bias.data = nn.functional.pad(
                                position_module.bias.data, (0, m.weight.shape[0] - position_module.bias.shape[0]), "constant", 0)
                # remove the first block
                module_list = list(model.encoder.block)[1:]
                model.encoder.block = nn.ModuleList()
                for i, m in zip(range(start, end), module_list):
                    model.encoder.block.add_module(str(i), m)
                # set relative_attention_bias as non-trainable
                for n, m in model.named_modules():
                    if n.endswith('relative_attention_bias'):
                        m.weight.requires_grad = False
                        if getattr(m, "bias", None) is not None:
                            m.bias.requires_grad = False
        return res

    def parallelize(self):
        raise NotImplementedError("PartialT5EncoderModel does not support parallelization.")
    
    def deparallelize(self):
        raise NotImplementedError("PartialT5EncoderModel does not support parallelization.")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        inputs_hidden_state: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        position_bias: Optional[torch.Tensor] = None,
        encoder_decoder_position_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import T5Tokenizer, T5EncoderModel

        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5EncoderModel.from_pretrained("t5-small")
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            inputs_hidden_state=inputs_hidden_state,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            position_bias=position_bias,
            encoder_decoder_position_bias=encoder_decoder_position_bias,
            past_key_values=past_key_values,
            return_dict=return_dict,
        )

        return encoder_outputs

if __name__ == "__main__":
    from transformers import AutoTokenizer, T5EncoderModel, logging
    logging.set_verbosity_error()
    import torch
    import os
    huggingface_cache = "../huggingface_cache"
    os.environ["HUGGINGFACE_HUB_CACHE"] = huggingface_cache

    # test loading single decoder layer
    a, b = None, None

    t5 = PartialT5EncoderModel.from_pretrained(
        pretrained_model_name_or_path="google/flan-t5-small",
        keep_embed_layer=False,
        keep_hidden_layers_range=1)
    for name, param in t5.named_parameters():
        if name=="encoder.block.1.layer.0.SelfAttention.q.weight":
            a = param
            
    t5 = T5EncoderModel.from_pretrained("google/flan-t5-small")
    for name, param in t5.named_parameters():
        if name=="encoder.block.1.layer.0.SelfAttention.q.weight":
            b = param
            
    assert int((a != b).sum()) == 0

    # test layer-wise loading and inferencing
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
    og_model = T5EncoderModel.from_pretrained('google/flan-t5-small')
    for params in og_model.parameters():
        params.requires_grad = False
    
    data = tokenizer('hello world')
    input_ids = torch.tensor(data['input_ids']).unsqueeze(0)
    attn_mask = torch.tensor(data['attention_mask']).unsqueeze(0)

    og_model.eval()
    with torch.no_grad():
        og_output = og_model(input_ids=input_ids, attention_mask=attn_mask)

    partial_models = [PartialT5EncoderModel.from_pretrained('google/flan-t5-small', keep_embed_layer=True, keep_hidden_layers_range=None)]

    for i in range(0, 8):
        partial_models.append(PartialT5EncoderModel.from_pretrained('google/flan-t5-small', keep_embed_layer=False, keep_hidden_layers_range=i))

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
                assert torch.isnan(last_hidden_state).any() == False, f"NaN found in layer {i}"
                
            try:
                os.remove('temp.pt')
            except:
                pass
        
    assert int((output.last_hidden_state != og_output.last_hidden_state).sum()) == 0

