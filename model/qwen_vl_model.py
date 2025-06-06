# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch Qwen2-VL model."""

from typing import List, Optional, Tuple, Union, Callable
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers import Qwen2VLForConditionalGeneration, VivitModel
from transformers.models.qwen2_vl.modeling_qwen2_vl import (Qwen2VLCausalLMOutputWithPast,
                                                            Qwen2VisionTransformerPretrainedModel,
                                                            Qwen2VLVisionConfig)
import os
from safetensors.torch import load_file, save_file
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.cache_utils import StaticCache
from transformers import AutoModel, VivitConfig

logger = logging.get_logger(__name__)



# Copied from transformers.models.llama.modeling_llama._prepare_4d_causal_attention_mask_with_cache_position
def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask

class WeightedMerge(nn.Module):
    def __init__(self, a, d):
        """
        a: 矩阵第二维度大小
        d: 矩阵第三维度大小
        这里假设三个输入矩阵形状都是 (b, a, d)
        """
        super(WeightedMerge, self).__init__()
        # 定义三条分支，每条将 (a*d) -> 1
        self.fc_L = nn.Linear(a * d, 1)
        self.fc_Sv = nn.Linear(a * d, 1)
        self.fc_St = nn.Linear(a * d, 1)

    def forward(self, L, S_v, S_t):
        """
        L, S_v, S_t: 形状均为 (b, a, d)
        输出: 加权合并后的矩阵 (b, a, d)
        """
        b = L.size(0)  # batch size

        # 1. flatten 到 (b, a*d)
        L_flat = L.view(b, -1)  # (b, a*d)
        S_v_flat = S_v.view(b, -1)  # (b, a*d)
        S_t_flat = S_t.view(b, -1)  # (b, a*d)

        # 2. 分别做全连接得到 (b, 1)
        alpha_L = self.fc_L(L_flat)  # (b, 1)
        alpha_Sv = self.fc_Sv(S_v_flat)  # (b, 1)
        alpha_St = self.fc_St(S_t_flat)  # (b, 1)

        # 3. 将三个 (b, 1) 拼成 (b, 3)，对每个batch做 softmax 得到权重
        alphas = torch.cat([alpha_L, alpha_Sv, alpha_St], dim=1)  # (b, 3)
        weights = F.softmax(alphas, dim=1)  # (b, 3)

        # 4. 将权重维度拓展为 (b, 1, 1)，便于与 (b, a, d) 广播
        w_L = weights[:, 0].unsqueeze(-1).unsqueeze(-1)  # (b, 1, 1)
        w_Sv = weights[:, 1].unsqueeze(-1).unsqueeze(-1)  # (b, 1, 1)
        w_St = weights[:, 2].unsqueeze(-1).unsqueeze(-1)  # (b, 1, 1)

        # 5. 按权重加权后再相加，得到 (b, a, d)
        out = w_L * L + w_Sv * S_v + w_St * S_t

        return out


class LMRModule(nn.Module):
    def __init__(self, d):
        super(LMRModule, self).__init__()
        # 定义可训练矩阵(4, d)，相当于4个可训练向量
        self.vecs = nn.Parameter(torch.randn(4, d), requires_grad=True)

    def forward(self, t1, t2, t3):
        """
        t1, t2, t3: 形状均为 (b, a, d) 的张量
        返回:       形状为 (b, 3a + 4, d) 的张量
        """
        b, a, d = t1.shape

        # 将 (4, d) 的可训练矩阵扩展到 (b, 4, d) 以便和 t1, t2, t3 在 batch 维度上对齐
        vecs_expanded = self.vecs.unsqueeze(0).expand(b, -1, -1)  # (b, 4, d)

        # 分别取出四个向量，并在第二维(即原先的 a 维)上进行拼接
        # vecs_expanded[:, 0:1, :] -> (b, 1, d)
        # t1                    -> (b, a, d)
        # vecs_expanded[:, 1:2, :] -> (b, 1, d)
        # t2                    -> (b, a, d)
        # vecs_expanded[:, 2:3, :] -> (b, 1, d)
        # t3                    -> (b, a, d)
        # vecs_expanded[:, 3:4, :] -> (b, 1, d)

        out = torch.cat([
            vecs_expanded[:, 0:1, :],
            t1,
            vecs_expanded[:, 1:2, :],
            t2,
            vecs_expanded[:, 2:3, :],
            t3,
            vecs_expanded[:, 3:4, :],
        ], dim=1)  # 在第1维进行拼接

        # out 的形状将是 (b, 3a + 4, d)
        return out


class LMRAttModule(nn.Module):
    def __init__(self, d):
        super(LMRAttModule, self).__init__()
        # 定义可训练矩阵(4, d)，相当于4个可训练向量
        self.vecs = nn.Parameter(torch.randn(2, d), requires_grad=True)

    def forward(self, t1):
        """
        t1, t2, t3: 形状均为 (b, a, d) 的张量
        返回:       形状为 (b, 3a + 4, d) 的张量
        """
        b, a, d = t1.shape

        # 将 (4, d) 的可训练矩阵扩展到 (b, 4, d) 以便和 t1, t2, t3 在 batch 维度上对齐
        vecs_expanded = self.vecs.unsqueeze(0).expand(b, -1, -1)  # (b, 4, d)

        # 分别取出四个向量，并在第二维(即原先的 a 维)上进行拼接
        # vecs_expanded[:, 0:1, :] -> (b, 1, d)
        # t1                    -> (b, a, d)
        # vecs_expanded[:, 1:2, :] -> (b, 1, d)
        # t2                    -> (b, a, d)
        # vecs_expanded[:, 2:3, :] -> (b, 1, d)
        # t3                    -> (b, a, d)
        # vecs_expanded[:, 3:4, :] -> (b, 1, d)

        out = torch.cat([
            vecs_expanded[:, 0:1, :],
            t1,
            vecs_expanded[:, 1:2, :],
        ], dim=1)  # 在第1维进行拼接

        # out 的形状将是 (b, 3a + 4, d)
        return out


class Qwen2LMRVLForConditionalGeneration(Qwen2VLForConditionalGeneration):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.attention = None
        self.lmr_module = None
        self.lmr_projection = None
        if not hasattr(self.config, "use_lmr"):
            self.config.use_lmr = False
        if not hasattr(self.config, "use_attention"):
            self.config.use_attention = False
        self.pad_length = 0

        if self.config.use_lmr:
            self.use_lmr()
        if self.config.use_attention:
            self.enable_attention()

    def use_lmr(self):
        self.lmr_module = LMRModule(64)
        self.config.use_lmr = True
        self.lmr_projection = nn.Linear(64, self.config.hidden_size)
        self.pad_length = 49 * 3 + 4
        print('lmr is enabled')

    def enable_attention(self):
        assert self.use_lmr
        self.lmr_module = LMRAttModule(64)
        self.attention = WeightedMerge(49, 64)
        self.config.use_attention = True
        self.pad_length = 49 + 2
        print('attention is enabled')

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        l_matrix: Optional[torch.FloatTensor] = None,
        s_v_matrix: Optional[torch.FloatTensor] = None,
        s_t_matrix: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # import pdb; pdb.set_trace()
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if self.config.use_lmr and not inputs_embeds.shape[1] == 1:
                if self.config.use_attention:
                    out = self.attention(l_matrix, s_v_matrix, s_t_matrix)
                    out = self.lmr_module(out)
                else:
                    out = self.lmr_module(l_matrix, s_v_matrix, s_t_matrix)
                out = self.lmr_projection(out)
                inputs_embeds = torch.cat([out, inputs_embeds], dim=1)
                prefix_mask = input_ids.new_ones((input_ids.size(0), out.size(1)))
                input_ids = torch.cat([prefix_mask, input_ids], dim=1)
                if attention_mask is not None:
                    # attention_mask.shape = [B, T]
                    prefix_mask = attention_mask.new_ones((attention_mask.size(0), out.size(1)))
                    # 其中 out.size(1) = ΔT
                    attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
                if labels is not None:
                    # labels.shape = [B, T]
                    prefix_labels = labels.new_full((labels.size(0), out.size(1)), -100)
                    labels = torch.cat([prefix_labels, labels], dim=1)

            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # import pdb; pdb.set_trace()
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
        )

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.
            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        l_matrix=None,
        s_v_matrix=None,
        s_t_matrix=None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        rope_deltas = kwargs.get("rope_deltas", None)
        if attention_mask is not None and position_ids is None:
            if cache_position is None or (cache_position is not None and cache_position[0] == 0):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                if self.pad_length > 0:
                    position_ids = position_ids + self.pad_length
                    prefix = torch.arange(self.pad_length).view(1, 1, -1).expand(3, position_ids.shape[1], self.pad_length)
                    prefix = prefix.to(position_ids.device)
                    position_ids = torch.cat([prefix, position_ids], dim=-1)
                # import pdb; pdb.set_trace()
            else:
                batch_size, seq_length = input_ids.shape
                # seq_length += self.pad_length
                delta = (
                    cache_position[0] + rope_deltas if cache_position is not None and rope_deltas is not None else 0
                )
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "rope_deltas": rope_deltas,
                "l_matrix": l_matrix,
                "s_v_matrix": s_v_matrix,
                "s_t_matrix": s_t_matrix
            }
        )
        return model_inputs
