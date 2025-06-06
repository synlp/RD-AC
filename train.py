import torch
from datasets import Dataset
from dataclasses import dataclass, field
from transformers import AutoTokenizer, HfArgumentParser
from qwen_vl_utils import process_vision_info
from typing import Optional
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
from PIL import Image
import argparse
import json
import re
import os
import numpy as np
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from model.qwen_vl_model import Qwen2LMRVLForConditionalGeneration
from dataclasses import dataclass
from typing import Any, List, Dict
from transformers import PreTrainedTokenizerBase

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lmr: bool = field(
            default=False,
            metadata={"help": "use lmr"}
        )
    use_attention: bool = field(
            default=False,
            metadata={"help": "use attention"}
        )


@dataclass
class DataArguments:
    training_data_path: str = field(default=None,
                                    metadata={"help": "Path to the training data."})
    training_image_dir: str = field(default=None,
                                    metadata={"help": "Path to the training data."})
    training_lmr_dir: str = field(default=None,
                                    metadata={"help": "Path to the training data."})
    data_name: str = field(default=None,
                      metadata={"help": "Path to the training data."})
    small_data: bool = field(
            default=False,
            metadata={"help": "use attention"}
        )


@dataclass
class MultiModalCollator:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1. Separate out text features that we want to tokenize/pad:
        text_features = []
        for i, f in enumerate(features):
            # 可以先简单检查一下 f 的结构
            if not isinstance(f.get("input_ids"), list):
                print(f"[DEBUG] feature index={i} has non-list input_ids:", f["input_ids"])
            if not isinstance(f.get("attention_mask"), list):
                print(f"[DEBUG] feature index={i} has non-list attention_mask:", f["attention_mask"])
            if not isinstance(f.get("labels"), list):
                print(f"[DEBUG] feature index={i} has non-list labels:", f["labels"])

            text_features.append({
                "input_ids": f["input_ids"],
                "attention_mask": f["attention_mask"],
                "labels": f["labels"]
            })

        # 2. 用 try/except 捕获 tokenizer.pad(...) 的报错
        try:
            batch_text = self.tokenizer.pad(
                text_features,
                return_tensors="pt",
            )
        except Exception as e:
            # 如果这里报错，就打印出出问题的 text_features
            print("\n[ERROR] tokenizer.pad(...) failed. Below is the text_features content:\n")
            for i, tf in enumerate(text_features):
                print(f"  === Sample {i} ===")
                print("  input_ids:", tf["input_ids"])
                print("  attention_mask:", tf["attention_mask"])
                print("  labels:", tf["labels"])
                print("  ----------------")
            raise e  # 再把错误抛出

        # 3. 对其他 multi-modal 字段做 stack
        pixel_values = torch.stack([torch.tensor(f["pixel_values"]) for f in features], dim=0)
        image_grid_thw = torch.stack([torch.tensor(f["image_grid_thw"]) for f in features], dim=0)
        l_matrix = torch.stack([torch.tensor(f["l_matrix"]) for f in features], dim=0)
        s_v_matrix = torch.stack([torch.tensor(f["s_v_matrix"]) for f in features], dim=0)
        s_t_matrix = torch.stack([torch.tensor(f["s_t_matrix"]) for f in features], dim=0)

        # 4. Merge
        batch = {
            "input_ids": batch_text["input_ids"],
            "attention_mask": batch_text["attention_mask"],
            "labels": batch_text["labels"],
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "l_matrix": l_matrix,
            "s_v_matrix": s_v_matrix,
            "s_t_matrix": s_t_matrix,
        }

        # import pdb; pdb.set_trace()

        return batch

def process_func_msed(example, data_args, tokenizer, processor, max_length=32000):
    """
    专门针对 msde 的数据格式进行预处理。
    JSON示例:
    {
      "id": 1,
      "img": "train/images/1.jpg",
      "text": "Overweight man and woman jogging in the city...",
      "label": "neutral",
      "image_emb": "1/image_emb.npy",
      "text_emb": "1/text_emb.npy",
      "L_path": "1/L.npy",
      "S_image_path": "1/S_image.npy",
      "S_text_path": "1/S_text.npy"
    }
    """

    # ---------- 1. 读取图像 ----------
    img_path = os.path.join(data_args.training_image_dir, example["img"])  # 可能是"train/images/1.jpg"
    image_pil = Image.open(img_path).convert("RGB")
    fixed_size = (224, 224)
    image_pil = image_pil.resize(fixed_size, Image.BICUBIC)

    # ---------- 2. 读取 L, S_image, S_text 矩阵 ----------
    try:
        l_path = os.path.join(data_args.training_lmr_dir, example["L_path"])
        s_img_path = os.path.join(data_args.training_lmr_dir, example["S_image_path"])
        s_txt_path = os.path.join(data_args.training_lmr_dir, example["S_text_path"])

        l_matrix = torch.from_numpy(np.load(l_path)).float()
        s_image_matrix = torch.from_numpy(np.load(s_img_path)).float()
        s_text_matrix = torch.from_numpy(np.load(s_txt_path)).float()
    except Exception as e:
        l_matrix = torch.zeros((49, 64))
        s_image_matrix = torch.zeros((49, 64))
        s_text_matrix = torch.zeros((49, 64))

    # ---------- 3. 构造输入文本 & 输出文本 ----------
    # 这里示例：把 example["text"] 放在"用户问题"里，问模型: "What's the sentiment? (positive/neutral/negative)"
    user_text = example["text"] + "\nWhat is the emotion of the multimodal content?\n"
    label_text = example["label"]  # 例如 "neutral"

    # ---------- 4. 用 Qwen2VL 的对话模板进行包装 ----------
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_pil,
                    "resize_height": 224,
                    "resize_width": 224,
                },
                {
                    "type": "text",
                    "text": user_text
                },
            ],
        },
    ]

    # ---------- 5. 用 processor 得到 tokenized 输入 ----------
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        do_resize=True,
        padding=True,
    )
    # processor 返回 tensor，需要转成 list 便于后面手动拼接
    inputs = {k: v.tolist() for k, v in inputs.items()}

    # assistant 的文字（即 label_text）也要拼进最终 tokens
    response = tokenizer([label_text], add_special_tokens=False)

    # ---------- 6. 拼接 input_ids、attention_mask、labels ----------
    input_ids = inputs["input_ids"][0] + response["input_ids"][0] + [tokenizer.pad_token_id]
    attention_mask = inputs["attention_mask"][0] + response["attention_mask"][0] + [1]
    labels = (
        [-100] * len(inputs["input_ids"][0]) + response["input_ids"][0] + [tokenizer.pad_token_id]
    )

    # 截断
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    # ---------- 7. 处理 image_grid_thw (去掉多余的 batch 维度) ----------
    gthw = torch.tensor(inputs["image_grid_thw"])
    if gthw.shape == (1, 3):
        gthw = gthw.squeeze(0)

    # ---------- 8. 返回构造好的字典 ----------
    # 注意: 不要再转换为 torch.Tensor，直接返回 list 或小 tensor
    #       collator 中会统一处理
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": inputs["pixel_values"],
        "image_grid_thw": gthw,
        "l_matrix": l_matrix,
        "s_v_matrix": s_image_matrix,
        "s_t_matrix": s_text_matrix,
    }


def process_func_hmc(example, data_args, tokenizer, processor, max_length=32000):
    """
    专门针对 hmc 的数据格式进行预处理。
    example: {
      "id": 42953,
      "img": "img/42953.png",
      "label": "not-hateful",
      "text": "its their character not their color that matters",
      "image_last_hidden_state": "42953/image_last_hidden_state.npy",
      "text_last_hidden_state": "42953/text_last_hidden_state.npy",
      "L_path": "42953/L.npy",
      "S1_path": "42953/S1.npy",
      "S2_path": "42953/S2.npy"
    }
    """

    # ---------- 1. 读取图像 ----------
    img_path = os.path.join(data_args.training_image_dir, example["img"])
    image_pil = Image.open(img_path).convert("RGB")
    fixed_size = (224, 224)
    image_pil = image_pil.resize(fixed_size, Image.BICUBIC)

    # ---------- 2. 读取 L, S1, S2 矩阵 ----------
    l_path = os.path.join(data_args.training_lmr_dir, example["L_path"])
    s1_path = os.path.join(data_args.training_lmr_dir, example["S1_path"])
    s2_path = os.path.join(data_args.training_lmr_dir, example["S2_path"])

    l_matrix = torch.from_numpy(np.load(l_path)).float()
    s_v_matrix = torch.from_numpy(np.load(s1_path)).float()
    s_t_matrix = torch.from_numpy(np.load(s2_path)).float()

    # ---------- 3. 准备构造输入文本 & 输出文本 ----------
    #    这里示例做法：把用户的文本 + 图片视作输入，label 视作模型需要生成的输出。
    #    也可以根据自己需要改成其它prompt风格。
    user_text = example["text"] + '\nIs the multimodal content hateful or not-hateful?\n'
    label_text = example["label"]  # hmc是分类标签，如 "not-hateful"
    # label_text = trans[label_text]

    # 用 Qwen2VL 的对话模板进行包装:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_pil,
                    "resize_height": 224,
                    "resize_width": 224,
                },
                {
                    "type": "text",
                    "text": user_text
                },
            ],
        },
    ]

    # ---------- 4. 用 processor 得到 tokenized 输入 ----------
    #    注意：Qwen2VL 自带的 chat 模板，可以先把 messages 丢给 apply_chat_template，
    #    然后在 process_vision_info 里做图像预处理等等。
    #    这里演示将图像预处理合并到 single image 方式，只需符合 Qwen2VL 的输入格式即可。
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    # 因为这里只有 1 张图
    # 也可以考虑对多图做循环处理

    inputs = processor(
        text=[text_input],
        images=image_inputs,  # 只有1张时就是 [ PIL_image ]
        videos=video_inputs,
        return_tensors="pt",
        do_resize=True,
        padding=True,
    )
    # processor 返回的 inputs 默认是 tensor，需要变成 list 以便和下面逻辑合并
    inputs = {k: v.tolist() for k, v in inputs.items()}

    # assistant 的文字（即 label_text）也要拼进最终 tokens
    # 这里把 label_text 视作最终要预测的部分
    response = tokenizer([label_text], add_special_tokens=False)

    # ---------- 5. 拼接 input_ids、attention_mask、labels ----------
    input_ids = inputs["input_ids"][0] + response["input_ids"][0] + [tokenizer.pad_token_id]
    attention_mask = inputs["attention_mask"][0] + response["attention_mask"][0] + [1]
    labels = (
            [-100] * len(inputs["input_ids"][0])
            + response["input_ids"][0]
            + [tokenizer.pad_token_id]
    )

    # 截断
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    # 不要再对 image_grid_thw 进行多余的 stack 了，只要保证它是 (3,)
    gthw = torch.tensor(inputs["image_grid_thw"])
    # 如果它是 (1,3)，就 squeeze 成 (3,)
    if gthw.shape == (1, 3):
        gthw = gthw.squeeze(0)

    # DO NOT turn them into torch.Tensor here. Keep them as lists:
    # input_ids = torch.tensor(input_ids, dtype=torch.long)  # remove
    # attention_mask = torch.tensor(attention_mask, dtype=torch.long)  # remove
    # labels = torch.tensor(labels, dtype=torch.long)  # remove

    # Keep pixel_values, image_grid_thw, etc. as lists (or at least consistently shaped).
    # Or if you must keep them as Tensors, you'll need a custom collator that knows how to handle them.

    # Return lists for text fields so DataCollatorForSeq2Seq can do the padding
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": inputs["pixel_values"],     # might still be nested
        "image_grid_thw": gthw, # might still be nested
        "l_matrix": l_matrix,
        "s_v_matrix": s_v_matrix,
        "s_t_matrix": s_t_matrix
    }

def process_func_twitter(example, data_args, tokenizer, processor, max_length=32000):
    """
    专门针对 Twitter 的数据格式进行预处理。
    例如:
      {
        "id": 1,
        "label": "positive",
        "img": "1860693.jpg",
        "text": "RT @ ltsChuckBass : <t> Chuck Bass </t> is everything # MCM",
        "image_emb": "1/image_emb.npy",
        "text_emb": "1/text_emb.npy",
        "L_path": "1/L.npy",
        "S_image_path": "1/S_image.npy",
        "S_text_path": "1/S_text.npy"
      }
    """

    # 1. 读取图像
    img_path = os.path.join(data_args.training_image_dir, example["img"])
    image_pil = Image.open(img_path).convert("RGB")
    fixed_size = (224, 224)
    image_pil = image_pil.resize(fixed_size, Image.BICUBIC)

    # 2. 读取 L, S_image, S_text 矩阵
    l_path = os.path.join(data_args.training_lmr_dir, example["L_path"])
    s_img_path = os.path.join(data_args.training_lmr_dir, example["S_image_path"])
    s_txt_path = os.path.join(data_args.training_lmr_dir, example["S_text_path"])

    l_matrix = torch.from_numpy(np.load(l_path)).float()
    s_image_matrix = torch.from_numpy(np.load(s_img_path)).float()
    s_text_matrix = torch.from_numpy(np.load(s_txt_path)).float()

    # 3. 构造用户输入 & 标签
    user_text = example["text"] + "\nWhat is the sentiment of the multimodal content positive, negative, or neutral?\n"
    label_text = example["label"]  # 比如 "positive" / "neutral" / "negative"

    # 用 Qwen2VL 的对话模板
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_pil,
                    "resize_height": 224,
                    "resize_width": 224,
                },
                {
                    "type": "text",
                    "text": user_text
                },
            ],
        },
    ]

    # 4. 用 processor 得到 tokenized 输入
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        do_resize=True,
        padding=True,
    )
    # 把 tensor 转成 list，便于后面手动拼接
    inputs = {k: v.tolist() for k, v in inputs.items()}

    # 5. 构造要预测的 labels
    response = tokenizer([label_text], add_special_tokens=False)
    input_ids = inputs["input_ids"][0] + response["input_ids"][0] + [tokenizer.pad_token_id]
    attention_mask = inputs["attention_mask"][0] + response["attention_mask"][0] + [1]
    labels = (
        [-100] * len(inputs["input_ids"][0])  # 前面上下文都用 -100
        + response["input_ids"][0]
        + [tokenizer.pad_token_id]
    )

    # 截断
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    # 6. 处理 image_grid_thw
    gthw = torch.tensor(inputs["image_grid_thw"])
    if gthw.shape == (1, 3):
        gthw = gthw.squeeze(0)

    # 7. 返回结果 dict (lists or small tensors)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": inputs["pixel_values"],
        "image_grid_thw": gthw,
        "l_matrix": l_matrix,
        "s_v_matrix": s_image_matrix,  # 这两个命名看你喜欢
        "s_t_matrix": s_text_matrix,
    }


# parser = argparse.ArgumentParser()
# parser.add_argument('--model_name_or_path', required=True, type=str)
# parser.add_argument('--training_data_path', required=True, type=str)
# # parser.add_argument('--output_dir', required=True, type=str)
# main_args = parser.parse_args()

parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# 使用Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

with open(data_args.training_data_path, 'r', encoding='utf-8') as f:
    training_data = json.load(f)['train']

# 读取 hmc 格式数据
with open(data_args.training_data_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)['train']

# 转换成 huggingface 的 Dataset 对象
train_ds = Dataset.from_list(raw_data)

if data_args.small_data:
    # 这里演示只取一部分数据
    train_ds = train_ds.select(range(min(100, len(train_ds))))

if data_args.data_name == "hmc":
    # 调用我们的专用函数进行 map
    train_dataset = train_ds.map(
        lambda ex: process_func_hmc(ex, data_args, tokenizer, processor),
        num_proc=8
    )
elif data_args.data_name == 'twitter':
    train_dataset = train_ds.map(
        lambda ex: process_func_twitter(ex, data_args, tokenizer, processor),
        num_proc=8
    )
elif data_args.data_name == 'msed':
    train_dataset = train_ds.map(
        lambda ex: process_func_msed(ex, data_args, tokenizer, processor),
        num_proc=8
    )
else:
    raise ValueError()

model = Qwen2LMRVLForConditionalGeneration.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
# model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
if model_args.use_lmr:
    model.use_lmr()
if model_args.use_attention:
    model.enable_attention()

data_collator = MultiModalCollator(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,  # custom collator
)

# # 配置Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
# )

# 开启模型训练
trainer.train()
trainer.save_model(training_args.output_dir)
# model.save_model(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)
model.config.save_pretrained(training_args.output_dir)

