import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
from transformers import AutoProcessor, AutoTokenizer, CLIPModel

# -----------------------------
# 简单数据集类
# -----------------------------
class MyDataset(Dataset):
    """
    读取 JSONL 文件, 将每条记录(json)存储到 self.samples 里
    """
    def __init__(self, jsonl_path):
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# 其他辅助函数
def text_upsample_or_truncate(emb_text: np.ndarray, target_rows: int = 576) -> np.ndarray:
    T, d = emb_text.shape
    if T == target_rows:
        return emb_text
    if T > target_rows:
        return emb_text[:target_rows, :]

    repeated = np.zeros((target_rows, d), dtype=emb_text.dtype)
    block_size = target_rows // T
    remainder = target_rows % T
    idx = 0
    for i in range(T):
        count = block_size
        if i < remainder:
            count += 1
        for _ in range(count):
            repeated[idx] = emb_text[i]
            idx += 1
    return repeated


def average_pool_1d_divisible(x: torch.Tensor, c: int) -> torch.Tensor:
    """
    将形状 (a, b) 的张量在维度 1（第二个维度）平均池化到 size = c。
    假设 b 可以被 c 整除，不引入可学习参数。

    :param x: 输入张量，形状 (a, b)
    :param c: 目标输出维度
    :return: shape = (a, c) 的输出张量
    """
    # x.shape = (a, b)
    a, b = x.shape

    if b % c != 0:
        raise ValueError(f"维度 b={b} 不能被 c={c} 整除，请使用自适应方式或其他方法。")

    k = b // c  # 每段的大小
    # 重塑为 (a, c, k)
    x_reshaped = x.view(a, c, k)
    # 对最后一维 k 做平均
    # 结果 shape = (a, c)
    out = x_reshaped.mean(dim=-1)
    return out

# -----------------------------
# 在单个线程中，对子集数据进行处理的函数
# -----------------------------
def process_subset_on_gpu(
    subset_data,
    gpu_id,
    output_dir,
    output_prefix,
    data_dir,
    clip_model_path,
    ### 新增：传入 pbar 和 pbar_lock，用于实时更新进度条 ###
    pbar,
    pbar_lock
):
    """
    在本函数里：
    1) 加载模型(vision+text)到指定 GPU
    2) 逐条处理 subset_data
    3) 保存生成的临时 JSONL 文件
    4) 返回 tmp_jsonl_path
    """
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    # 1. 加载 CLIP 模型和处理器
    clip_model = CLIPModel.from_pretrained(clip_model_path).to(device)
    clip_model.eval()
    vision_model = clip_model.vision_model
    vision_projection = clip_model.visual_projection
    vision_processor = AutoProcessor.from_pretrained(clip_model_path)
    text_model = clip_model.text_model
    text_projection = clip_model.text_projection
    tokenizer = AutoTokenizer.from_pretrained(clip_model_path)

    # 创建一个临时 JSONL 文件，用于保存本线程处理后的结果
    tmp_jsonl_path = os.path.join(output_dir, f"{output_prefix}_gpu{gpu_id}.jsonl")
    tmp_file = open(tmp_jsonl_path, 'w', encoding='utf-8')

    with torch.no_grad():
        for record in subset_data:
            sample_id = record["id"]
            img_rel_path = record["img"]
            text = record["text"]

            # 处理图像
            img_full_path = os.path.join(data_dir, img_rel_path)
            image = Image.open(img_full_path).convert("RGB")
            image_inputs = vision_processor(images=image, return_tensors="pt")
            for k in image_inputs:
                image_inputs[k] = image_inputs[k].to(device)

            image_outputs = vision_model(**image_inputs)
            img_last_hidden_state = image_outputs.last_hidden_state
            img_last_hidden_state = vision_projection(img_last_hidden_state)[0][1:]
            l, d = img_last_hidden_state.shape
            c = 256
            img_last_hidden_state = average_pool_1d_divisible(img_last_hidden_state, c)
            img_last_hidden_state = img_last_hidden_state.cpu().numpy().astype(np.float16)

            # 处理文本
            text_inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
            for k in text_inputs:
                text_inputs[k] = text_inputs[k].to(device)

            text_outputs = text_model(**text_inputs)
            txt_last_hidden_state = text_outputs.last_hidden_state
            txt_last_hidden_state = text_projection(txt_last_hidden_state)[0]
            txt_last_hidden_state = average_pool_1d_divisible(txt_last_hidden_state, c)
            txt_last_hidden_state = txt_last_hidden_state.cpu().numpy().astype(np.float16)
            txt_last_hidden_state = text_upsample_or_truncate(
                txt_last_hidden_state, l
            )

            # 保存到文件
            sample_out_dir = os.path.join(output_dir, str(sample_id))
            os.makedirs(sample_out_dir, exist_ok=True)

            img_emb_path = os.path.join(sample_out_dir, "image_last_hidden_state.npy")
            txt_emb_path = os.path.join(sample_out_dir, "text_last_hidden_state.npy")
            np.save(img_emb_path, img_last_hidden_state)
            np.save(txt_emb_path, txt_last_hidden_state)

            record["image_last_hidden_state"] = os.path.join(str(sample_id), "image_last_hidden_state.npy")
            record["text_last_hidden_state"] = os.path.join(str(sample_id), "text_last_hidden_state.npy")
            tmp_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            # === 关键：处理完一条，就更新进度条 ===
            with pbar_lock:
                pbar.update(1)

    tmp_file.close()
    return tmp_jsonl_path  # 不再返回 count

# -----------------------------
# 多线程主函数：将数据均分到多个 GPU，分别处理
# -----------------------------
def process_split_in_threads(
    jsonl_path,
    output_dir,
    output_prefix,
    data_dir,
    gpu_ids,
    clip_model_path,
):
    dataset = MyDataset(jsonl_path)
    all_data = dataset.samples
    total_samples = len(all_data)
    print(f"[{jsonl_path}] 总共有 {total_samples} 条数据")

    if total_samples == 0:
        print(f"警告：{jsonl_path} 没有数据，跳过处理")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1) 均分数据到 len(gpu_ids) 份
    num_gpus = len(gpu_ids)
    chunk_size = (total_samples + num_gpus - 1) // num_gpus
    chunks = []
    start_idx = 0
    for i in range(num_gpus):
        end_idx = start_idx + chunk_size
        subset_data = all_data[start_idx:end_idx]
        chunks.append(subset_data)
        start_idx = end_idx

    # 2) 多线程并行处理
    from threading import Lock
    pbar_lock = Lock()  # 用于保护进度条
    futures = []
    tmp_jsonl_files = []

    print(f"为 {num_gpus} 个 GPU 启动 {num_gpus} 个线程，每个线程单独加载模型处理相应数据")

    with ThreadPoolExecutor(max_workers=num_gpus) as executor, \
         tqdm(total=total_samples, desc=f"Processing {output_prefix}") as pbar:
        for i, subset_data in enumerate(chunks):
            if len(subset_data) == 0:
                continue
            gpu_id = gpu_ids[i]
            future = executor.submit(
                process_subset_on_gpu,
                subset_data=subset_data,
                gpu_id=gpu_id,
                output_dir=output_dir,
                output_prefix=output_prefix,
                data_dir=data_dir,
                clip_model_path=clip_model_path,
                pbar=pbar,             # 传给子线程
                pbar_lock=pbar_lock    # 传给子线程
            )
            futures.append(future)

        # 3) 收集结果 (不再更新进度，这里只是等待所有线程完成)
        for fut in as_completed(futures):
            tmp_file = fut.result()
            tmp_jsonl_files.append(tmp_file)

    # 4) 合并临时 JSONL
    final_jsonl_path = os.path.join(output_dir, f"{output_prefix}_out.jsonl")
    with open(final_jsonl_path, 'w', encoding='utf-8') as fout:
        for tf in sorted(tmp_jsonl_files):
            with open(tf, 'r', encoding='utf-8') as fi:
                for line in fi:
                    fout.write(line)

    # 删除临时文件
    for tf in tmp_jsonl_files:
        os.remove(tf)

    print(f"[{jsonl_path}] 处理完成，结果写到 -> {final_jsonl_path}")

# -----------------------------
# 主函数：对 train / dev / test 分别调用
# -----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="包含 train.jsonl, dev.jsonl, test.jsonl, img/ 等的目录")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出特征和新的 jsonl 的目录")
    parser.add_argument("--gpu_ids", type=str, default="0",
                        help="可用的 GPU ID 列表, 用逗号分隔, e.g. '1,2,3,4'")
    parser.add_argument("--clip_model_path", type=str, default="openai/clip-vit-base-patch32",
                        help="CLIP模型路径")
    args = parser.parse_args()

    # GPU 列表
    gpu_id_list = [int(x) for x in args.gpu_ids.split(",")]

    # 数据文件
    train_jsonl = os.path.join(args.data_dir, "train.jsonl")
    dev_jsonl   = os.path.join(args.data_dir, "dev.jsonl")
    test_jsonl  = os.path.join(args.data_dir, "test.jsonl")

    if os.path.isfile(train_jsonl):
        process_split_in_threads(
            jsonl_path=train_jsonl,
            output_dir=args.output_dir,
            output_prefix="train",
            data_dir=args.data_dir,
            gpu_ids=gpu_id_list,
            clip_model_path=args.clip_model_path,
        )
    if os.path.isfile(dev_jsonl):
        process_split_in_threads(
            jsonl_path=dev_jsonl,
            output_dir=args.output_dir,
            output_prefix="dev",
            data_dir=args.data_dir,
            gpu_ids=gpu_id_list,
            clip_model_path=args.clip_model_path,
        )
    if os.path.isfile(test_jsonl):
        process_split_in_threads(
            jsonl_path=test_jsonl,
            output_dir=args.output_dir,
            output_prefix="test",
            data_dir=args.data_dir,
            gpu_ids=gpu_id_list,
            clip_model_path=args.clip_model_path,
        )

    print("全部处理完成。")

if __name__ == "__main__":
    main()
