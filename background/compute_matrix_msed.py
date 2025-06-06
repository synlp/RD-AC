import os
import csv
import json
import argparse
import numpy as np
from PIL import Image
from typing import List, Dict, Any
from tqdm import tqdm
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from transformers import CLIPModel, AutoProcessor, AutoTokenizer


def parse_csv_line(row: Dict[str, str], row_idx: int) -> Dict[str, Any]:
    """
    给定 CSV 的一行 (字典形式: key=列名, value=字符串)，以及行号 row_idx (1-based)。
    CSV 列示例: Title,Caption,Sentiment,Emotion,Desire,Inference Sequence
    我们只关心:
      - Title
      - Caption
      - Emotion -> 作为 label
    生成:
      "id": row_idx
      "text": Title + " " + Caption
      "label": emotion
    """
    title = row.get("Title", "").strip()
    caption = row.get("Caption", "").strip()
    emotion = row.get("Emotion", "").strip()

    combined_text = f"{title} {caption}".strip()
    if not combined_text:
        combined_text = "N/A"

    return {
        "id": row_idx,
        "text": combined_text,
        "label": emotion
    }


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


def process_csv_lines_on_gpu(
    lines: List[Dict[str, str]],  # CSV 的每行(字典形式)
    split_name: str,             # "train"/"dev"/"test"
    gpu_id: int,
    output_split_dir: str,       # 对应 train/ dev/ 或 test/ 的输出目录
    clip_model_path: str,
    dataset_root: str,
    pbar,
    pbar_lock,
    tmp_suffix: str = ""
):
    """
    单线程函数：在指定 GPU 上处理给定的 CSV 行数据。
    1) 逐条解析 -> 组装 text / label / 行号 -> id
    2) 用 CLIP 编码图像 & 文本 -> 保存 embedding -> 写到临时 JSON Lines 文件
    3) 每处理完1条 -> pbar.update(1)
    返回：该线程生成的临时 JSONL 文件路径 (如 train_gpu0_0.jsonl)
    """
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    c = 64
    a = 49

    # 载入 CLIP
    clip_model = CLIPModel.from_pretrained(clip_model_path).to(device)
    clip_model.eval()
    vision_model = clip_model.vision_model
    vision_projection = clip_model.visual_projection
    text_model = clip_model.text_model
    text_projection = clip_model.text_projection
    processor = AutoProcessor.from_pretrained(clip_model_path)
    tokenizer = AutoTokenizer.from_pretrained(clip_model_path)

    # 临时文件，比如 train_gpu0_0.jsonl
    tmp_jsonl_path = os.path.join(output_split_dir, f"{split_name}_gpu{gpu_id}{tmp_suffix}.jsonl")
    f_tmp = open(tmp_jsonl_path, "w", encoding="utf-8")

    for row_idx, row_dict in enumerate(lines, start=1):
        # 解析 CSV 行
        parsed = parse_csv_line(row_dict, row_idx)
        text = parsed["text"]
        label = parsed["label"]
        idx = parsed["id"]  # 1-based 行号

        # 拼出图像路径: {split_name}/images/{id}.jpg
        # dataset_root 是最外层目录; => /path/to/dataset/{split_name}/images/{id}.jpg
        img_rel_path = os.path.join(split_name, "images", f"{idx}.jpg")
        img_full_path = os.path.join(dataset_root, img_rel_path)

        # === 图像处理 ===
        if os.path.exists(img_full_path):
            image = Image.open(img_full_path).convert("RGB")
            image_inputs = processor(images=image, return_tensors="pt")
            for k in image_inputs:
                image_inputs[k] = image_inputs[k].to(device)
            with torch.no_grad():
                vision_out = vision_model(**image_inputs)
                vision_emb = vision_projection(vision_out.last_hidden_state)[0][1:]
                a, _ = vision_emb.shape
                vision_emb = average_pool_1d_divisible(vision_emb, c)
            vision_emb_np = vision_emb.cpu().numpy().astype(np.float16)
        else:
            # 若图片不存在，则用空阵占位
            vision_emb_np = np.zeros((a, c), dtype=np.float16)

        # === 文本处理 ===
        text_inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        for k in text_inputs:
            text_inputs[k] = text_inputs[k].to(device)
        with torch.no_grad():
            text_out = text_model(**text_inputs)
            text_emb = text_projection(text_out.last_hidden_state)[0]
            text_emb = average_pool_1d_divisible(text_emb, c)
        text_emb_np = text_emb.cpu().numpy().astype(np.float16)
        text_emb_np = text_upsample_or_truncate(
            text_emb_np, a
        )

        assert text_emb_np.shape == vision_emb_np.shape

        # === 保存 .npy ===
        # 以 id 为子目录 => e.g. train/1/...
        sample_dir = os.path.join(output_split_dir, str(idx))
        os.makedirs(sample_dir, exist_ok=True)
        img_emb_path = os.path.join(sample_dir, "image_emb.npy")
        txt_emb_path = os.path.join(sample_dir, "text_emb.npy")
        np.save(img_emb_path, vision_emb_np)
        np.save(txt_emb_path, text_emb_np)

        # === 写到临时 JSONL ===
        out_record = {
            "id": idx,
            "img": img_rel_path,  # eg. "train/images/1.jpg"
            "text": text,
            "label": label,
            "image_emb": os.path.relpath(img_emb_path, output_split_dir),
            "text_emb": os.path.relpath(txt_emb_path, output_split_dir)
        }
        f_tmp.write(json.dumps(out_record, ensure_ascii=False) + "\n")

        # === 更新进度条 ===
        with pbar_lock:
            pbar.update(1)

    f_tmp.close()
    return tmp_jsonl_path


def process_split_in_threads(
    split_csv: str,     # /path/to/dataset/train/train.csv
    split_name: str,    # "train"/"dev"/"test"
    dataset_root: str,  # /path/to/dataset
    output_base_dir: str,    # args.output_dir
    clip_model_path: str,
    gpu_ids: List[int]
):
    """
    读取 split_csv 文件，解析成行数据，然后多线程分 GPU 处理。
    处理完成后合并临时 JSONL => train.jsonl / dev.jsonl / test.jsonl
    并且输出目录为: output_base_dir/{split_name}/
    """
    if not os.path.isfile(split_csv):
        print(f"文件不存在: {split_csv}, 跳过。")
        return

    # 读取 CSV
    import csv
    lines = []
    with open(split_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lines.append(row)

    total = len(lines)
    if total == 0:
        print(f"[{split_csv}] 行数为 0, 跳过。")
        return

    print(f"[{split_csv}] 共有 {total} 条数据, 分配到 {len(gpu_ids)} 个线程.")

    # 在 output_base_dir 下，为当前 split 建立目录
    # e.g. output_base_dir/train
    output_split_dir = os.path.join(output_base_dir, split_name)
    os.makedirs(output_split_dir, exist_ok=True)

    # 均分
    num_gpus = len(gpu_ids)
    chunk_size = (total + num_gpus - 1) // num_gpus
    chunks = []
    start_idx = 0
    for i in range(num_gpus):
        end_idx = start_idx + chunk_size
        subset = lines[start_idx:end_idx]
        chunks.append(subset)
        start_idx = end_idx

    futures = []
    tmp_files = []

    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    with ThreadPoolExecutor(max_workers=num_gpus) as executor, \
         tqdm(total=total, desc=f"Processing {split_name}") as pbar:
        pbar_lock = Lock()
        for i, subset_lines in enumerate(chunks):
            if not subset_lines:
                continue
            gpu_id = gpu_ids[i]
            tmp_suffix = f"_{i}"
            future = executor.submit(
                process_csv_lines_on_gpu,
                lines=subset_lines,
                split_name=split_name,
                gpu_id=gpu_id,
                output_split_dir=output_split_dir,
                clip_model_path=clip_model_path,
                dataset_root=dataset_root,
                pbar=pbar,
                pbar_lock=pbar_lock,
                tmp_suffix=tmp_suffix
            )
            futures.append(future)

        for fut in as_completed(futures):
            tmp_files.append(fut.result())

    # 合并临时 JSONL
    final_jsonl_path = os.path.join(output_split_dir, f"{split_name}.jsonl")
    with open(final_jsonl_path, "w", encoding="utf-8") as fout:
        for tf in sorted(tmp_files):
            with open(tf, "r", encoding="utf-8") as fi:
                for line in fi:
                    fout.write(line)

    # 删除临时文件
    for tf in tmp_files:
        os.remove(tf)

    print(f"[{split_csv}] 处理完成, 结果 => {final_jsonl_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True,
        help="数据集根目录，其中有 train/, dev/, test/, 每个里面有 images/ 和 *.csv")
    parser.add_argument("--output_dir", required=True,
        help="输出目录，最终会在其中的 train/, dev/, test/ 目录存放结果")
    parser.add_argument("--gpu_ids", default="0",
        help="逗号分隔的 GPU ID 列表, 例如 '0,1'")
    parser.add_argument("--clip_model_path", default="openai/clip-vit-base-patch32",
        help="CLIP 模型名称或本地路径")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    gpu_id_list = [int(x) for x in args.gpu_ids.split(",")]

    # 分别处理 train / dev / test
    for split in ["train", "dev", "test"]:
        csv_path = os.path.join(args.dataset_root, split, f"{split}.csv")
        if os.path.isfile(csv_path):
            process_split_in_threads(
                split_csv=csv_path,
                split_name=split,
                dataset_root=args.dataset_root,
                output_base_dir=args.output_dir,
                clip_model_path=args.clip_model_path,
                gpu_ids=gpu_id_list
            )

    print("全部处理完成。")


if __name__ == "__main__":
    main()
