import os
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

LABEL_MAP = {
    "0": "negative",
    "1": "neutral",
    "2": "positive"
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


def parse_tsv_line(line: str) -> Dict[str, Any]:
    """
    假设列顺序: index \t #1 Label \t #2 ImageID \t #3 String \t #3 String
    例如:
      1    2    1860693.jpg    RT @ ltsChuckBass : $T$ is everything # MCM    Chuck Bass
    返回一个 dict, 包括:
      "idx": str,  # index
      "label": str("0"|"1"|"2"),
      "img": str(文件名),
      "strA": 第一个 #3 String
      "strB": 第二个 #3 String
    如果行格式不对，返回 None
    """
    fields = line.strip().split("\t")
    if len(fields) < 5:
        return None
    idx_str, label_str, img_str, textA, textB = fields[:5]
    return {
        "idx": idx_str,
        "label": label_str,
        "img": img_str,
        "strA": textA,
        "strB": textB
    }

def combine_text(textA: str, textB: str) -> str:
    """
    将 textA 中的 $T$ 替换为 <t>textB</t>.
    若有多个 $T$, 可以改成 replace("$T$", ...) 做全量替换。
    这里示例只替换第一个。
    """
    if "$T$" in textA:
        return textA.replace("$T$", f"<t> {textB} </t>", 1)
    else:
        return textA

def process_lines_on_gpu(
    lines: List[str],
    gpu_id: int,
    output_dir: str,
    output_prefix: str,
    img_home_dir: str,
    clip_model_path: str,
    pbar,
    pbar_lock,
    tmp_file_suffix: str = ""
):
    """
    单线程函数：在指定 GPU 上处理给定行的 TSV 数据。
    逐条解析 -> 得到 id, label, text, img -> 用 CLIP 编码 -> 保存embedding -> 写到临时 JSONL。
    每处理完1条就对 pbar.update(1)。
    返回: 生成的 tmp_jsonl_path
    """
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    # 加载模型到该 GPU
    clip_model = CLIPModel.from_pretrained(clip_model_path).to(device)
    clip_model.eval()
    vision_model = clip_model.vision_model
    vision_projection = clip_model.visual_projection
    text_model = clip_model.text_model
    text_projection = clip_model.text_projection
    processor = AutoProcessor.from_pretrained(clip_model_path)
    tokenizer = AutoTokenizer.from_pretrained(clip_model_path)

    # 临时文件，比如 train_gpu0_0.jsonl, train_gpu0_1.jsonl ...
    tmp_jsonl_path = os.path.join(
        output_dir, f"{output_prefix}_gpu{gpu_id}{tmp_file_suffix}.jsonl"
    )
    f_tmp = open(tmp_jsonl_path, "w", encoding="utf-8")
    c = 256
    # a = 49

    for line in lines:
        record = parse_tsv_line(line)
        if not record:
            continue

        idx = record["idx"]
        label_raw = record["label"]
        label_str = LABEL_MAP.get(label_raw, "unknown")

        textA = record["strA"]
        textB = record["strB"]
        combined_text = combine_text(textA, textB)

        # 1) 图像处理
        # 根据 img_home_dir 拼接图像路径
        img_path = os.path.join(img_home_dir, record["img"])
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
            image_inputs = processor(images=image, return_tensors="pt")
            for k in image_inputs:
                image_inputs[k] = image_inputs[k].to(device)
            with torch.no_grad():
                vision_out = vision_model(**image_inputs)
                vision_emb = vision_projection(vision_out.last_hidden_state)[0][1:]
                a, _ = vision_emb.shape
                vision_emb = average_pool_1d_divisible(vision_emb, c)
            vision_emb_np = vision_emb.cpu().numpy().astype(np.float16)  # shape (seq_len, hidden_dim)
        else:
            # 如果图像不存在，用占位
            vision_emb_np = np.zeros((a, c), dtype=np.float16)

        # 2) 文本处理
        text_inputs = tokenizer([combined_text], return_tensors="pt", padding=True, truncation=True)
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

        # 3) 保存 npy
        # 以 id 为子目录
        sample_dir = os.path.join(output_dir, str(idx))
        os.makedirs(sample_dir, exist_ok=True)
        img_emb_path = os.path.join(sample_dir, "image_emb.npy")
        txt_emb_path = os.path.join(sample_dir, "text_emb.npy")
        np.save(img_emb_path, vision_emb_np)
        np.save(txt_emb_path, text_emb_np)

        # 4) 写 JSONL
        out_record = {
            "id": int(idx),
            "label": label_str,
            "img": record["img"],     # 仅存相对文件名
            "text": combined_text,
            "image_emb": os.path.relpath(img_emb_path, output_dir),
            "text_emb": os.path.relpath(txt_emb_path, output_dir)
        }
        f_tmp.write(json.dumps(out_record, ensure_ascii=False) + "\n")

        # === 每处理一条就更新进度条 ===
        with pbar_lock:
            pbar.update(1)

    f_tmp.close()
    return tmp_jsonl_path

def process_tsv_in_threads(
    tsv_path: str,
    output_dir: str,
    output_prefix: str,
    img_home_dir: str,
    clip_model_path: str,
    gpu_ids: List[int]
):
    """
    读取 tsv_path, 将行数据均分到 len(gpu_ids) 个线程，每个线程绑定一个 GPU 处理。
    最后合并临时文件 -> 输出 e.g. {output_prefix}_out.jsonl
    """
    if not os.path.isfile(tsv_path):
        print(f"文件不存在: {tsv_path}, 跳过处理。")
        return

    with open(tsv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]

    total = len(lines)
    if total == 0:
        print(f"文件 {tsv_path} 行数为 0, 跳过。")
        return

    print(f"[{tsv_path}] 共有 {total} 行数据，准备分配给 {len(gpu_ids)} 个线程进行处理")

    os.makedirs(output_dir, exist_ok=True)

    # 均分行数据
    num_gpus = len(gpu_ids)
    chunk_size = (total + num_gpus - 1) // num_gpus
    chunks = []
    start = 0
    for i in range(num_gpus):
        end = start + chunk_size
        subset = lines[start:end]
        chunks.append(subset)
        start = end

    tmp_files = []
    futures = []

    with ThreadPoolExecutor(max_workers=num_gpus) as executor, \
         tqdm(total=total, desc=f"Processing {output_prefix}") as pbar:
        pbar_lock = Lock()  # 保护进度条

        for i, subset_lines in enumerate(chunks):
            if len(subset_lines) == 0:
                continue
            gpu_id = gpu_ids[i]
            # 为避免多个线程的临时文件重名，可在文件名加个 suffix
            suffix = f"_{i}"
            fut = executor.submit(
                process_lines_on_gpu,
                lines=subset_lines,
                gpu_id=gpu_id,
                output_dir=output_dir,
                output_prefix=output_prefix,
                img_home_dir=img_home_dir,
                clip_model_path=clip_model_path,
                pbar=pbar,
                pbar_lock=pbar_lock,
                tmp_file_suffix=suffix
            )
            futures.append(fut)

        for fut in as_completed(futures):
            tmp_jsonl = fut.result()
            tmp_files.append(tmp_jsonl)

    # 合并临时文件
    final_jsonl = os.path.join(output_dir, f"{output_prefix}_out.jsonl")
    with open(final_jsonl, "w", encoding="utf-8") as fout:
        for tf in sorted(tmp_files):
            with open(tf, "r", encoding="utf-8") as fi:
                for line in fi:
                    fout.write(line)

    # 清理临时文件
    for tf in tmp_files:
        os.remove(tf)

    print(f"[{tsv_path}] 处理完成, 结果写到 -> {final_jsonl}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="包含 train.tsv / dev.tsv / test.tsv 的目录")
    parser.add_argument("--img_home_dir", required=True,
                        help="输入图像的 home 目录，用于拼接 #2 ImageID")
    parser.add_argument("--output_dir", required=True,
                        help="输出目录, 保存 embedding npy 和合并后的 jsonl")
    parser.add_argument("--gpu_ids", default="0",
                        help="逗号分隔的 GPU ID 列表，如 '0,1,2,3'")
    parser.add_argument("--clip_model_path", default="openai/clip-vit-base-patch32",
                        help="CLIP 模型名称或路径")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    gpu_id_list = [int(x) for x in args.gpu_ids.split(",")]

    # 分别处理 train / dev / test
    for split in ["train", "dev", "test"]:
        tsv_file = os.path.join(args.data_dir, f"{split}.tsv")
        if not os.path.isfile(tsv_file):
            continue
        # 对每个 split，新建一个子目录来存 embedding
        split_outdir = os.path.join(args.output_dir, f"{split}_emb")
        process_tsv_in_threads(
            tsv_path=tsv_file,
            output_dir=split_outdir,
            output_prefix=split,
            img_home_dir=args.img_home_dir,
            clip_model_path=args.clip_model_path,
            gpu_ids=gpu_id_list
        )

    print("全部处理完成。")

if __name__ == "__main__":
    main()
