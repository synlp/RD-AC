import os
import json
import argparse
import numpy as np
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

##############################################################################
# 1) 分解函数: 共享低秩 (L) + 两个稀疏矩阵 (S_img, S_txt)
#    满足 M_img = L + S_img, M_txt = L + S_txt
##############################################################################
def shared_lmr_image_text(
    M_img: np.ndarray,
    M_txt: np.ndarray,
    lambda_param: float = 1e-3,
    mu: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-7,
    min_iter: int = 5,
    item_idx: int = None,
    total_items: int = None
):
    """
    对 M_img, M_txt 做“共享低秩 + 两个稀疏”分解:
      min  ||L||_* + lambda (||S_img||_1 + ||S_txt||_1)
      s.t. M_img = L + S_img
           M_txt = L + S_txt

    使用增广拉格朗日迭代. 返回 (L, S_img, S_txt).

    参数:
      M_img, M_txt: 形状相同 (m, n)
      lambda_param: 稀疏正则系数
      mu: 罚因子
      max_iter, tol, min_iter: 迭代控制
      item_idx, total_items: 用于进度条后缀显示 "item: x/y"
    """
    assert M_img.shape == M_txt.shape, "image_emb 与 text_emb 形状必须一致"
    m, n = M_img.shape

    # 初始化
    L     = np.zeros((m, n), dtype=np.float64)
    S_img = np.zeros((m, n), dtype=np.float64)
    S_txt = np.zeros((m, n), dtype=np.float64)
    Z_img = np.zeros((m, n), dtype=np.float64)
    Z_txt = np.zeros((m, n), dtype=np.float64)

    def soft_threshold(X, th):
        return np.sign(X) * np.maximum(np.abs(X) - th, 0)

    pbar = tqdm(range(max_iter), desc="S-LMR", leave=False)
    for it in pbar:
        # (1) 更新 S_img, S_txt
        S_img = soft_threshold(M_img - L + (1.0/mu)*Z_img, lambda_param/mu)
        S_txt = soft_threshold(M_txt - L + (1.0/mu)*Z_txt, lambda_param/mu)

        # (2) 保存旧的 L 用于收敛度量
        L_old = L.copy()

        # (3) 更新 L
        #     A = 0.5 * [ (M_img - S_img + Z_img/mu) + (M_txt - S_txt + Z_txt/mu) ]
        A = ( (M_img - S_img + (1.0/mu)*Z_img) +
              (M_txt - S_txt + (1.0/mu)*Z_txt) ) / 2.0
        try:
            U, sigma, VT = np.linalg.svd(A, full_matrices=False)
        except np.linalg.LinAlgError:
            # 若 SVD 不收敛, 则加点小扰动或直接 fallback
            # 这里演示加扰动
            A += 1e-12 * np.random.randn(*A.shape)
            U, sigma, VT = np.linalg.svd(A, full_matrices=False)

        sigma_thresh = np.maximum(sigma - 1.0/mu, 0)
        L_new = (U * sigma_thresh) @ VT

        # (4) 更新 Z_img, Z_txt
        R_img = M_img - L_new - S_img
        R_txt = M_txt - L_new - S_txt
        Z_img += mu * R_img
        Z_txt += mu * R_txt

        # (5) 收敛度
        denom = max(1e-8, np.linalg.norm(L_old, 'fro'))
        rel_change = np.linalg.norm(L_new - L_old, 'fro') / denom

        # 更新 L
        L = L_new

        # 进度条
        postfix_dict = {"rel_change": f"{rel_change:.2e}"}
        if item_idx is not None and total_items is not None:
            postfix_dict["item"] = f"{item_idx+1}/{total_items}"
        pbar.set_postfix(postfix_dict)

        # (6) 收敛判定
        if it >= min_iter and rel_change < tol:
            break

    pbar.close()
    return L, S_img, S_txt


##############################################################################
# 2) 子进程处理函数: 读取 JSON 记录 => 加载 image_emb, text_emb => 分解 => 写回
##############################################################################
def process_record(
    record: Dict[str, Any],
    split_dir: str,
    lambda_param: float,
    mu: float,
    max_iter: int,
    tol: float,
    min_iter: int,
    item_idx: int,
    total_items: int
) -> Dict[str, Any]:
    """
    处理单条记录:
      - 从 record["image_emb"] / record["text_emb"] 获取 npy 路径(相对 split_dir).
      - 加载后调用 shared_lmr_image_text
      - 若中途出错, fallback => L=0, S_img=M_img, S_txt=M_txt
      - 将 L, S_img, S_txt 保存到 split_dir/<id> 目录
      - 把这些路径写回 record["L_path"], record["S_image_path"], record["S_text_path"]
    """
    rec_id = record.get("id")
    if rec_id is None:
        return record

    out_subdir = os.path.join(split_dir, str(rec_id))
    os.makedirs(out_subdir, exist_ok=True)

    # 目标文件
    L_path  = os.path.join(out_subdir, "L.npy")
    S_img_path = os.path.join(out_subdir, "S_image.npy")
    S_txt_path = os.path.join(out_subdir, "S_text.npy")

    # 若已存在 L.npy, 说明处理过
    # if os.path.exists(L_path) and os.path.exists(S_img_path) and os.path.exists(S_txt_path):
    #     return record  # 已处理过

    image_emb_rel = record.get("image_emb")
    text_emb_rel  = record.get("text_emb")
    # 拼绝对路径
    M_img_path = os.path.join(split_dir, image_emb_rel)
    M_txt_path = os.path.join(split_dir, text_emb_rel)

    try:
        # 加载
        if not (os.path.isfile(M_img_path) and os.path.isfile(M_txt_path)):
            raise FileNotFoundError("image_emb 或 text_emb 文件不存在")

        M_img = np.load(M_img_path)
        M_txt = np.load(M_txt_path)

        # 分解
        L, S_img, S_txt = shared_lmr_image_text(
            M_img, M_txt,
            lambda_param=lambda_param,
            mu=mu,
            max_iter=max_iter,
            tol=tol,
            min_iter=min_iter,
            item_idx=item_idx,
            total_items=total_items
        )
    except Exception as e:
        # fallback
        print(f"[Fallback] record id={rec_id}, error={e}")
        try:
            M_img
        except NameError:
            M_img = np.zeros((1,1), dtype=np.float64)
        try:
            M_txt
        except NameError:
            M_txt = np.zeros((1,1), dtype=np.float64)

        L     = np.zeros_like(M_img)
        S_img = M_img
        S_txt = M_txt

    # 保存
    np.save(L_path,     L)
    np.save(S_img_path, S_img)
    np.save(S_txt_path, S_txt)

    # 更新 record
    record["L_path"]       = os.path.relpath(L_path, split_dir)
    record["S_image_path"] = os.path.relpath(S_img_path, split_dir)
    record["S_text_path"]  = os.path.relpath(S_txt_path, split_dir)

    return record


def _worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """ 多进程辅助函数，解包后调用 process_record """
    return process_record(
        record       = task["record"],
        split_dir    = task["split_dir"],
        lambda_param = task["lambda_param"],
        mu           = task["mu"],
        max_iter     = task["max_iter"],
        tol          = task["tol"],
        min_iter     = task["min_iter"],
        item_idx     = task["item_idx"],
        total_items  = task["total_items"]
    )

##############################################################################
# 3) 处理单个 split (train_emb / dev_emb / test_emb)
##############################################################################
def process_split(
    split_name: str,   # "train" / "dev" / "test"
    split_dir: str,    # "train_emb" / "dev_emb" / "test_emb"
    num_workers: int,
    lambda_param: float,
    mu: float,
    max_iter: int,
    tol: float,
    min_iter: int
) -> List[Dict[str,Any]]:
    """
    读取 split_dir 下面的 {split_name}_out.jsonl, 对每条记录做 shared_lmr_image_text.
    返回处理后的记录列表.
    """
    jsonl_path = os.path.join(split_dir, f"{split_name}_out.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"[{split_name}] {jsonl_path} 不存在, 跳过.")
        return []

    # 读取 jsonl
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if split_name == "train":
        lines = lines[:1000]
    data = [json.loads(line.strip()) for line in lines]
    total = len(data)
    print(f"[{split_name}] {jsonl_path} 共 {total} 条记录, 准备 {num_workers} workers 处理.")

    if total == 0:
        return []

    tasks = []
    for i, record in enumerate(data):
        tasks.append({
            "record": record,
            "split_dir": split_dir,
            "lambda_param": lambda_param,
            "mu": mu,
            "max_iter": max_iter,
            "tol": tol,
            "min_iter": min_iter,
            "item_idx": i,
            "total_items": total
        })

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for ret in tqdm(executor.map(_worker, tasks), total=total, desc=f"{split_name}"):
            results.append(ret)

    print(f"[{split_name}] 处理完成, {len(results)} 条.")
    return results


##############################################################################
# 4) 主函数: 分别对 train_emb, dev_emb, test_emb 做处理
##############################################################################
def main():
    parser = argparse.ArgumentParser(description="对 train_emb/dev_emb/test_emb 做共享低秩分解(L, S_image, S_text)")
    parser.add_argument("--data_dir", required=True, help="包含 train_emb, dev_emb, test_emb 的上级目录")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lambda_param", type=float, default=1e-3)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--min_iter", type=int, default=5)
    parser.add_argument("--output_json", default="combined_result.json",
                        help="输出 JSON 文件名, 存放于 data_dir 下")
    args = parser.parse_args()

    # 这里我们把 train_emb => split_name="train"
    #             dev_emb   => split_name="dev"
    #             test_emb  => split_name="test"
    # 你也可根据实际情况修改命名
    splits = [
        ("train", os.path.join(args.data_dir, "train_emb")),
        # ("dev",   os.path.join(args.data_dir, "dev_emb")),
        ("test",  os.path.join(args.data_dir, "test_emb"))
    ]

    # final_dict = {"train": [], "dev": [], "test": []}
    final_dict = {"train": [], "test": []}
    for split_name, split_dir in splits:
        if not os.path.isdir(split_dir):
            print(f"[warn] {split_dir} 不存在, 跳过.")
            continue

        res = process_split(
            split_name=split_name,
            split_dir=split_dir,
            num_workers=args.num_workers,
            lambda_param=args.lambda_param,
            mu=args.mu,
            max_iter=args.max_iter,
            tol=args.tol,
            min_iter=args.min_iter
        )
        final_dict[split_name] = res

    out_path = os.path.join(args.data_dir, args.output_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_dict, f, ensure_ascii=False, indent=2)

    print(f"全部处理完成，结果写到: {out_path}")

if __name__ == "__main__":
    main()
