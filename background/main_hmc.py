import os
import json
import argparse
import numpy as np
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

##############################################################################
# 1) 分解函数: 共享低秩 (L) + 两个稀疏矩阵 (S1, S2)
#    满足 M1 = L + S1, M2 = L + S2
##############################################################################
def shared_lmr_two_matrices(
    M1: np.ndarray,
    M2: np.ndarray,
    lambda_param: float = 1e-3,
    mu: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-7,
    min_iter: int = 5,
    item_idx: int = None,
    total_items: int = None
):
    """
    对 M1, M2 做“共享低秩 + 两个稀疏”分解:
      min  ||L||_* + lambda (||S1||_1 + ||S2||_1)
      s.t. M1 = L + S1
           M2 = L + S2

    使用增广拉格朗日迭代 (Z1, Z2为乘子). 返回 (L, S1, S2).

    参数:
      M1, M2: 形状相同 (m, n)
      lambda_param: 稀疏正则系数
      mu: 罚因子
      max_iter: 最大迭代次数
      tol: 收敛阈值(对 L 的相对变化)
      min_iter: 最少迭代次数
      item_idx, total_items: 用于进度条后缀显示 "item: x/y"
    """
    assert M1.shape == M2.shape, "M1, M2 must have same shape"
    m, n = M1.shape

    L  = np.zeros((m, n), dtype=np.float64)
    S1 = np.zeros((m, n), dtype=np.float64)
    S2 = np.zeros((m, n), dtype=np.float64)
    Z1 = np.zeros((m, n), dtype=np.float64)
    Z2 = np.zeros((m, n), dtype=np.float64)

    def soft_threshold(X, th):
        return np.sign(X) * np.maximum(np.abs(X) - th, 0)

    pbar = tqdm(range(max_iter), desc="S-LMR", leave=False)
    for it in pbar:
        # 1) 更新 S1, S2
        S1 = soft_threshold(M1 - L + (1.0/mu)*Z1, lambda_param/mu)
        S2 = soft_threshold(M2 - L + (1.0/mu)*Z2, lambda_param/mu)

        # 2) 记录 L 旧值
        L_old = L.copy()

        # 3) 更新 L
        #    A = 0.5 * [ (M1 - S1 + Z1/mu) + (M2 - S2 + Z2/mu) ]
        A = ((M1 - S1 + (1.0/mu)*Z1) + (M2 - S2 + (1.0/mu)*Z2)) / 2.0
        U, sigma, VT = np.linalg.svd(A, full_matrices=False)
        sigma_thresh = np.maximum(sigma - 1.0/mu, 0)
        L_new = (U * sigma_thresh) @ VT

        # 4) 更新 Z1, Z2
        R1 = M1 - L_new - S1
        R2 = M2 - L_new - S2
        Z1 += mu * R1
        Z2 += mu * R2

        # 5) 收敛度
        denom = max(1e-8, np.linalg.norm(L_old, 'fro'))
        rel_change = np.linalg.norm(L_new - L_old, 'fro') / denom

        # 更新 L
        L = L_new

        # 进度条显示
        postfix_dict = {"rel_change": f"{rel_change:.2e}"}
        if item_idx is not None and total_items is not None:
            postfix_dict["item"] = f"{item_idx+1}/{total_items}"
        pbar.set_postfix(postfix_dict)

        if it >= min_iter and rel_change < tol:
            break

    pbar.close()
    return L, S1, S2

##############################################################################
# 2) 子进程处理函数: 每条记录(对应某个 ID 目录) 做分解
##############################################################################
def process_record(
    record: Dict[str, Any],
    input_dir: str,
    lambda_param: float,
    mu: float,
    max_iter: int,
    tol: float,
    min_iter: int,
    item_idx: int,
    total_items: int
) -> Dict[str, Any]:
    """
    对单条记录进行分解:
      1) 解析 record, 找到 M1, M2 的 .npy 路径(相对 input_dir).
      2) 加载 M1, M2.
      3) 调用 shared_lmr_two_matrices(M1, M2) => L, S1, S2
      4) 如出错, fallback: L=0, S1=M1, S2=M2
      5) 将 L,S1,S2 保存到 input_dir/{id}/ 目录, 并更新 record
    """


    rec_id = record.get("id")
    if not rec_id:
        return record

    # 准备输出子目录 => input_dir/{id}
    out_subdir = os.path.join(input_dir, str(rec_id))
    os.makedirs(out_subdir, exist_ok=True)

    # 目标文件
    L_path  = os.path.join(out_subdir, "L.npy")
    S1_path = os.path.join(out_subdir, "S1.npy")
    S2_path = os.path.join(out_subdir, "S2.npy")

    # 如果 L.npy 已存在, 表示已处理过
    # if os.path.exists(L_path) and os.path.exists(S1_path) and os.path.exists(S2_path):
    #     return record  # 已处理过，直接返回


    # 尝试从 record 中取出 image_last_hidden_state / text_last_hidden_state
    mat1_rel = record.get("image_last_hidden_state")
    mat2_rel = record.get("text_last_hidden_state")
    # 拼绝对路径
    mat1_path = os.path.join(input_dir, mat1_rel)
    mat2_path = os.path.join(input_dir, mat2_rel)

    try:
        # ============ 尝试加载 M1, M2 并执行分解 ============
        if (not os.path.isfile(mat1_path)) or (not os.path.isfile(mat2_path)):
            raise FileNotFoundError("某个矩阵文件不存在")

        M1 = np.load(mat1_path)
        M2 = np.load(mat2_path)

        # 调用共享低秩 + 两个稀疏分解
        L, S1, S2 = shared_lmr_two_matrices(
            M1, M2,
            lambda_param=lambda_param,
            mu=mu,
            max_iter=max_iter,
            tol=tol,
            min_iter=min_iter,
            item_idx=item_idx,
            total_items=total_items
        )

    except Exception as e:
        # ============ 发生错误, fallback 逻辑 ============
        # 1) 如果无法正常分解, 我们就把 L 做成全0矩阵, S1=M1, S2=M2
        # 2) 同样写到 L.npy, S1.npy, S2.npy 里
        # 3) 标记 record["fallback"]=True 方便追踪

        # 若无法加载 M1,M2, 就都做成空;
        # 否则, 若只是在分解过程中出错, 我们还能用已加载的 M1,M2
        try:
            M1  # 看能不能访问, 若本次也报错说明没加载成功
        except NameError:
            # 没加载成功, 给个空 0 矩阵
            M1 = np.zeros((1,1), dtype=np.float64)
        try:
            M2
        except NameError:
            M2 = np.zeros((1,1), dtype=np.float64)

        L  = np.zeros_like(M1)
        S1 = M1
        S2 = M2

    # ============ 无论是否 fallback, 都把 L,S1,S2 保存 ============
    np.save(L_path,  L)
    np.save(S1_path, S1)
    np.save(S2_path, S2)

    # 在 record 里更新信息
    record["L_path"]  = os.path.relpath(L_path, input_dir)
    record["S1_path"] = os.path.relpath(S1_path, input_dir)
    record["S2_path"] = os.path.relpath(S2_path, input_dir)

    return record


##############################################################################
# 3) 多进程 worker
##############################################################################
def _worker(task: Dict[str, Any]) -> Dict[str, Any]:
    # 反解 task
    return process_record(
        record       = task["record"],
        input_dir    = task["input_dir"],
        lambda_param = task["lambda_param"],
        mu           = task["mu"],
        max_iter     = task["max_iter"],
        tol          = task["tol"],
        min_iter     = task["min_iter"],
        item_idx     = task["item_idx"],
        total_items  = task["total_items"]
    )

##############################################################################
# 4) 主函数: 分别处理 train_out.jsonl / dev_out.jsonl / test_out.jsonl
#            最后合并 => {"train": [...], "dev": [...], "test": [...]}
##############################################################################
def process_split(
    split: str,
    input_dir: str,
    num_workers: int,
    lambda_param: float,
    mu: float,
    max_iter: int,
    tol: float,
    min_iter: int
) -> List[Dict[str,Any]]:
    """
    处理某个 split (train/dev/test) 的 out.jsonl, 并返回处理后的 record 列表
    """
    jsonl_path = os.path.join(input_dir, f"{split}_out.jsonl")
    if not os.path.isfile(jsonl_path):
        print(f"[{split}] {jsonl_path} 不存在, 跳过.")
        return []

    # 加载
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if split == "train":
        lines = lines[:1000]

    data = [json.loads(line.strip()) for line in lines]
    total = len(data)
    print(f"[{split}] {jsonl_path} 有 {total} 条记录, 准备启动 {num_workers} 进程.")

    if total == 0:
        return []

    # 组装 tasks
    tasks = []
    for idx, record in enumerate(data):
        tasks.append({
            "record": record,
            "input_dir": input_dir,
            "lambda_param": lambda_param,
            "mu": mu,
            "max_iter": max_iter,
            "tol": tol,
            "min_iter": min_iter,
            "item_idx": idx,
            "total_items": total
        })

    # 多进程
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 外层进度条
        for res in tqdm(executor.map(_worker, tasks), total=total, desc=f"Processing {split}"):
            results.append(res)

    print(f"[{split}] 处理完成, 共 {len(results)} 条.")
    return results

def main():
    parser = argparse.ArgumentParser(description="对 train_out.jsonl, dev_out.jsonl, test_out.jsonl 做三矩阵分解(L, S1, S2)")
    parser.add_argument("--input_dir", required=True, help="输入目录, 包含 train_out.jsonl, dev_out.jsonl, test_out.jsonl, 以及数字ID子目录")
    parser.add_argument("--num_workers", type=int, default=4, help="多进程数")
    parser.add_argument("--lambda_param", type=float, default=1e-3, help="稀疏正则")
    parser.add_argument("--mu", type=float, default=1.0, help="增广拉格朗日的罚因子")
    parser.add_argument("--max_iter", type=int, default=500, help="最大迭代次数")
    parser.add_argument("--tol", type=float, default=1e-7, help="收敛阈值")
    parser.add_argument("--min_iter", type=int, default=5, help="最少迭代次数")
    parser.add_argument("--output_json", default="low_rank_result.json",
                        help="最终合并成 {train: [...], dev: [...], test: [...]} 写到 input_dir 下的此文件")
    args = parser.parse_args()

    # 分别处理 train / dev / test
    final_result = {"train": [], "dev": []}
    for split in ["train", "dev"]:
        res = process_split(
            split=split,
            input_dir=args.input_dir,
            num_workers=args.num_workers,
            lambda_param=args.lambda_param,
            mu=args.mu,
            max_iter=args.max_iter,
            tol=args.tol,
            min_iter=args.min_iter
        )
        final_result[split] = res

    # 合并写一个 JSON
    out_path = os.path.join(args.input_dir, args.output_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    print(f"全部处理完成, 结果写到: {out_path}")

if __name__ == "__main__":
    main()
