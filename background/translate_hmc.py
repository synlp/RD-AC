import json
import os


def translate_labels(json_path: str, output_path: str = None):
    """
    读取指定 JSON 文件，对 `train` 和 `dev` 列表中的 label 做翻译 (0->not-hateful, 1->hateful)，
    并将处理结果写入新的 JSON 文件中。

    :param json_path: 输入 JSON 文件路径
    :param output_path: 输出 JSON 文件路径 (默认为 None，自动在原文件名后加 `_translated` )
    """
    # 1. 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 定义翻译规则函数
    def label_to_str(label_val):
        if label_val == 0:
            return "not-hateful"
        elif label_val == 1:
            return "hateful"
        else:
            # 如果不在 [0, 1]，可按需求处理，这里简单返回原数值
            return label_val

    # 3. 翻译 `train` 和 `dev` 中的 label
    #   注意：假设原始 JSON 中有 "train"、"dev"、"test" 三个 key, 我们只处理 train/dev。
    if "train" in data:
        for item in data["train"]:
            if "label" in item:
                item["label"] = label_to_str(item["label"])

    if "dev" in data:
        for item in data["dev"]:
            if "label" in item:
                item["label"] = label_to_str(item["label"])

    # 4. 生成输出文件路径
    if output_path is None:
        base, ext = os.path.splitext(json_path)
        output_path = f"{base}_translated{ext}"

    # 5. 写回新的 JSON 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"已处理完毕，输出文件: {output_path}")


if __name__ == "__main__":
    # 示例：假设原始文件为 data.json
    input_file = "../../processed_data/hmc/all_data.json"
    output_file = input_file
    translate_labels(input_file, output_file)
