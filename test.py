from transformers import AutoProcessor
from transformers import AutoTokenizer, AutoProcessor
from model.qwen_vl_model import Qwen2VLForSequenceClassification
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import random
from PIL import Image, UnidentifiedImageError
import torch
import numpy as np
import json
import argparse
import os
import re
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize


random.seed(42)


def unify_example_hmc(example):
    """
    原HMC示例:
    {
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
    unified = {
        "id": example["id"],
        "img": example["img"],
        "text": example["text"] + '\nIs the multimodal content hateful or not-hateful?\n',
        "label": example["label"],
        "image_emb": example.get("image_last_hidden_state", ""),  # 或者留空字符串
        "text_emb": example.get("text_last_hidden_state", ""),
        "L_path": example["L_path"],
        "S_image_path": example["S1_path"],  # HMC里 S1_path 对应 S_image
        "S_text_path": example["S2_path"],  # HMC里 S2_path 对应 S_text
    }
    return unified


def unify_example_twitter(example):
    """
    原Twitter示例:
    {
      "id": 1,
      "label": "positive",
      "img": "1860693.jpg",
      "text": "...",
      "image_emb": "1/image_emb.npy",
      "text_emb": "1/text_emb.npy",
      "L_path": "1/L.npy",
      "S_image_path": "1/S_image.npy",
      "S_text_path": "1/S_text.npy"
    }
    """
    unified = {
        "id": example["id"],
        "img": example["img"],
        "text": example["text"] + "\nWhat is the sentiment of the multimodal content positive, negative, or neutral?\n",
        "label": example["label"],
        "image_emb": example.get("image_emb", ""),
        "text_emb": example.get("text_emb", ""),
        "L_path": example["L_path"],
        "S_image_path": example["S_image_path"],
        "S_text_path": example["S_text_path"],
    }
    return unified


def unify_example_msed(example):
    """
    原MSDE示例:
    {
      "id": 1,
      "img": "train/images/1.jpg",
      "text": "Overweight man and woman jogging...",
      "label": "neutral",
      "image_emb": "1/image_emb.npy",
      "text_emb": "1/text_emb.npy",
      "L_path": "1/L.npy",
      "S_image_path": "1/S_image.npy",
      "S_text_path": "1/S_text.npy"
    }
    """
    unified = {
        "id": example["id"],
        "img": example["img"],
        "text": example["text"] + "\nWhat is the emotion of the multimodal content?\n",
        "label": example["label"],
        "image_emb": example.get("image_emb", ""),
        "text_emb": example.get("text_emb", ""),
        "L_path": example.get("L_path", None),
        "S_image_path": example.get("S_image_path", None),
        "S_text_path": example.get("S_text_path", None),
    }
    return unified


def unify_dataset(input_json_path, data_name):
    """
    将一个JSON文件里 (可能包含 "train"/"test"/"dev" 等分割) 的数据，统一转换。
    """
    with open(input_json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if data_name == "hmc":
        unify_fn = unify_example_hmc
        raw_data = raw_data['dev']  # or 'test' if needed
        label_list = ['not-hateful', 'hateful']
    elif data_name == "twitter":
        unify_fn = unify_example_twitter
        raw_data = raw_data['test']
        label_list = ['positive', 'negative', 'neutral']
    elif data_name == "msed":
        unify_fn = unify_example_msed
        raw_data = raw_data['test']
        label_list = ['fear', 'disgust', 'neutral', 'anger', 'happiness', 'sad']
    else:
        raise ValueError("data_name must be one of [hmc, twitter, msed].")

    unified_list = []
    for ex in raw_data:
        unified_ex = unify_fn(ex)
        unified_list.append(unified_ex)
    return unified_list, label_list


def process_json(args):
    # Load model
    model = Qwen2VLForSequenceClassification.from_pretrained(
        args.model_path, device_map="auto"
    )
    print('pad_length: ', model.pad_length)
    model.half()  # use FP16

    processor = AutoProcessor.from_pretrained(args.model_path)
    all_data, _ = unify_dataset(args.input_json, args.data_name)
    # label2id = model.config.label2id
    label2id = {
        "hateful": 1,
        "not-hateful": 0
    }
    id2label = {v: k for k, v in label2id.items()}
    print(f'label2id: {label2id}')
    print(f'id2label: {id2label}')

    # Directories
    image_dir = args.image_dir
    lmr_dir = args.lmr_dir

    all_gold = []
    all_pred = []
    all_prob = []  # Store predicted probabilities for AUROC
    all_output = []

    for example in tqdm(all_data):
        img_path = os.path.join(image_dir, example["img"])
        image_pil = Image.open(img_path).convert("RGB")
        fixed_size = (224, 224)
        image_pil = image_pil.resize(fixed_size, Image.BICUBIC)

        # Read L, S_image, S_text
        try:
            l_path = os.path.join(lmr_dir, example["L_path"])
            s_img_path = os.path.join(lmr_dir, example["S_image_path"])
            s_txt_path = os.path.join(lmr_dir, example["S_text_path"])

            l_matrix = torch.from_numpy(np.load(l_path)).half()
            s_image_matrix = torch.from_numpy(np.load(s_img_path)).half()
            s_text_matrix = torch.from_numpy(np.load(s_txt_path)).half()
        except Exception:
            # Fallback if no valid matrix found
            l_matrix = torch.zeros((49, 64)).half()
            s_image_matrix = torch.zeros((49, 64)).half()
            s_text_matrix = torch.zeros((49, 64)).half()

        # Prepare conversation
        user_text = example["text"]
        label_text = example["label"]  # gold label

        input_info = {
            'text': user_text,
            'img': example["img"]
        }
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

        # Tokenization via Qwen's processor
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
        # Convert to python list to manually pack them
        inputs = {k: v.tolist() for k, v in inputs.items()}

        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        # grid for images
        gthw = torch.tensor(inputs["image_grid_thw"])
        if gthw.shape == (1, 3):
            gthw = gthw.squeeze(0)

        # Move all to CUDA
        device = torch.device("cuda")
        inputs = {
            "input_ids": torch.tensor([input_ids]).to(device),
            "attention_mask": torch.tensor([attention_mask]).to(device),
            "pixel_values": torch.tensor([inputs["pixel_values"]]).to(device),
            "image_grid_thw": torch.stack([gthw], dim=0).to(device),
            "l_matrix": torch.stack([l_matrix]).to(device),
            "s_v_matrix": torch.stack([s_image_matrix]).to(device),
            "s_t_matrix": torch.stack([s_text_matrix]).to(device),
        }

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # shape (1, num_labels)

        # Convert logits to predicted label
        pred_idx = logits.argmax(dim=-1).item()
        pred_label_text = id2label[pred_idx]

        # Also get probability distribution (softmax) for computing AUROC
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        all_prob.append(probs)

        all_gold.append(label_text)
        all_pred.append(pred_label_text)

        print(f'pred: {pred_label_text}')
        print(f'gold: {label_text}')
        print(f'prob: {probs}')

        instance = {
            'input': input_info,
            'pred': pred_label_text,
            'gold': label_text
        }
        all_output.append(instance)

    # Compute classification metrics
    acc = accuracy_score(all_gold, all_pred)
    micro_f1 = f1_score(all_gold, all_pred, average='micro')
    macro_f1 = f1_score(all_gold, all_pred, average='macro')

    # Compute AUROC
    # We need integer IDs for each label
    y_true = [label2id[lbl] for lbl in all_gold]
    y_prob = np.array(all_prob)  # shape (num_samples, num_labels)

    if len(label2id.items()) == 2:
        # Binary classification
        # If label_list = ['not-hateful', 'hateful'] or ['positive','negative'],
        # we can treat index=1 as the "positive" class:
        auroc = roc_auc_score(y_true, y_prob[:, 1])
    else:
        # Multi-class
        # Binarize y_true for each class
        y_true_bin = label_binarize(y_true, classes=range(len(label2id.items())))
        auroc = roc_auc_score(y_true_bin, y_prob, multi_class='ovr')

    results = {
        'acc': acc,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'auroc': auroc,
    }
    print(results)

    # Save final results
    final_results = {
        'results': results,
        'all_output': all_output,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, args.output_file), 'w', encoding='utf8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON and images.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory for text files.")
    parser.add_argument("--lmr_dir", type=str, required=True, help="Directory for text files.")
    parser.add_argument("--model_path", type=str, required=True, help="Directory for text files.")
    parser.add_argument("--input_json", type=str, required=True, help="Input JSON file.")
    parser.add_argument("--output_dir", type=str, default='results', help="Output directory.")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file.")
    parser.add_argument("--data_name", type=str, required=True, help="Which dataset: [hmc, twitter, msed]")

    args = parser.parse_args()
    process_json(args)
