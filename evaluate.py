import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import warnings
from bert_score import score
from sklearn.model_selection import KFold
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


# =========================
#  K-Fold 控制参数
# =========================
NUM_FOLDS = 5
RUN_SINGLE_FOLD = False
SINGLE_FOLD_ID = 1
SEED = 42

# =========================
# 配置参数
# =========================
MODEL_NAME = "/home/zzh/lora_train"
TEST_DATA_PATH = "combined_test.json"
MAX_NEW_TOKENS = 2048
BATCH_SIZE = 8
NUM_SAMPLES = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SentenceTransformer("intfloat/e5-large-v2")
warnings.filterwarnings("ignore")

# =========================
# 模型加载
# =========================
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    os.environ["DISABLE_FLASH_ATTENTION_2"] = "1"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    semantic_model.to(model.device)
    return tokenizer, model

tokenizer, model = load_models()

# =========================
# E5 embedding 加载（正确方式）
# =========================
E5_NAME = "intfloat/e5-large-v2"
e5_tokenizer = AutoTokenizer.from_pretrained(E5_NAME)
e5_model = AutoModel.from_pretrained(E5_NAME).to(DEVICE)
e5_model.eval()

def encode_e5(texts, batch_size=32):
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = e5_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            output = e5_model(**inputs).last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            emb = (output * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            emb = F.normalize(emb, p=2, dim=1)
            all_emb.append(emb.cpu())

    return torch.cat(all_emb, dim=0)

# =========================
# 数据加载
# =========================
def load_test_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[:NUM_SAMPLES]

all_data = load_test_data(TEST_DATA_PATH)

# =========================
# 文本清洗
# =========================
def clean_text(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|im_end\|>|<\|endoftext\|>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =========================
# Token 指标
# =========================
def calculate_token_metrics(preds, refs):
    # ✅ 过滤 empty
    pairs = [(p, r) for p, r in zip(preds, refs) if p.strip() and r.strip()]
    if not pairs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    precisions, recalls, f1s, accuracies = [], [], [], []

    for pred, ref in pairs:
        ps, rs = set(pred.split()), set(ref.split())
        common = ps & rs

        p = len(common) / len(ps) if ps else 0
        r = len(common) / len(rs) if rs else 0
        f1 = 2 * p * r / (p + r + 1e-9)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        accuracies.append(r)

    return {
        "precision": round(np.mean(precisions), 4),
        "recall": round(np.mean(recalls), 4),
        "f1": round(np.mean(f1s), 4),
        "accuracy": round(np.mean(accuracies), 4)
    }

# =========================
# ROUGE + BLEU（ 官方 rouge-score）
# =========================
def calculate_classic_metrics(preds, refs):
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=False)
    smoothie = SmoothingFunction().method4

    r1, r2, rl, bleu_scores = [], [], [], []

    # ✅ 过滤 empty
    pairs = [(p, r) for p, r in zip(preds, refs) if p.strip() and r.strip()]
    if not pairs:
        return {
            "rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0,
            "bleu": 0.0, "precision": 0.0, "recall": 0.0,
            "f1": 0.0, "accuracy": 0.0
        }

    for p, r in pairs:
        scores = scorer.score(r, p)
        r1.append(scores['rouge1'].fmeasure)
        r2.append(scores['rouge2'].fmeasure)
        rl.append(scores['rougeL'].fmeasure)
        bleu_scores.append(sentence_bleu([r.split()], p.split(), smoothing_function=smoothie))

    token_metrics = calculate_token_metrics(
        [p for p, r in pairs],
        [r for p, r in pairs]
    )

    return {
        "rouge-1": round(np.mean(r1), 4),
        "rouge-2": round(np.mean(r2), 4),
        "rouge-l": round(np.mean(rl), 4),
        "bleu": round(np.mean(bleu_scores), 4),
        **token_metrics
    }

# =========================
# 语义相似度（过滤 empty）
# =========================
def calculate_semantic_metrics(preds, refs):
    pairs = [(p, r) for p, r in zip(preds, refs) if p.strip() and r.strip()]
    preds = [p for p, r in pairs]
    refs  = [r for p, r in pairs]

    pred_emb = encode_e5(preds)
    ref_emb  = encode_e5(refs)

    cos = torch.nn.functional.cosine_similarity(pred_emb, ref_emb).numpy()

    return {
        "semantic_cosine_mean": round(np.mean(cos), 4),
        "semantic_cosine_std":  round(np.std(cos), 4),
        "semantic_cosine_min":  round(np.min(cos), 4),
        "semantic_cosine_max":  round(np.max(cos), 4)
    }


# =========================
# 中文 BERTScore（过滤 empty）
# =========================
def calculate_bertscore(preds, refs):
    pairs = [(p, r) for p, r in zip(preds, refs) if p.strip() and r.strip()]
    preds = [p for p, r in pairs]
    refs  = [r for p, r in pairs]

    P, R, F1 = score(preds, refs, lang="zh", verbose=False)

    return {
        "bertscore_precision_mean": round(P.mean().item(), 4),
        "bertscore_precision_std":  round(P.std().item(), 4),
        "bertscore_recall_mean":    round(R.mean().item(), 4),
        "bertscore_recall_std":     round(R.std().item(), 4),
        "bertscore_f1_mean":        round(F1.mean().item(), 4),
        "bertscore_f1_std":         round(F1.std().item(), 4)
    }

# =========================
# 生成
# =========================
def batch_generate(questions):
    all_responses = []

    for q in tqdm(questions):
        inputs = tokenizer(q, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        all_responses.append(clean_text(decoded))

    return all_responses

# =========================
# 单折评估
# =========================
def evaluate_one_fold(fold_data, fold_id):
    questions = [clean_text(d["question"]) for d in fold_data]
    references = [clean_text(d["answer"]) for d in fold_data]

    answers = batch_generate(questions)

    metrics = {
        "classic": calculate_classic_metrics(answers, references),
        "semantic": calculate_semantic_metrics(answers, references),
        "bertscore": calculate_bertscore(answers, references)
    }

    with open(f"metrics_fold_{fold_id}.json","w",encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=lambda x: float(x))

    return metrics

# =========================
# K-Fold 主流程
# =========================
def evaluate_kfold():
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    all_results = []

    for fold_id, (_, test_idx) in enumerate(kf.split(all_data), 1):

        if RUN_SINGLE_FOLD and fold_id != SINGLE_FOLD_ID:
            continue

        print(f"\n===== Running Fold {fold_id}/{NUM_FOLDS} =====")
        fold_data = [all_data[i] for i in test_idx]
        fold_metrics = evaluate_one_fold(fold_data, fold_id)
        all_results.append(fold_metrics)

    summary = {}
    for key in all_results[0]:
        summary[key] = {}
        for metric in all_results[0][key]:
            values = [r[key][metric] for r in all_results]
            summary[key][metric] = {
                "mean": round(float(np.mean(values)), 4),
                "std":  round(float(np.std(values)), 4)
            }

    with open("metrics_5fold_summary.json","w",encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== 5-Fold Summary Saved =====")

# =========================
# 程序入口
# =========================
if __name__ == "__main__":
    evaluate_kfold()
