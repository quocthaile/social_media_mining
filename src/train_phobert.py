import os
import random
import argparse
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModel,
    PreTrainedTokenizerFast,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import SequenceClassifierOutput


DEFAULT_TEXT_COLUMN = "tokens_text"  # Mặc định PhoBERT dùng text CÓ gạch dưới
DEFAULT_FEATURE_COLUMNS = [
    "feat_log_num_tokens",        
    "feat_upper_ratio",           
    "feat_emoji_density",         
    "feat_bad_word_density",      
    "feat_aggressive_pronoun",    
    "feat_laugh_density",         
    "feat_sarcastic_punct",       
    "feat_scare_quotes",          
    "feat_intensifier_words",     
    "feat_elongated_ratio",       
]

def debug(msg: str) -> None:
    print(f"[DEBUG][Transformer] {msg}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_default_data_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset-vihsd"))


def get_default_output_dir(subdir: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", subdir))


def load_split(csv_path: str, text_column: str, feature_columns):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required_cols = {text_column, "label_id", *feature_columns}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain columns: {required_cols}")

    texts = df[text_column].fillna("").astype(str).tolist()
    meta_features = (
        df[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype(np.float32)
        .values
    )
    labels = pd.to_numeric(df["label_id"], errors="raise").astype(int).tolist()
    return texts, meta_features, labels


def create_label_mapping(*label_lists):
    all_labels = []
    for labels in label_lists:
        all_labels.extend(labels)
    unique_labels = sorted(set(all_labels))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def encode_labels(labels, label2id):
    return [label2id[label] for label in labels]


class TransformerDataset(Dataset):
    def __init__(self, texts, meta_features, labels, tokenizer, max_len):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )
        self.meta_features = meta_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx], dtype=torch.long) for k, v in self.encodings.items()}
        item["meta_features"] = torch.tensor(self.meta_features[idx], dtype=torch.float)
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

class TransformerWithMetaFeatures(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        num_meta_features: int,
        feature_hidden_size: int,
        text_dropout: float,
        feature_dropout: float,
        class_weights=None, # THÊM THAM SỐ CLASS WEIGHTS
        id2label=None,
        label2id=None,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        encoder_hidden_size = self.encoder.config.hidden_size

        self.text_dropout = nn.Dropout(text_dropout)
        self.meta_proj = nn.Sequential(
            nn.BatchNorm1d(num_meta_features),
            nn.Linear(num_meta_features, feature_hidden_size),
            nn.SiLU(),
            nn.Dropout(feature_dropout),
        )
        self.classifier = nn.Linear(encoder_hidden_size + feature_hidden_size, num_labels)

        self.encoder.config.num_labels = num_labels
        if id2label is not None:
            self.encoder.config.id2label = id2label
        if label2id is not None:
            self.encoder.config.label2id = label2id

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        meta_features=None,
        labels=None,
    ):
        encoder_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_inputs)
        pooled_output = outputs.last_hidden_state[:, 0, :]

        text_repr = self.text_dropout(pooled_output)
        if meta_features is None:
            meta_features = torch.zeros(
                (text_repr.size(0), self.meta_proj[0].normalized_shape[0]),
                dtype=text_repr.dtype,
                device=text_repr.device,
            )
        meta_repr = self.meta_proj(meta_features)

        fused = torch.cat([text_repr, meta_repr], dim=1)
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            # SỬ DỤNG TRỌNG SỐ ĐỘNG TỪ BỘ ĐỆM
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
            
        return SequenceClassifierOutput(loss=loss, logits=logits)


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    debug(f"Training epoch with {len(loader)} batches")

    for batch_idx, batch in enumerate(loader, start=1):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * batch["labels"].size(0)
        if batch_idx % 20 == 0 or batch_idx == len(loader):
            debug(f"Train batch {batch_idx}/{len(loader)} | loss={loss.item():.4f}")

    return total_loss / len(loader.dataset)

# @torch.no_grad()
# def evaluate(model, loader, device):
#     model.eval()
#     total_loss = 0.0
#     y_true, y_pred = [], []
#     debug(f"Evaluating with {len(loader)} batches")

#     for batch_idx, batch in enumerate(loader, start=1):
#         batch = {k: v.to(device) for k, v in batch.items()}
#         outputs = model(**batch)

#         loss = outputs.loss
#         logits = outputs.logits
        
#         # --- BẬT LẠI THRESHOLD MOVING ---
#         probs = torch.softmax(logits, dim=1)
#         batch_preds = []
#         for p in probs:
#             if p[2] > 0.30:        # Mốc Thù địch: 30%
#                 batch_preds.append(2)
#             elif p[1] > 0.25:      # Mốc Xúc phạm: 25%
#                 batch_preds.append(1)
#             else:
#                 batch_preds.append(torch.argmax(p).item())
                
#         preds = torch.tensor(batch_preds)
#         # --------------------------------

#         total_loss += loss.item() * batch["labels"].size(0)
#         y_true.extend(batch["labels"].cpu().tolist())
#         y_pred.extend(preds.cpu().tolist())
        
#         if batch_idx % 20 == 0 or batch_idx == len(loader):
#             debug(f"Eval batch {batch_idx}/{len(loader)}")

#     return total_loss / len(loader.dataset), y_true, y_pred
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []

    for batch_idx, batch in enumerate(loader, start=1):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        
        loss = outputs.loss
        logits = outputs.logits
        
        # CHỈ DÙNG ARGMAX TINH KHIẾT (Vì Loss đã có Class Weights)
        preds = torch.argmax(logits, dim=1)
        
        total_loss += loss.item() * batch["labels"].size(0)
        y_true.extend(batch["labels"].cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        
    return total_loss / len(loader.dataset), y_true, y_pred


def print_confusion_and_scores(y_true, y_pred, id2label):
    label_ids = list(range(len(id2label)))
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)

    cm_df = pd.DataFrame(
        cm,
        index=[f"Origin_{id2label[i]}" for i in label_ids],
        columns=[f"Prediction_{id2label[i]}" for i in label_ids],
    )

    print("\n=== Confusion Matrix (Origin rows x Prediction columns) ===")
    print(cm_df.to_string())

    total = cm.sum()
    rows = []
    for i in label_ids:
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(total - tp - fn - fp)
        rows.append(
            {
                "Class": id2label[i],
                "TN": tn,
                "TP": tp,
                "FN": fn,
                "FP": fp,
            }
        )

    print("\n=== One-vs-Rest Table (TN TP FN FP) ===")
    print(pd.DataFrame(rows).to_string(index=False))

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1-macro: {f1_macro:.4f}")
    # print(f"\nAccuracy: {acc:.4f}")
    # print(f"F1-macro: {f1_macro:.4f}")
    
    # THÊM DÒNG NÀY VÀO CUỐI HÀM
    return cm_df, pd.DataFrame(rows), acc, f1_macro

def build_export_filename(model_prefix: str, filename: str) -> str:
    if not filename:
        return f"{model_prefix}_artifact.csv"
    base_name = os.path.basename(filename)
    if base_name.startswith(f"{model_prefix}_"):
        return base_name
    return f"{model_prefix}_{base_name}"


def save_evaluation_artifacts(
    output_dir, split_df, y_true, y_pred, id2label,
    confusion_matrix_file, ovr_metrics_file, misclassified_file, model_prefix="phobert"
):
    cm_df, ovr_df, acc, f1_macro = print_confusion_and_scores(y_true, y_pred, id2label)
    os.makedirs(output_dir, exist_ok=True)
    
    cm_path = os.path.join(output_dir, build_export_filename(model_prefix, confusion_matrix_file))
    ovr_path = os.path.join(output_dir, build_export_filename(model_prefix, ovr_metrics_file))
    mis_path = os.path.join(output_dir, build_export_filename(model_prefix, misclassified_file))

    cm_df.to_csv(cm_path, encoding="utf-8-sig")
    ovr_df.to_csv(ovr_path, index=False, encoding="utf-8-sig")

    eval_df = split_df.reset_index(drop=True).copy()
    eval_df["label_id_true"] = [id2label[i] for i in y_true]
    eval_df["label_id_pred"] = [id2label[i] for i in y_pred]
    eval_df["is_misclassified"] = eval_df["label_id_true"] != eval_df["label_id_pred"]

    mis_df = eval_df[eval_df["is_misclassified"]].copy()
    mis_df.to_csv(mis_path, index=False, encoding="utf-8-sig")


def parse_feature_columns(raw: str):
    columns = [c.strip() for c in raw.split(",") if c.strip()]
    if not columns:
        raise ValueError("feature_columns must include at least one column")
    return columns


# THÊM tham số default_text_column="tokens_text" vào hàm
def run_experiment(
    default_model_name="vinai/phobert-base",
    default_output_subdir="phobert",
    run_name="PhoBERT",
    default_text_column="tokens_text",
    model_prefix="phobert"
):
    model_prefix = "phobert"
    parser = argparse.ArgumentParser(description=f"Train {run_name}")
    parser.add_argument("--data_dir", type=str, default=get_default_data_dir())
    parser.add_argument("--train_file", type=str, default="features_train.csv")
    parser.add_argument("--dev_file", type=str, default="features_dev.csv")
    parser.add_argument("--test_file", type=str, default="features_test.csv")
    
    # SỬA TẠI ĐÂY: Trỏ biến default vào tham số mới truyền vào
    parser.add_argument("--text_column", type=str, default=default_text_column) 
    # === BỔ SUNG DÒNG NÀY ĐỂ VÁ LỖI ATTRIBUTEERROR ===
    parser.add_argument("--model_name", type=str, default=default_model_name, help="Tên hoặc đường dẫn mô hình Transformer")
    # =================================================
    
    parser.add_argument("--feature_columns", type=str, default=",".join(DEFAULT_FEATURE_COLUMNS))
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32) 
    parser.add_argument("--epochs", type=int, default=15)     
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--feature_hidden_size", type=int, default=32)
    parser.add_argument("--text_dropout", type=float, default=0.1)     
    parser.add_argument("--feature_dropout", type=float, default=0.3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=get_default_output_dir(default_output_subdir))
    parser.add_argument("--confusion_matrix_file", type=str, default="confusion_matrix_test.csv")
    parser.add_argument("--ovr_metrics_file", type=str, default="ovr_metrics_test.csv")
    parser.add_argument("--misclassified_file", type=str, default="misclassified_test.csv")
    args = parser.parse_args()
    feature_columns = parse_feature_columns(args.feature_columns)

    debug(f"Starting run: {run_name}")
    debug(f"Arguments: {vars(args)}")
    set_seed(args.seed)
    debug(f"Seed set to {args.seed}")
    debug(f"Feature columns: {feature_columns}")

    debug("Loading train/dev/test splits")
    train_texts, train_meta, train_labels_raw = load_split(
        os.path.join(args.data_dir, args.train_file),
        text_column=args.text_column,
        feature_columns=feature_columns,
    )
    dev_texts, dev_meta, dev_labels_raw = load_split(
        os.path.join(args.data_dir, args.dev_file),
        text_column=args.text_column,
        feature_columns=feature_columns,
    )
    test_texts, test_meta, test_labels_raw = load_split(
        os.path.join(args.data_dir, args.test_file),
        text_column=args.text_column,
        feature_columns=feature_columns,
    )
    test_df = pd.read_csv(os.path.join(args.data_dir, args.test_file), encoding="utf-8-sig")
    debug(
        f"Loaded rows | train={len(train_texts)}, dev={len(dev_texts)}, test={len(test_texts)}"
    )

    label2id, id2label = create_label_mapping(train_labels_raw, dev_labels_raw, test_labels_raw)
    train_labels = encode_labels(train_labels_raw, label2id)
    dev_labels = encode_labels(dev_labels_raw, label2id)
    test_labels = encode_labels(test_labels_raw, label2id)
    debug(f"Label mapping: {label2id}")

    if "bamibert" in args.model_name.lower():
        debug(f"Loading tokenizer: {args.model_name} (PreTrainedTokenizerFast, bypassing AutoTokenizer)")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_name)
    else:
        use_fast = False if "phobert" in args.model_name.lower() else True
        debug(f"Loading tokenizer: {args.model_name} (use_fast={use_fast})")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=use_fast)

    debug("Building tokenized datasets")
    train_ds = TransformerDataset(train_texts, train_meta, train_labels, tokenizer, args.max_len)
    dev_ds = TransformerDataset(dev_texts, dev_meta, dev_labels, tokenizer, args.max_len)
    test_ds = TransformerDataset(test_texts, test_meta, test_labels, tokenizer, args.max_len)

# Tính toán trọng số lấy mẫu cho từng dòng dữ liệu trong tập Train
    # class_sample_counts = np.bincount(train_labels)
    # class_weights_sampler = 1. / class_sample_counts
    # sample_weights = np.array([class_weights_sampler[t] for t in train_labels])
    # sample_weights = torch.from_numpy(sample_weights).double()
    
    # Tạo Sampler thay cho việc Shuffle mặc định
    # sampler = WeightedRandomSampler(
    #     weights=sample_weights,
    #     num_samples=len(sample_weights),
    #     replacement=True
    # )

    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, # Bật lại shuffle tự nhiên
        num_workers=args.num_workers
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    debug(
        f"DataLoader batches | train={len(train_loader)}, dev={len(dev_loader)}, test={len(test_loader)}"
    )

    num_labels = len(id2label)
    hf_id2label = {i: str(id2label[i]) for i in range(num_labels)}
    hf_label2id = {str(id2label[i]): i for i in range(num_labels)}

    debug("Calculating dynamic class weights based on Train distribution")
    # TÍNH TOÁN TRỌNG SỐ ĐỘNG
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)

    calculated_weights = np.sqrt(total_samples / (num_labels * class_counts))
    class_weights_tensor = torch.tensor(calculated_weights, dtype=torch.float)
    debug(f"Computed Smoothed Class Weights: {class_weights_tensor.tolist()}")
    debug(f"Computed Class Weights: {class_weights_tensor.tolist()}")
    debug(f"Loading model: {args.model_name}")
    model = TransformerWithMetaFeatures(
        model_name=args.model_name,
        num_labels=num_labels,
        num_meta_features=len(feature_columns),
        feature_hidden_size=args.feature_hidden_size,
        text_dropout=args.text_dropout,       
        feature_dropout=args.feature_dropout, 
        class_weights=class_weights_tensor,
        id2label=hf_id2label,
        label2id=hf_label2id,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    debug(f"Using device: {device}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    debug(f"Scheduler steps | total={total_steps}, warmup={warmup_steps}")

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_dev_f1 = -1.0
    best_state = None
    no_improve = 0

    print(f"=== Running: {run_name} | model={args.model_name} ===")
    for epoch in range(1, args.epochs + 1):
        debug(f"Epoch {epoch}/{args.epochs} started")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        dev_loss, dev_true, dev_pred = evaluate(model, dev_loader, device)

        dev_acc = accuracy_score(dev_true, dev_pred)
        dev_f1 = f1_score(dev_true, dev_pred, average="macro")

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"dev_loss={dev_loss:.4f} | dev_acc={dev_acc:.4f} | dev_f1_macro={dev_f1:.4f}"
        )

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_state = deepcopy(model.state_dict())
            no_improve = 0
            debug(f"New best dev_f1={best_dev_f1:.4f} at epoch {epoch}")
        else:
            no_improve += 1
            debug(f"No improvement count: {no_improve}/{args.patience}")
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience}).")
                break

    if best_state is not None:
        debug("Loading best checkpoint from training")
        model.load_state_dict(best_state)

    debug("Running final evaluation on test set")
    _, test_true, test_pred = evaluate(model, test_loader, device)

    print(f"\n===== {run_name} Test Metrics =====")
    # print_confusion_and_scores(test_true, test_pred, id2label)
    save_evaluation_artifacts(
        output_dir=args.output_dir,
        split_df=test_df,
        y_true=test_true,
        y_pred=test_pred,
        id2label=id2label,
        confusion_matrix_file=args.confusion_matrix_file,
        ovr_metrics_file=args.ovr_metrics_file,
        misclassified_file=args.misclassified_file,
        model_prefix=model_prefix
    )

    os.makedirs(args.output_dir, exist_ok=True)
    debug(f"Saving model and tokenizer to {args.output_dir}")
    save_path = os.path.join(args.output_dir, f"{model_prefix}_best_model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": args.model_name,
            "num_labels": num_labels,
            "text_column": args.text_column,
            "feature_columns": feature_columns,
            "feature_hidden_size": args.feature_hidden_size,
            "feature_dropout": args.feature_dropout,
            "label2id": label2id,
            "id2label": id2label,
            "args": vars(args),
        },
        save_path,
    )
    tokenizer.save_pretrained(args.output_dir)
    model.encoder.config.save_pretrained(args.output_dir)

    map_df = pd.DataFrame(
        {
            "mapped_id": list(range(num_labels)),
            "original_label": [id2label[i] for i in range(num_labels)],
        }
    )
    map_df.to_csv(os.path.join(args.output_dir, "label_mapping.csv"), index=False)
    print(f"\nSaved best model and tokenizer to: {args.output_dir}")
    print(f"Checkpoint path: {save_path}")
    debug(f"Run finished: {run_name}")


def main():
    run_experiment(
        default_model_name="vinai/phobert-base",
        default_output_subdir="phobert",
        run_name="PhoBERT",
    )


if __name__ == "__main__":
    main()
