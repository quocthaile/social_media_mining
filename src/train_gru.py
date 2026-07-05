import gzip
import os
import random
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

DEFAULT_TEXT_COLUMN = "tokens_text"
DEFAULT_FEATURE_COLUMNS = [
    "feat_log_num_tokens",
    "feat_log_num_chars",
    "feat_avg_token_len",
    "feat_emoji_density",
    "feat_punct_density",
    "feat_upper_ratio",
    "feat_digit_ratio",
]


def debug(msg: str) -> None:
    print(f"[DEBUG][GRU] {msg}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_default_data_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset-vihsd"))


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
    return df, texts, meta_features, labels


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


def tokenize(text: str):
    return text.split()


def build_vocab(train_texts, max_vocab_size=50000, min_freq=1):
    counter = Counter()
    for text in train_texts:
        counter.update(tokenize(text))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if len(vocab) >= max_vocab_size:
            break
        vocab[token] = len(vocab)
    return vocab


def encode_text(text, vocab, max_len):
    token_ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokenize(text)[:max_len]]
    if len(token_ids) < max_len:
        token_ids += [vocab[PAD_TOKEN]] * (max_len - len(token_ids))
    return token_ids


class TextDataset(Dataset):
    def __init__(self, texts, meta_features, labels, vocab, max_len):
        self.inputs = [encode_text(t, vocab, max_len) for t in texts]
        self.meta_features = meta_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.long)
        meta = torch.tensor(self.meta_features[idx], dtype=torch.float)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, meta, y


class AttentionPooling(nn.Module):
    """Additive attention over GRU output tokens; ignores padding positions."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, outputs: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # outputs : (B, T, H)   pad_mask : (B, T) True = non-padding
        scores = self.attn(outputs).squeeze(-1)           # (B, T)
        scores = scores.masked_fill(~pad_mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        return (outputs * weights).sum(dim=1)             # (B, H)


class GRUClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_size,
        num_classes,
        num_layers=1,
        bidirectional=True,
        dropout=0.3,
        padding_idx=0,
        num_meta_features=0,
        meta_hidden_size=32,
        pretrained_embeddings=None,
    ):
        super().__init__()
        if pretrained_embeddings is not None:
            weight = torch.tensor(pretrained_embeddings, dtype=torch.float)
            self.embedding = nn.Embedding.from_pretrained(
                weight, freeze=False, padding_idx=padding_idx
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        gru_out_dim = hidden_size * (2 if bidirectional else 1)
        self.attention = AttentionPooling(gru_out_dim)
        self.dropout = nn.Dropout(dropout)
        self.meta_proj = None
        fc_in_dim = gru_out_dim
        if num_meta_features > 0:
            self.meta_proj = nn.Sequential(
                nn.Linear(num_meta_features, meta_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            fc_in_dim = gru_out_dim + meta_hidden_size
        self.fc = nn.Linear(fc_in_dim, num_classes)

    def forward(self, input_ids, meta_features=None):
        emb = self.embedding(input_ids)
        outputs, _ = self.gru(emb)

        pad_mask = (input_ids != 0)          # (B, T) True = non-padding token
        pooled = self.attention(outputs, pad_mask)

        pooled = self.dropout(pooled)
        if self.meta_proj is not None and meta_features is not None:
            meta_repr = self.meta_proj(meta_features)
            pooled = torch.cat([pooled, meta_repr], dim=1)
        return self.fc(pooled)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    debug(f"Training epoch with {len(loader)} batches")

    for batch_idx, (x, meta, y) in enumerate(loader, start=1):
        x, meta, y = x.to(device), meta.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x, meta)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        if batch_idx % 50 == 0 or batch_idx == len(loader):
            debug(f"Train batch {batch_idx}/{len(loader)} | loss={loss.item():.4f}")

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []
    debug(f"Evaluating with {len(loader)} batches")

    for batch_idx, (x, meta, y) in enumerate(loader, start=1):
        x, meta, y = x.to(device), meta.to(device), y.to(device)
        logits = model(x, meta)
        loss = criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        total_loss += loss.item() * x.size(0)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        if batch_idx % 50 == 0 or batch_idx == len(loader):
            debug(f"Eval batch {batch_idx}/{len(loader)}")

    return total_loss / len(loader.dataset), y_true, y_pred


def load_fasttext_embeddings(vocab: dict, fasttext_path: str, embed_dim: int) -> np.ndarray:
    """
    Build an embedding matrix (vocab_size x embed_dim) from a pre-trained fastText file.

    Supported formats (no extra library required for text formats):
      - .vec        : fastText text format
      - .vec.gz     : gzip-compressed text format
      - .bin        : binary format — requires `pip install fasttext-wheel`
                      (not yet available for Python 3.14; use .vec instead)

    Download Vietnamese vectors from:
      https://fasttext.cc/docs/en/crawl-vectors.html
      e.g. cc.vi.300.vec.gz  (text format, ~1.2 GB compressed)
    """
    debug(f"Loading fastText embeddings from: {fasttext_path}")
    path_lower = fasttext_path.lower()
    matrix = np.zeros((len(vocab), embed_dim), dtype=np.float32)

    if path_lower.endswith(".bin"):
        try:
            import fasttext as _ft  # noqa: PLC0415

            ft = _ft.load_model(fasttext_path)
            found = sum(
                1
                for word, idx in vocab.items()
                if word not in (PAD_TOKEN, UNK_TOKEN)
                and _assign_vec(matrix, idx, ft.get_word_vector(word), embed_dim)
            )
            debug(f"fastText (.bin): {found}/{len(vocab)} tokens loaded")
        except ImportError as exc:
            raise ImportError(
                "Binary fastText (.bin) requires the 'fasttext' package which is not yet\n"
                "available for Python 3.14.\n"
                "Solution: download the .vec.gz text file instead:\n"
                "  https://fasttext.cc/docs/en/crawl-vectors.html  (cc.vi.300.vec.gz)\n"
                "Then pass: --fasttext_path path/to/cc.vi.300.vec.gz"
            ) from exc
    else:
        open_fn = gzip.open if path_lower.endswith(".gz") else open
        found = 0
        with open_fn(fasttext_path, "rt", encoding="utf-8") as fh:
            fh.readline()  # skip header: "<n_vocab> <dim>"
            for line in fh:
                parts = line.rstrip().split(" ")
                if len(parts) < embed_dim + 1:
                    continue
                word = parts[0]
                if word not in vocab:
                    continue
                try:
                    vec = np.array(parts[1:], dtype=np.float32)
                    if vec.shape[0] == embed_dim:
                        matrix[vocab[word]] = vec
                        found += 1
                except ValueError:
                    continue
        debug(f"fastText (.vec): {found}/{len(vocab)} tokens loaded")

    return matrix


def _assign_vec(matrix, idx, vec, embed_dim):
    """Helper to assign a vector; returns True if dimension matches."""
    if len(vec) == embed_dim:
        matrix[idx] = vec
        return True
    return False


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
    ovr_df = pd.DataFrame(rows)
    print(ovr_df.to_string(index=False))

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1-macro: {f1_macro:.4f}")
    return cm_df, ovr_df, acc, f1_macro


def save_evaluation_artifacts(
    output_dir,
    split_df,
    y_true,
    y_pred,
    id2label,
    confusion_matrix_file,
    ovr_metrics_file,
    misclassified_file,
):
    if len(split_df) != len(y_true) or len(split_df) != len(y_pred):
        raise ValueError("Prediction length does not match split dataframe length")

    cm_df, ovr_df, acc, f1_macro = print_confusion_and_scores(y_true, y_pred, id2label)

    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, confusion_matrix_file)
    ovr_path = os.path.join(output_dir, ovr_metrics_file)
    mis_path = os.path.join(output_dir, misclassified_file)

    cm_df.to_csv(cm_path, encoding="utf-8-sig")
    ovr_df.to_csv(ovr_path, index=False, encoding="utf-8-sig")

    eval_df = split_df.reset_index(drop=True).copy()
    eval_df["label_id_true"] = [id2label[i] for i in y_true]
    eval_df["label_id_pred"] = [id2label[i] for i in y_pred]
    eval_df["is_misclassified"] = eval_df["label_id_true"] != eval_df["label_id_pred"]

    mis_df = eval_df[eval_df["is_misclassified"]].copy()
    mis_df.to_csv(mis_path, index=False, encoding="utf-8-sig")

    debug(f"Exported confusion matrix to: {cm_path}")
    debug(f"Exported one-vs-rest table to: {ovr_path}")
    debug(f"Exported misclassified rows to: {mis_path} (rows={len(mis_df)})")
    debug(f"Final test metrics | accuracy={acc:.4f}, f1_macro={f1_macro:.4f}")


def parse_feature_columns(raw: str):
    columns = [c.strip() for c in raw.split(",") if c.strip()]
    if not columns:
        raise ValueError("feature_columns must include at least one column")
    return columns


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate GRU")
    parser.add_argument("--data_dir", type=str, default=get_default_data_dir())
    parser.add_argument("--train_file", type=str, default="features_train.csv")
    parser.add_argument("--dev_file", type=str, default="features_dev.csv")
    parser.add_argument("--test_file", type=str, default="features_test.csv")
    parser.add_argument("--text_column", type=str, default=DEFAULT_TEXT_COLUMN)
    parser.add_argument("--feature_columns", type=str, default=",".join(DEFAULT_FEATURE_COLUMNS))

    parser.add_argument("--max_vocab_size", type=int, default=50000)
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=128)

    parser.add_argument("--embed_dim", type=int, default=200)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument(
        "--bidirectional",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bidirectional GRU (default: True). Use --no-bidirectional to disable.",
    )
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--meta_hidden_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "gru")),
    )
    parser.add_argument("--confusion_matrix_file", type=str, default="confusion_matrix_test.csv")
    parser.add_argument("--ovr_metrics_file", type=str, default="ovr_metrics_test.csv")
    parser.add_argument("--misclassified_file", type=str, default="misclassified_test.csv")
    # --- Optimization flags ---
    parser.add_argument(
        "--class_weight_mode",
        type=str,
        choices=["balanced", "none"],
        default="balanced",
        help="'balanced' weights CrossEntropyLoss inversely by class frequency. 'none' = uniform.",
    )
    parser.add_argument(
        "--use_weighted_sampler",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="WeightedRandomSampler on train set to oversample minority classes (default: True).",
    )
    parser.add_argument(
        "--fasttext_path",
        type=str,
        default=None,
        help="Path to pre-trained fastText vectors (.vec, .vec.gz). Optional.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    feature_columns = parse_feature_columns(args.feature_columns)
    debug("Starting script")
    debug(f"Arguments: {vars(args)}")
    set_seed(args.seed)
    debug(f"Seed set to {args.seed}")
    debug(f"Feature columns: {feature_columns}")

    debug("Loading train/dev/test splits")
    _, train_texts, train_meta, train_labels_raw = load_split(
        os.path.join(args.data_dir, args.train_file),
        text_column=args.text_column,
        feature_columns=feature_columns,
    )
    _, dev_texts, dev_meta, dev_labels_raw = load_split(
        os.path.join(args.data_dir, args.dev_file),
        text_column=args.text_column,
        feature_columns=feature_columns,
    )
    test_df, test_texts, test_meta, test_labels_raw = load_split(
        os.path.join(args.data_dir, args.test_file),
        text_column=args.text_column,
        feature_columns=feature_columns,
    )
    debug(
        f"Loaded rows | train={len(train_texts)}, dev={len(dev_texts)}, test={len(test_texts)}"
    )

    label2id, id2label = create_label_mapping(train_labels_raw, dev_labels_raw, test_labels_raw)
    train_labels = encode_labels(train_labels_raw, label2id)
    dev_labels = encode_labels(dev_labels_raw, label2id)
    test_labels = encode_labels(test_labels_raw, label2id)
    debug(f"Label mapping: {label2id}")

    debug("Building vocabulary")
    vocab = build_vocab(train_texts, max_vocab_size=args.max_vocab_size, min_freq=args.min_freq)
    debug(f"Vocabulary size: {len(vocab)}")

    pretrained_embeddings = None
    if args.fasttext_path:
        pretrained_embeddings = load_fasttext_embeddings(vocab, args.fasttext_path, args.embed_dim)

    debug("Building datasets and dataloaders")
    train_ds = TextDataset(train_texts, train_meta, train_labels, vocab, args.max_len)
    dev_ds = TextDataset(dev_texts, dev_meta, dev_labels, vocab, args.max_len)
    test_ds = TextDataset(test_texts, test_meta, test_labels, vocab, args.max_len)

    if args.use_weighted_sampler:
        class_counts = Counter(train_labels)
        sample_weights = [1.0 / class_counts[lbl] for lbl in train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
        )
        debug(f"WeightedRandomSampler enabled | class_counts: {dict(sorted(class_counts.items()))}")
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    debug(
        f"DataLoader batches | train={len(train_loader)}, dev={len(dev_loader)}, test={len(test_loader)}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug(f"Using device: {device}")

    model = GRUClassifier(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        num_classes=len(id2label),
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        padding_idx=vocab[PAD_TOKEN],
        num_meta_features=len(feature_columns),
        meta_hidden_size=args.meta_hidden_size,
        pretrained_embeddings=pretrained_embeddings,
    ).to(device)
    debug("Model initialized")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.class_weight_mode == "balanced":
        class_weight_arr = compute_class_weight(
            class_weight="balanced",
            classes=np.array(sorted(id2label.keys())),
            y=train_labels,
        )
        weight_tensor = torch.tensor(class_weight_arr, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        debug(
            f"Class weights (balanced): "
            + str({id2label[i]: round(float(w), 4) for i, w in enumerate(class_weight_arr)})
        )
    else:
        criterion = nn.CrossEntropyLoss()
    debug("Optimizer and criterion initialized")

    best_dev_f1 = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        debug(f"Epoch {epoch}/{args.epochs} started")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        dev_loss, dev_true, dev_pred = evaluate(model, dev_loader, criterion, device)

        dev_acc = accuracy_score(dev_true, dev_pred)
        dev_f1 = f1_score(dev_true, dev_pred, average="macro")

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"dev_loss={dev_loss:.4f} | dev_acc={dev_acc:.4f} | dev_f1_macro={dev_f1:.4f}"
        )

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
    _, test_true, test_pred = evaluate(model, test_loader, criterion, device)

    print("\n===== GRU Test Metrics =====")
    save_evaluation_artifacts(
        output_dir=args.output_dir,
        split_df=test_df,
        y_true=test_true,
        y_pred=test_pred,
        id2label=id2label,
        confusion_matrix_file=args.confusion_matrix_file,
        ovr_metrics_file=args.ovr_metrics_file,
        misclassified_file=args.misclassified_file,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "best_gru.pt")
    debug(f"Saving model to {save_path}")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "label2id": label2id,
            "id2label": id2label,
            "text_column": args.text_column,
            "feature_columns": feature_columns,
            "args": vars(args),
        },
        save_path,
    )
    print(f"\nSaved best model to: {save_path}")
    debug("Script finished successfully")


if __name__ == "__main__":
    main()
