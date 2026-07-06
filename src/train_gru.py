import os
import random
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader

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
    "feat_bad_word_density",      # <--- BỔ SUNG: Rất quan trọng cho nhãn OFFENSIVE/HATE
    "feat_exclamation_density",  # <--- BỔ SUNG: Biểu thị sắc thái kích động
    "feat_allcaps_ratio"            # <--- BỔ SUNG: Biểu thị la hét/chửi bới
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


# def load_split(csv_path: str, text_column: str, feature_columns):
#     df = pd.read_csv(csv_path, encoding="utf-8-sig")
#     required_cols = {text_column, "label_id", *feature_columns}
#     if not required_cols.issubset(df.columns):
#         raise ValueError(f"{csv_path} must contain columns: {required_cols}")

#     texts = df[text_column].fillna("").astype(str).tolist()
#     meta_features = (
#         df[feature_columns]
#         .apply(pd.to_numeric, errors="coerce")
#         .fillna(0.0)
#         .astype(np.float32)
#         .values
#     )
#     labels = pd.to_numeric(df["label_id"], errors="raise").astype(int).tolist()
#     return df, texts, meta_features, labels
def load_split(csv_path: str, text_column: str, feature_columns):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required_cols = {text_column, "label_id", *feature_columns}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain columns: {required_cols}")

    # ===== THAY ĐỔI TẠI ĐÂY =====
    # Ép toàn bộ cột text về chữ in thường ngay khi vừa load từ CSV
    texts = df[text_column].fillna("").astype(str).str.lower().tolist()
    # ============================
    
    meta_features = (
        df[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype(np.float32)
        .values
    )
    labels = pd.to_numeric(df["label_id"], errors="raise").astype(int).tolist()
    
    # Đối với train_gru.py: return df, texts, meta_features, labels
    # Đối với train_textcnn.py: return texts, meta_features, labels
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


def load_fasttext_vectors(vec_path: str, vocab=None):
    """Tải file FastText (.vec) vào bộ nhớ dưới dạng dictionary."""
    debug(f"Đang tải FastText vectors từ {vec_path} (Quá trình này có thể mất vài phút)...")
    embeddings_dict = {}
    with open(vec_path, "r", encoding="utf-8") as f:
        first_line = f.readline().split()
        if len(first_line) != 2:
            f.seek(0)

        for line in f:
            values = line.rstrip().split(" ")
            word = values[0]
            word_lower = word.lower()

            if vocab is not None and word_lower not in vocab:
                continue

            vector = np.asarray(values[1:], dtype="float32")
            embeddings_dict[word_lower] = vector

    debug(f"Đã tải thành công {len(embeddings_dict)} vector từ vựng.")
    return embeddings_dict


def build_embedding_matrix(vocab, embeddings_dict, embed_dim=None):
    """Ánh xạ FastText vectors vào vocab của model."""
    debug("Đang khởi tạo Ma trận Embedding cho mô hình...")
    if not embeddings_dict:
        raise ValueError("embeddings_dict is empty, cannot build embedding matrix")

    fasttext_dim = len(next(iter(embeddings_dict.values())))
    if embed_dim is None:
        embed_dim = fasttext_dim
    elif embed_dim != fasttext_dim:
        debug(
            f"embed_dim mismatch: embed_dim={embed_dim}, FastText dim={fasttext_dim}. "
            f"Tự động đồng bộ embed_dim -> {fasttext_dim}."
        )
        embed_dim = fasttext_dim

    vocab_size = len(vocab)
    embedding_matrix = np.random.normal(scale=0.1, size=(vocab_size, embed_dim))

    hits = 0
    for word, idx in vocab.items():
        if word == PAD_TOKEN:
            embedding_matrix[idx] = np.zeros(embed_dim)
        elif word in embeddings_dict:
            embedding_matrix[idx] = embeddings_dict[word]
            hits += 1

    debug(f"Tỷ lệ khớp FastText: {hits}/{vocab_size} từ ({(hits / vocab_size) * 100:.2f}%).")
    return torch.tensor(embedding_matrix, dtype=torch.float32)


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
        pretrained_embeddings=None
    ):
        super().__init__()
        # KIỂM TRA VÀ NẠP MA TRẬN NHÚNG
        if pretrained_embeddings is not None:
            # freeze=False cho phép cập nhật lại trọng số của FastText trong lúc train
            # để mô hình thích nghi tốt hơn với từ lóng của dataset
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=False, padding_idx=padding_idx
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
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        # self.meta_proj = None
        # if num_meta_features > 0:
        #     self.meta_proj = nn.Sequential(
        #         nn.Linear(num_meta_features, meta_hidden_size),
        #         nn.ReLU(),
        #         nn.Dropout(dropout),
        #     )
        #     out_dim = out_dim + meta_hidden_size
        self.meta_proj = None
        if num_meta_features > 0:
            self.meta_proj = nn.Sequential(
                # BƯỚC 1: Ép 11 features về cùng biên độ N(0,1)
                nn.BatchNorm1d(num_meta_features), 
                
                # BƯỚC 2: Chiếu qua lớp Linear để học mối tương quan
                nn.Linear(num_meta_features, meta_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            out_dim = out_dim + meta_hidden_size
        self.fc = nn.Linear(out_dim, num_classes)

        

    def forward(self, input_ids, meta_features=None):
        emb = self.embedding(input_ids)
        outputs, _ = self.gru(emb)

        mask = (input_ids != 0).unsqueeze(-1).float()
        summed = (outputs * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / lengths

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

        # preds = torch.argmax(logits, dim=1)
        # BỔ SUNG THRESHOLD MOVING
        probs = torch.softmax(logits, dim=1)
        preds = []
        for p in probs:
            # p[0] là Sạch, p[1] là Xúc phạm, p[2] là Thù địch
            # Nếu xác suất Thù địch > 0.25 -> Chọn Thù địch (không cần đợi tới > 0.33)
            if p[2] > 0.25:
                preds.append(2)
            # Nếu xác suất Xúc phạm > 0.25 -> Chọn Xúc phạm
            elif p[1] > 0.25:
                preds.append(1)
            else:
                preds.append(torch.argmax(p).item()) # Quay về argmax nếu ko đạt ngưỡng
                
        preds = torch.tensor(preds)
        total_loss += loss.item() * x.size(0)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        if batch_idx % 50 == 0 or batch_idx == len(loader):
            debug(f"Eval batch {batch_idx}/{len(loader)}")

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
    parser.add_argument(
        "--fasttext_path",
        type=str,
        default=os.path.join(get_default_data_dir(), "cc.vi.300.vec"),
        help="Đường dẫn đến file FastText (mặc định trong dataset-vihsd)",
    )
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--batch_size", type=int, default=100)
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
    train_texts, train_meta, train_labels_raw = load_split(
        os.path.join(args.data_dir, args.train_file),
        text_column=args.text_column,
        feature_columns=feature_columns,
    )
    # === BỔ SUNG KỸ THUẬT OVERSAMPLING CHO TẬP TRAIN ===
    debug("Thực hiện Oversampling (Nhân bản dữ liệu) để cân bằng nhãn...")
    
    # Tách dữ liệu theo nhãn
    clean_indices = [i for i, lbl in enumerate(train_labels_raw) if lbl == 0]     # Nhãn 0 (Đa số)
    offensive_indices = [i for i, lbl in enumerate(train_labels_raw) if lbl == 1] # Nhãn 1 (Thiểu số)
    hate_indices = [i for i, lbl in enumerate(train_labels_raw) if lbl == 2]      # Nhãn 2 (Thiểu số)
    
    # Tính số lần cần nhân bản để cân bằng tương đối (không cần bằng 100%, chỉ cần xấp xỉ 50-70%)
    # Ví dụ: Nhân bản OFFENSIVE lên 5 lần, HATE lên 3 lần
    import random
    augmented_indices = clean_indices.copy()
    augmented_indices.extend(offensive_indices * 5) # Nhân 5 lần dữ liệu Offensive
    augmented_indices.extend(hate_indices * 3)      # Nhân 3 lần dữ liệu Hate
    
    # Xáo trộn lại tập dữ liệu
    random.shuffle(augmented_indices)
    
    # Áp dụng lại vào mảng train
    train_texts = [train_texts[i] for i in augmented_indices]
    train_meta = np.array([train_meta[i] for i in augmented_indices])
    train_labels_raw = [train_labels_raw[i] for i in augmented_indices]
    
    debug(f"Kích thước tập Train sau Oversampling: {len(train_texts)} mẫu.")
    # ===================================================
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

    debug("Building vocabulary")
    vocab = build_vocab(train_texts, max_vocab_size=args.max_vocab_size, min_freq=args.min_freq)
    debug(f"Vocabulary size: {len(vocab)}")

    pretrained_embeddings = None
    fasttext_path = args.fasttext_path
    if not os.path.isabs(fasttext_path):
        candidate_fasttext_path = os.path.join(args.data_dir, fasttext_path)
        if os.path.exists(candidate_fasttext_path):
            fasttext_path = candidate_fasttext_path

    if os.path.exists(fasttext_path):
        # fasttext_dict = load_fasttext_vectors(fasttext_path)
        fasttext_dict = load_fasttext_vectors(fasttext_path, vocab)
        pretrained_embeddings = build_embedding_matrix(vocab, fasttext_dict, embed_dim=args.embed_dim)
        fasttext_dim = len(next(iter(fasttext_dict.values())))
        if args.embed_dim != fasttext_dim:
            debug(
                f"embed_dim={args.embed_dim} không khớp FastText dim={fasttext_dim}. "
                f"Tự động đồng bộ embed_dim -> {fasttext_dim}."
            )
            args.embed_dim = fasttext_dim
    else:
        print(f"[CẢNH BÁO] Không tìm thấy file {fasttext_path}. Mô hình sẽ khởi tạo nhúng ngẫu nhiên!")

    debug("Building datasets and dataloaders")
    train_ds = TextDataset(train_texts, train_meta, train_labels, vocab, args.max_len)
    dev_ds = TextDataset(dev_texts, dev_meta, dev_labels, vocab, args.max_len)
    test_ds = TextDataset(test_texts, test_meta, test_labels, vocab, args.max_len)

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
    criterion = nn.CrossEntropyLoss()

    # BỔ SUNG SCHEDULER: Giảm LR đi một nửa (factor=0.5) nếu dev_f1 không tăng sau 2 epoch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )

    # ----------------------------------------------------------
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
        # Cập nhật Scheduler
        scheduler.step(dev_f1)

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
