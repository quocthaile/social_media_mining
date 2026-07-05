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
]


def debug(msg: str) -> None:
    print(f"[DEBUG][TextCNN] {msg}")


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

def load_fasttext_vectors(vec_path: str):
    """
    Tải file FastText (.vec) vào bộ nhớ dưới dạng dictionary.
    """
    debug(f"Đang tải FastText vectors từ {vec_path} (Quá trình này có thể mất vài phút)...")
    embeddings_dict = {}
    with open(vec_path, 'r', encoding='utf-8') as f:
        # File .vec thường có dòng đầu tiên chứa: [số_lượng_từ] [số_chiều]
        first_line = f.readline().split()
        if len(first_line) == 2:
            pass # Bỏ qua dòng đầu
        else:
            f.seek(0)
            
        for line in f:
            values = line.rstrip().split(' ')
            word = values[0]
            # FastText mặc định là chữ in thường, nên ta đồng bộ bằng .lower()
            word_lower = word.lower() 
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_dict[word_lower] = vector
            
    debug(f"Đã tải thành công {len(embeddings_dict)} vector từ vựng.")
    return embeddings_dict


def build_embedding_matrix(vocab, embeddings_dict, embed_dim=300):
    """
    Ánh xạ FastText vectors vào cấu trúc từ điển (vocab) của model.
    """
    debug("Đang khởi tạo Ma trận Embedding cho mô hình...")
    vocab_size = len(vocab)
    
    # Khởi tạo ma trận ngẫu nhiên (Normal Distribution) cho các từ OOV (Out-of-Vocab)
    # Các từ lóng/teencode mới không có trong FastText sẽ lấy vector ngẫu nhiên này để học tiếp.
    embedding_matrix = np.random.normal(scale=0.1, size=(vocab_size, embed_dim))
    
    hits = 0
    for word, idx in vocab.items():
        if word == PAD_TOKEN:
            embedding_matrix[idx] = np.zeros(embed_dim) # PAD luôn bằng 0
        elif word in embeddings_dict:
            embedding_matrix[idx] = embeddings_dict[word]
            hits += 1
            
    debug(f"Tỷ lệ khớp FastText: {hits}/{vocab_size} từ ({(hits/vocab_size)*100:.2f}%).")
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


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_classes,
        num_filters=128,
        kernel_sizes=(3, 4, 5),
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
            
        # Sửa lỗi chuẩn hóa BatchNorm cho meta-features đã phân tích trước đó
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes])
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        classifier_in_dim = num_filters * len(kernel_sizes)
        # self.meta_proj = None
        # if num_meta_features > 0:
        #     self.meta_proj = nn.Sequential(
        #         nn.Linear(num_meta_features, meta_hidden_size),
        #         nn.ReLU(),
        #         nn.Dropout(dropout),
        #     )
        #     classifier_in_dim = classifier_in_dim + meta_hidden_size

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
            # (Đối với GRU, biến ở dòng dưới là out_dim thay vì classifier_in_dim)
            classifier_in_dim = classifier_in_dim + meta_hidden_size

        self.fc = nn.Linear(classifier_in_dim, num_classes)

    def forward(self, input_ids, meta_features=None):
        x = self.embedding(input_ids)
        x = x.transpose(1, 2)
        conv_outs = [torch.relu(conv(x)) for conv in self.convs]
        pooled = [torch.max(c, dim=2).values for c in conv_outs]
        z = torch.cat(pooled, dim=1)
        z = self.dropout(z)
        if self.meta_proj is not None and meta_features is not None:
            meta_repr = self.meta_proj(meta_features)
            z = torch.cat([z, meta_repr], dim=1)
        return self.fc(z)


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


def parse_kernel_sizes(s: str):
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def parse_feature_columns(raw: str):
    columns = [c.strip() for c in raw.split(",") if c.strip()]
    if not columns:
        raise ValueError("feature_columns must include at least one column")
    return columns


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate TextCNN")
    parser.add_argument("--data_dir", type=str, default=get_default_data_dir())
    parser.add_argument("--train_file", type=str, default="features_train.csv")
    parser.add_argument("--dev_file", type=str, default="features_dev.csv")
    parser.add_argument("--test_file", type=str, default="features_test.csv")
    parser.add_argument("--text_column", type=str, default=DEFAULT_TEXT_COLUMN)
    parser.add_argument("--feature_columns", type=str, default=",".join(DEFAULT_FEATURE_COLUMNS))

    parser.add_argument("--max_vocab_size", type=int, default=50000)
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=128)

    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument(
        "--fasttext_path",
        type=str,
        default=os.path.join(get_default_data_dir(), "cc.vi.300.vec"),
        help="Đường dẫn đến file FastText (mặc định trong dataset-vihsd)",
    )
    parser.add_argument("--num_filters", type=int, default=128)
    parser.add_argument("--kernel_sizes", type=str, default="3,4,5")
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
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "textcnn")),
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
    # TẢI VÀ XÂY DỰNG FASTTEXT EMBEDDING
    pretrained_embeddings = None
    fasttext_path = args.fasttext_path
    if not os.path.isabs(fasttext_path):
        candidate_fasttext_path = os.path.join(args.data_dir, fasttext_path)
        if os.path.exists(candidate_fasttext_path):
            fasttext_path = candidate_fasttext_path

    if os.path.exists(fasttext_path):
        fasttext_dict = load_fasttext_vectors(fasttext_path)
        pretrained_embeddings = build_embedding_matrix(vocab, fasttext_dict, embed_dim=args.embed_dim)
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
    kernel_sizes = parse_kernel_sizes(args.kernel_sizes)
    debug(f"Using device: {device}")
    debug(f"Kernel sizes: {kernel_sizes}")

    model = TextCNN(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        num_classes=len(id2label),
        num_filters=args.num_filters,
        kernel_sizes=kernel_sizes,
        dropout=args.dropout,
        padding_idx=vocab[PAD_TOKEN],
        num_meta_features=len(feature_columns),
        meta_hidden_size=args.meta_hidden_size,
        pretrained_embeddings=pretrained_embeddings
    ).to(device)
    debug("Model initialized")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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

    print("\n===== TextCNN Test Metrics =====")
    print_confusion_and_scores(test_true, test_pred, id2label)

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "best_textcnn.pt")
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
