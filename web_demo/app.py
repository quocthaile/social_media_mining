import os
import sys
import re
import unicodedata
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Thêm thư mục gốc vào path để có thể import từ src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train_bamibert import TransformerWithMetaFeatures
from src.train_gru import GRUClassifier
from src.train_textcnn import TextCNN
from src.preprocessing import (
    unicode_normalization,
    remove_noise,
    extract_emoji_features,
    normalize_lengthened_words,
    replace_slang_and_abbreviations_cased,
    replace_compound_words,
    segment_and_remove_stopwords,
)
from src.build_feature_dataset import extract_feature_row, BAD_WORDS_LIST

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Thiết bị chạy
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# -------------------------------------------------------------
# DYNAMIC MODEL MANAGER
# -------------------------------------------------------------
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
label_names = {0: "Clean (Sạch)", 1: "Offensive (Xúc phạm)", 2: "Hate Speech (Thù địch)"}

class ModelManager:
    def __init__(self):
        self.active_model_key = None
        self.model = None
        self.tokenizer = None
        self.vocab = None
        self.checkpoint = None
        self.model_type = None  # 'transformer', 'gru', 'textcnn'
        self.id2label = {0: "0", 1: "1", 2: "2"}

    def scan_available_models(self):
        """Quét thư mục models để tìm các checkpoint đã được huấn luyện."""
        available = {}
        if not os.path.exists(MODELS_DIR):
            return available

        for name in os.listdir(MODELS_DIR):
            sub_dir = os.path.join(MODELS_DIR, name)
            if not os.path.isdir(sub_dir):
                continue
            
            # Kiểm tra Transformer
            transformer_path = os.path.join(sub_dir, "best_transformer_with_features.pt")
            if os.path.exists(transformer_path):
                available[f"transformer-{name}"] = {
                    "name": f"Transformer ({name.upper()})",
                    "path": transformer_path,
                    "type": "transformer"
                }

            # Kiểm tra GRU
            gru_path = os.path.join(sub_dir, "best_gru.pt")
            if os.path.exists(gru_path):
                available[f"gru-{name}"] = {
                    "name": f"GRU ({name.upper()})",
                    "path": gru_path,
                    "type": "gru"
                }

            # Kiểm tra TextCNN
            textcnn_path = os.path.join(sub_dir, "best_textcnn.pt")
            if os.path.exists(textcnn_path):
                available[f"textcnn-{name}"] = {
                    "name": f"TextCNN ({name.upper()})",
                    "path": textcnn_path,
                    "type": "textcnn"
                }
        return available

    def load_model(self, model_key):
        """Tải mô hình đã chọn vào RAM/GPU."""
        available = self.scan_available_models()
        if model_key not in available:
            raise ValueError(f"Model key '{model_key}' không khả dụng.")

        self.cached_metrics = None
        meta = available[model_key]
        path = meta["path"]
        self.checkpoint_path = path
        mtype = meta["type"]

        print(f"[INFO] Đang tải mô hình {meta['name']} từ {path}...")
        checkpoint = torch.load(path, map_location=torch.device("cpu"))

        # Giải phóng mô hình cũ khỏi bộ nhớ
        self.model = None
        self.tokenizer = None
        self.vocab = None

        if mtype == "transformer":
            # Khởi tạo Transformer
            model = TransformerWithMetaFeatures(
                model_name=checkpoint["model_name"],
                num_labels=checkpoint["num_labels"],
                num_meta_features=len(checkpoint["feature_columns"]),
                feature_hidden_size=checkpoint["feature_hidden_size"],
                dropout=checkpoint["feature_dropout"],
                id2label=checkpoint["id2label"],
                label2id=checkpoint["label2id"],
                class_weights=None,
            )
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            model.eval().to(device)

            # Tải tokenizer tương ứng
            model_dir = os.path.dirname(path)
            if "bamibert" in checkpoint["model_name"].lower():
                tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
            else:
                use_fast = False if "phobert" in checkpoint["model_name"].lower() else True
                tokenizer = AutoTokenizer.from_pretrained(checkpoint["model_name"], use_fast=use_fast)

            self.model = model
            self.tokenizer = tokenizer
            self.model_type = "transformer"

        elif mtype == "gru":
            # Khởi tạo GRU
            args = checkpoint["args"]
            vocab = checkpoint["vocab"]
            model = GRUClassifier(
                vocab_size=len(vocab),
                embed_dim=args.get("embed_dim", 300),
                hidden_size=args.get("hidden_size", 128),
                num_classes=len(checkpoint["id2label"]),
                num_layers=args.get("num_layers", 1),
                bidirectional=args.get("bidirectional", True),
                dropout=args.get("dropout", 0.3),
                padding_idx=vocab.get("<pad>", 0),
                num_meta_features=len(checkpoint["feature_columns"]),
                meta_hidden_size=args.get("meta_hidden_size", 32),
                pretrained_embeddings=None,
            )
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            model.eval().to(device)

            self.model = model
            self.vocab = vocab
            self.model_type = "gru"

        elif mtype == "textcnn":
            # Khởi tạo TextCNN
            args = checkpoint["args"]
            vocab = checkpoint["vocab"]
            
            kernel_sizes = args.get("kernel_sizes", "3,4,5")
            if isinstance(kernel_sizes, str):
                kernel_sizes = tuple(int(x.strip()) for x in kernel_sizes.split(",") if x.strip())

            model = TextCNN(
                vocab_size=len(vocab),
                embed_dim=args.get("embed_dim", 300),
                num_classes=len(checkpoint["id2label"]),
                num_filters=args.get("num_filters", 128),
                kernel_sizes=kernel_sizes,
                dropout=args.get("dropout", 0.3),
                padding_idx=vocab.get("<pad>", 0),
                num_meta_features=len(checkpoint["feature_columns"]),
                meta_hidden_size=args.get("meta_hidden_size", 32),
                pretrained_embeddings=None,
            )
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            model.eval().to(device)

            self.model = model
            self.vocab = vocab
            self.model_type = "textcnn"

        self.checkpoint = checkpoint
        self.active_model_key = model_key
        self.id2label = {int(k): str(v) for k, v in checkpoint["id2label"].items()}
        print(f"[INFO] Nạp mô hình {meta['name']} thành công lên: {device}")

    def predict(self, text):
        """Suy luận (Inference) dựa vào mô hình đang hoạt động."""
        if not self.model:
            return None

        # 1. Tiền xử lý
        clean_text = unicode_normalization(text)
        clean_text = remove_noise(clean_text)
        clean_text = extract_emoji_features(clean_text)
        clean_text = normalize_lengthened_words(clean_text)
        clean_text = replace_slang_and_abbreviations_cased(clean_text)
        clean_text = replace_compound_words(clean_text)
        
        # RNN/CNN thường loại bỏ stopwords, Transformer giữ lại stopwords
        remove_sw = True if self.model_type in ["gru", "textcnn"] else False
        clean_text = segment_and_remove_stopwords(clean_text, remove_stopwords=remove_sw).strip()

        # Trích xuất đặc trưng meta
        meta_features = extract_feature_row(clean_text)
        feature_columns = self.checkpoint["feature_columns"]
        meta_features_values = [meta_features[col] for col in feature_columns]
        meta_features_tensor = torch.tensor([meta_features_values], dtype=torch.float, device=device)

        # 2. Xử lý Tokenize dựa theo Model Type
        if self.model_type == "transformer":
            # Tokenize cho Transformer
            transformer_text = clean_text.replace("_", " ")
            inputs = self.tokenizer(
                transformer_text,
                truncation=True,
                padding="max_length",
                max_length=self.checkpoint["args"].get("max_len", 128),
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    token_type_ids=inputs.get("token_type_ids"),
                    meta_features=meta_features_tensor,
                )
                logits = outputs.logits
        else:
            # Tokenize cho GRU/TextCNN
            tokens = clean_text.split()
            max_len = self.checkpoint["args"].get("max_len", 128)
            pad_idx = self.vocab.get("<pad>", 0)
            unk_idx = self.vocab.get("<unk>", 1)
            
            token_ids = [self.vocab.get(t, unk_idx) for t in tokens[:max_len]]
            if len(token_ids) < max_len:
                token_ids += [pad_idx] * (max_len - len(token_ids))
            
            input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

            with torch.no_grad():
                logits = self.model(input_tensor, meta_features_tensor)

        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # 3. Dịch chuyển ngưỡng quyết định (Threshold moving) nếu checkpoint cấu hình bật
        use_threshold_moving = self.checkpoint.get("args", {}).get("use_threshold_moving", True)
        if use_threshold_moving:
            if probs[2] > 0.25:
                pred_label_id = 2
            elif probs[1] > 0.25:
                pred_label_id = 1
            else:
                pred_label_id = int(np.argmax(probs))
        else:
            pred_label_id = int(np.argmax(probs))

        # Tìm và highlight từ lóng/tục
        highlighted_text, detected_bad_words = highlight_bad_words(text)

        return {
            "raw_text": text,
            "clean_text": clean_text,
            "highlighted_text": highlighted_text,
            "detected_bad_words": detected_bad_words,
            "label_id": pred_label_id,
            "label_name": label_names[pred_label_id],
            "probs": {
                "clean": float(probs[0]),
                "offensive": float(probs[1]),
                "hate": float(probs[2]),
            },
            "meta_features": {k: float(v) for k, v in meta_features.items()},
        }

    def get_test_metrics(self):
        """Chạy đánh giá trên tập test để tính toán độ chính xác và Confusion Matrix thực tế (có lưu cache ra file JSON)."""
        if not self.model or not hasattr(self, "checkpoint_path"):
            return None
        
        if hasattr(self, "cached_metrics") and self.cached_metrics is not None:
            return self.cached_metrics

        # Thử đọc cache từ file
        model_dir = os.path.dirname(self.checkpoint_path)
        cache_file = os.path.join(model_dir, "eval_results.json")
        if os.path.exists(cache_file):
            try:
                import json
                with open(cache_file, "r", encoding="utf-8") as f:
                    self.cached_metrics = json.load(f)
                print(f"[INFO] Tải thông số kiểm thử thành công từ tệp cache: {cache_file}")
                return self.cached_metrics
            except Exception as e:
                print(f"[WARNING] Lỗi đọc file cache metrics: {e}")

        test_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset-vihsd", "features_test.csv"))
        if not os.path.exists(test_file):
            print(f"[WARNING] Không tìm thấy file test tại {test_file}")
            return None
            
        try:
            print(f"[INFO] Bắt đầu tính toán metrics động trên tập test cho {self.active_model_key}...")
            df = pd.read_csv(test_file)
            
            # Lấy đúng text column từ checkpoint hoặc đoán cột
            text_column = self.checkpoint.get("text_column", "transformer_text")
            if text_column not in df.columns:
                text_column = "transformer_text" if "transformer_text" in df.columns else df.columns[0]
                
            texts = df[text_column].fillna("").astype(str).tolist()
            labels = df["label_id"].astype(int).tolist()
            
            y_pred = []
            batch_size = 128
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                if self.model_type == "transformer":
                    transformer_texts = [t.replace("_", " ") for t in batch_texts]
                    inputs = self.tokenizer(
                        transformer_texts,
                        truncation=True,
                        padding="max_length",
                        max_length=self.checkpoint["args"].get("max_len", 128),
                        return_tensors="pt",
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    batch_meta = []
                    for t in batch_texts:
                        features = extract_feature_row(t)
                        batch_meta.append([features[col] for col in self.checkpoint["feature_columns"]])
                    meta_tensor = torch.tensor(batch_meta, dtype=torch.float, device=device)
                    
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs.get("attention_mask"),
                            token_type_ids=inputs.get("token_type_ids"),
                            meta_features=meta_tensor,
                        )
                        logits = outputs.logits
                else:
                    batch_tokens = []
                    batch_meta = []
                    max_len = self.checkpoint["args"].get("max_len", 128)
                    pad_idx = self.vocab.get("<pad>", 0)
                    unk_idx = self.vocab.get("<unk>", 1)
                    
                    for t in batch_texts:
                        # Tiền xử lý tương tự train
                        t_clean = segment_and_remove_stopwords(t, remove_stopwords=True)
                        tokens = t_clean.split()
                        token_ids = [self.vocab.get(tok, unk_idx) for tok in tokens[:max_len]]
                        if len(token_ids) < max_len:
                            token_ids += [pad_idx] * (max_len - len(token_ids))
                        batch_tokens.append(token_ids)
                        
                        features = extract_feature_row(t_clean)
                        batch_meta.append([features[col] for col in self.checkpoint["feature_columns"]])
                        
                    input_tensor = torch.tensor(batch_tokens, dtype=torch.long, device=device)
                    meta_tensor = torch.tensor(batch_meta, dtype=torch.float, device=device)
                    
                    with torch.no_grad():
                        logits = self.model(input_tensor, meta_tensor)
                        
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                use_threshold_moving = self.checkpoint.get("args", {}).get("use_threshold_moving", True)
                for p in probs:
                    if use_threshold_moving:
                        if p[2] > 0.25:
                            y_pred.append(2)
                        elif p[1] > 0.25:
                            y_pred.append(1)
                        else:
                            y_pred.append(int(np.argmax(p)))
                    else:
                        y_pred.append(int(np.argmax(p)))
            
            acc = accuracy_score(labels, y_pred)
            f1 = f1_score(labels, y_pred, average="macro")
            cm = confusion_matrix(labels, y_pred, labels=[0, 1, 2]).tolist()
            
            self.cached_metrics = {
                "accuracy": float(acc),
                "f1_macro": float(f1),
                "confusion_matrix": cm
            }
            
            # Ghi đè vào tệp cache
            try:
                import json
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(self.cached_metrics, f, ensure_ascii=False, indent=2)
                print(f"[INFO] Đã lưu cache thông số đánh giá vào: {cache_file}")
            except Exception as e:
                print(f"[WARNING] Không thể ghi cache file: {e}")
                
            print(f"[INFO] Tính toán metrics xong! Accuracy: {acc:.4f}, F1-macro: {f1:.4f}")
            return self.cached_metrics
            
        except Exception as e:
            print(f"[ERROR] Lỗi tính toán metrics động: {e}")
            return None

manager = ModelManager()

# Khởi động app: tự động tải mô hình đầu tiên tìm thấy (thường là BamiBERT)
try:
    available_models = manager.scan_available_models()
    if available_models:
        # Ưu tiên load BamiBERT trước
        default_key = next((k for k in available_models if "bamibert" in k), next(iter(available_models)))
        manager.load_model(default_key)
    else:
        print("[WARNING] Không tìm thấy bất kỳ mô hình nào được huấn luyện trong models/!")
except Exception as e:
    print(f"[ERROR] Không thể khởi động mô hình mặc định: {e}")

# Helper highlight
BAD_WORDS_PATTERN = re.compile(
    r"\b(?:" + "|".join(map(re.escape, BAD_WORDS_LIST)) + r")\b", re.IGNORECASE
)

def highlight_bad_words(text: str) -> tuple[str, list[str]]:
    matches = list(set(BAD_WORDS_PATTERN.findall(text)))
    highlighted = text
    if matches:
        for match in sorted(matches, key=len, reverse=True):
            highlighted = re.sub(
                r"\b(" + re.escape(match) + r")\b",
                r"<mark class='toxic-term'>\1</mark>",
                highlighted,
                flags=re.IGNORECASE,
            )
    return highlighted, matches

# -------------------------------------------------------------
# CÁC ROUTE API CỦA FLASK
# -------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/models")
def api_models():
    """Lấy danh sách các mô hình đang có sẵn."""
    available = manager.scan_available_models()
    model_list = [{"key": k, "name": v["name"]} for k, v in available.items()]
    return jsonify({
        "models": model_list,
        "active_model": manager.active_model_key
    })

@app.route("/api/select_model", methods=["POST"])
def api_select_model():
    """Thay đổi mô hình hoạt động."""
    data = request.get_json() or {}
    model_key = data.get("model_key", "")
    if not model_key:
        return jsonify({"error": "Thiếu model_key"}), 400

    try:
        manager.load_model(model_key)
        return jsonify({"success": True, "active_model": manager.active_model_key})
    except Exception as e:
        return jsonify({"error": f"Lỗi tải mô hình: {str(e)}"}), 500

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not manager.model:
        return jsonify({"error": "Không có mô hình nào đang hoạt động"}), 500

    data = request.get_json() or {}
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "Nội dung bình luận trống"}), 400

    result = manager.predict(text)
    return jsonify(result)

@app.route("/api/predict_batch", methods=["POST"])
def api_predict_batch():
    if not manager.model:
        return jsonify({"error": "Không có mô hình nào đang hoạt động"}), 500

    if "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Không có file nào được chọn"}), 400

        if file and file.filename.endswith(".csv"):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            try:
                df = pd.read_csv(filepath)
                text_col = None
                for col in ["free_text", "clean_text", "text", "comment", "content"]:
                    if col in df.columns:
                        text_col = col
                        break

                if text_col is None:
                    text_col = df.columns[0]

                results = []
                for val in df[text_col].fillna("").astype(str).tolist():
                    res = manager.predict(val)
                    if res:
                        results.append({
                            "text": val,
                            "label_name": res["label_name"],
                            "label_id": res["label_id"],
                            "prob_clean": res["probs"]["clean"],
                            "prob_offensive": res["probs"]["offensive"],
                            "prob_hate": res["probs"]["hate"],
                        })

                out_filename = "predicted_" + filename
                out_filepath = os.path.join(app.config["UPLOAD_FOLDER"], out_filename)
                
                out_df = df.copy()
                out_df["predicted_label"] = [r["label_name"] for r in results]
                out_df["prob_clean"] = [r["prob_clean"] for r in results]
                out_df["prob_offensive"] = [r["prob_offensive"] for r in results]
                out_df["prob_hate"] = [r["prob_hate"] for r in results]
                out_df.to_csv(out_filepath, index=False, encoding="utf-8-sig")

                return jsonify({
                    "success": True,
                    "results": results,
                    "download_url": f"/download/{out_filename}",
                })
            except Exception as e:
                return jsonify({"error": f"Lỗi xử lý file: {str(e)}"}), 500

    data = request.get_json() or {}
    texts = data.get("texts", [])
    if not texts:
        return jsonify({"error": "Danh sách bình luận trống"}), 400

    results = []
    for text in texts:
        res = manager.predict(text)
        if res:
            results.append(res)
    return jsonify({"results": results})

@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=True)

@app.route("/api/stats")
def api_stats():
    if not manager.checkpoint:
        return jsonify({"error": "Không có mô hình nào hoạt động"}), 500

    checkpoint = manager.checkpoint
    args = checkpoint.get("args", {})
    
    # Tính toán metrics động trên tập test
    metrics = manager.get_test_metrics()
    
    return jsonify({
        "model_name": checkpoint.get("model_name", manager.active_model_key),
        "feature_columns": checkpoint.get("feature_columns", []),
        "max_len": args.get("max_len", 128),
        "epochs": args.get("epochs", 3),
        "batch_size": args.get("batch_size", 16),
        "lr": args.get("lr", 2e-5),
        "id2label": manager.id2label,
        "model_type": manager.model_type,
        "metrics": metrics
    })

if __name__ == "__main__":
    print("[INFO] Đang khởi động Flask Web Server trên http://127.0.0.1:5001...")
    app.run(host="127.0.0.1", port=5001, debug=True)
