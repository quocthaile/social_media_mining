import argparse
import os
import re
import unicodedata
import numpy as np
import pandas as pd

DEFAULT_INPUT_PREFIX = "preprocessed"
DEFAULT_OUTPUT_PREFIX = "features"

# BỔ SUNG 3 ĐẶC TRƯNG MỚI VÀO PIPELINE
FEATURE_COLUMNS = [
    "feat_log_num_tokens",
    "feat_log_num_chars",
    "feat_avg_token_len",
    "feat_emoji_density",
    "feat_punct_density",
    "feat_upper_ratio",
    "feat_digit_ratio",
    "feat_bad_word_density",
    "feat_elongated_ratio",
    "feat_exclamation_density",
    "feat_allcaps_ratio",
    "feat_laugh_density",         # Tín hiệu cười cợt mỉa mai
    "feat_sarcasm_words",         # Tín hiệu khen ngợi giả tạo
    "feat_contrast_score",        # Mức độ tương phản ngữ nghĩa (Chìa khóa bắt Mỉa mai)
]

EMOJI_TAG_PATTERN = re.compile(r"\bEMOJI_[A-Z_]+\b")
EMOJI_ALIAS_PATTERN = re.compile(r":\s*[a-z0-9_+\-]+\s*:", re.IGNORECASE)
ELONGATED_PATTERN = re.compile(r'(.)\1{2,}')

BAD_WORDS_LIST = [
    "lồn", "đéo", "địt", "đkm", "vcl", "cặc", "ngu", "chó", "đĩ", 
    "điếm", "đảng", "cộng sản", "phản động", "cc", "cđm", "vl", 
    "đm", "dkm", "đĩ điếm", "đĩ thoã", "coin card", "éo", "nham lon",
    "lủ chó", "nhảm lồn", "vãi lồn", "vãi cả lồn", "địt mẹ", "đụ", 
    "đụ má", "con card", "concard", "củ cặc", "xạo lồn", "tinh trùng", 
    "bê đê", "ml", "sml", "óc chó", "đực rựa", "đm", "cmm", "dcm",
    "nghiệt súc", "súc vật", "rác rưởi", "đáp cứt", "ngu học"
]

POSITIVE_SARCASTIC_LIST = [
    "tuyệt vời", "giỏi", "đỉnh", "hay quá", "xuất sắc", "thông minh",
    "khen", "hoan hô", "tuyệt", "hảo", "nhất bạn", "số 1", "thiên tài",
    "đáng tuyên dương", "hảo hán", "đỉnh cao", "xuất chúng", "tốt đẹp"
]

BAD_WORDS_PATTERN = re.compile(r'\b(?:' + '|'.join(map(re.escape, BAD_WORDS_LIST)) + r')\b', re.IGNORECASE)
POSITIVE_PATTERN = re.compile(r'\b(?:' + '|'.join(map(re.escape, POSITIVE_SARCASTIC_LIST)) + r')\b', re.IGNORECASE)

def debug(msg: str) -> None:
    print(f"[DEBUG][FeatureBuilder] {msg}")

def get_default_data_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset-vihsd"))

def normalize_whitespace(text: str) -> str:
    normalized = unicodedata.normalize("NFC", str(text)).strip()
    return re.sub(r"\s+", " ", normalized)

def count_emoji_features(text: str) -> int:
    return len(EMOJI_TAG_PATTERN.findall(text)) + len(EMOJI_ALIAS_PATTERN.findall(text))

def count_punctuation(text: str) -> int:
    return sum(1 for ch in text if unicodedata.category(ch).startswith("P"))

def count_bad_words(text: str) -> int:
    clean_text = text.replace('_', ' ').lower()
    return len(BAD_WORDS_PATTERN.findall(clean_text))

def count_positive_words(text: str) -> int:
    clean_text = text.replace('_', ' ').lower()
    return len(POSITIVE_PATTERN.findall(clean_text))

def count_laugh_signals(text: str) -> int:
    # Bắt chữ haha, hehe, kkk
    text_laugh = len(re.findall(r'(haha+|hehe+|hihi+|kkk+|hé hé|hô hô|há há)', text.lower()))
    # Bắt emoji cười cợt (Face with tears of joy, smirking, rofl)
    emoji_laugh = len(re.findall(r'(TEARS_OF_JOY|ROLLING_ON_THE_FLOOR|GRINNING_SQUINTING|SMIRKING)', text.upper()))
    # Bắt dấu ngoặc đóng lặp lại (Biểu tượng =))) hoặc :))) rất phổ biến ở VN)
    paren_laugh = len(re.findall(r'(\={1,}\)+|\:{1,}\)+)', text))
    
    return text_laugh + emoji_laugh + paren_laugh

def count_elongated_words(text: str) -> int:
    return len(ELONGATED_PATTERN.findall(text))

def count_exclamation_question(text: str) -> int:
    return sum(1 for ch in text if ch in ('!', '?'))

def count_all_caps_words(text: str) -> int:
    tokens = text.split()
    return sum(1 for t in tokens if t.isupper() and len(t) > 1)

def extract_feature_row(text: str) -> dict:
    tokens = [tok for tok in text.split(" ") if tok]
    num_tokens = len(tokens)
    num_chars = len(text)

    avg_token_len = float(np.mean([len(tok) for tok in tokens])) if num_tokens > 0 else 0.0
    emoji_count = count_emoji_features(text)
    punct_count = count_punctuation(text)

    alpha_count = sum(1 for ch in text if ch.isalpha())
    upper_count = sum(1 for ch in text if ch.isalpha() and ch.isupper())
    digit_count = sum(1 for ch in text if ch.isdigit())

    bad_word_count = count_bad_words(text)
    elongated_count = count_elongated_words(text)
    exclamation_count = count_exclamation_question(text)
    allcaps_count = count_all_caps_words(text)
    
    pos_word_count = count_positive_words(text)
    laugh_count = count_laugh_signals(text)
    
    # HÀM TƯƠNG PHẢN (CHÌA KHÓA BẮT MỈA MAI)
    # Nếu câu vừa có Khen vừa có Chửi -> Sarcasm Score = 1, 2...
    # Nếu câu vừa có Khen vừa Cười cợt -> Sarcasm Score = 1, 2...
    contrast_score = min(pos_word_count, bad_word_count) + min(pos_word_count, laugh_count)

    return {
        "feat_log_num_tokens": float(np.log1p(num_tokens)),
        "feat_log_num_chars": float(np.log1p(num_chars)),
        "feat_avg_token_len": float(avg_token_len),
        "feat_emoji_density": float(emoji_count / max(num_tokens, 1)),
        "feat_punct_density": float(punct_count / max(num_chars, 1)),
        "feat_upper_ratio": float(upper_count / max(alpha_count, 1)),
        "feat_digit_ratio": float(digit_count / max(num_chars, 1)),
        "feat_bad_word_density": float(bad_word_count / max(num_tokens, 1)),
        "feat_elongated_ratio": float(elongated_count / max(num_tokens, 1)),
        "feat_exclamation_density": float(exclamation_count / max(num_chars, 1)),
        "feat_allcaps_ratio": float(allcaps_count / max(num_tokens, 1)),
        # 3 Đặc trưng mới
        "feat_laugh_density": float(laugh_count / max(num_tokens, 1)),
        "feat_sarcasm_words": float(pos_word_count / max(num_tokens, 1)),
        "feat_contrast_score": float(contrast_score),
    }

def build_one_split(input_path: str, output_path: str) -> None:
    debug(f"Reading: {input_path}")
    df = pd.read_csv(input_path)

    required_cols = {"clean_text", "label_id"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{input_path} must contain columns: {required_cols}")

    token_text = df["clean_text"].fillna("").astype(str).map(normalize_whitespace)
    phobert_segmented_text = token_text 
    multilingual_raw_text = token_text.map(lambda t: t.replace("_", " "))
    
    feature_df = token_text.map(extract_feature_row).apply(pd.Series)

    out_df = pd.DataFrame(
        {
            "tokens_text": phobert_segmented_text,
            "transformer_text": multilingual_raw_text,
        }
    )

    out_df = pd.concat([out_df, feature_df], axis=1)
    out_df["label_id"] = pd.to_numeric(df["label_id"], errors="raise").astype(int)

    ordered_cols = ["tokens_text", "transformer_text", *FEATURE_COLUMNS, "label_id"]
    out_df = out_df[ordered_cols]

    debug(f"Writing: {output_path}")
    out_df.to_csv(output_path, index=False)
    debug(f"Done: rows={len(out_df)}, cols={len(out_df.columns)}")

def process_dataset(data_dir: str, input_prefix: str, output_prefix: str) -> None:
    for split in ["train", "dev", "test"]:
        input_path = os.path.join(data_dir, f"{input_prefix}_{split}.csv")
        output_path = os.path.join(data_dir, f"{output_prefix}_{split}.csv")
        build_one_split(input_path, output_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=get_default_data_dir())
    parser.add_argument("--input_prefix", type=str, default=DEFAULT_INPUT_PREFIX)
    parser.add_argument("--output_prefix", type=str, default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    process_dataset(args.data_dir, args.input_prefix, args.output_prefix)

if __name__ == "__main__":
    main()