import argparse
import os
import re
import unicodedata
import numpy as np
import pandas as pd

DEFAULT_INPUT_PREFIX = "preprocessed"
DEFAULT_OUTPUT_PREFIX = "features"

# BỘ 10 ĐẶC TRƯNG TỐI ƯU (THE OPTIMAL SET)
FEATURE_COLUMNS = [
    "feat_log_num_tokens",        # Phân biệt câu chửi đổng ngắn và bài viết công kích dài
    "feat_upper_ratio",           # Tín hiệu la hét, giận dữ (Capslock)
    "feat_emoji_density",         # Tín hiệu cảm xúc hình ảnh
    "feat_bad_word_density",      # Tín hiệu chửi tục / bạo lực
    "feat_aggressive_pronoun",    # Tín hiệu công kích cá nhân
    "feat_laugh_density",         # Tiếng cười cợt mỉa mai
    "feat_sarcastic_punct",       # Dấu câu tu từ mỉa mai
    "feat_scare_quotes",          # Dấu ngoặc kép nhại lại
    "feat_intensifier_words",     # Từ cường điệu mỉa mai
    "feat_elongated_ratio",       # Chữ kéo dài thể hiện sự bỡn cợt
]

EMOJI_TAG_PATTERN = re.compile(r"\bEMOJI_[A-Z_]+\b")
EMOJI_ALIAS_PATTERN = re.compile(r":\s*[a-z0-9_+\-]+\s*:", re.IGNORECASE)
ELONGATED_PATTERN = re.compile(r'(.)\1{2,}')
SARCASTIC_PUNCT_PATTERN = re.compile(r'(\?{2,})|(!{2,})|(\?!|\!\?)|(\.{3,})')
QUOTED_WORD_PATTERN = re.compile(r'["\'“‘]([^"\'”’\s]+)["\'”’]')

INTENSIFIER_SARCASTIC_LIST = [
    "quá cơ", "lắm cơ", "ghê", "cơ à", "hộ cái", "giùm cái", "quá chừng", "quá trời", "thế cơ", 
]

BAD_WORDS_LIST = [
    "lồn", "đéo", "địt", "đkm", "vcl", "cặc", "ngu", "chó", "đĩ", 
    "điếm", "đảng", "cộng sản", "phản động", "cc", "cđm", "vl", 
    "đm", "dkm", "đĩ điếm", "đĩ thoã", "coin card", "éo", "nham lon",
    "lủ chó", "nhảm lồn", "vãi lồn", "vãi cả lồn", "địt mẹ", "đụ", 
    "đụ má", "con card", "concard", "củ cặc", "xạo lồn", "tinh trùng", 
    "bê đê", "ml", "sml", "óc chó", "đực rựa", "đm", "cmm", "dcm",
    "nghiệt súc", "súc vật", "rác rưởi", "đáp cứt", "ngu học"
]

# ĐẶC TRƯNG MỚI: ĐẠI TỪ CÔNG KÍCH
AGGRESSIVE_PRONOUNS_LIST = [
    "mày", "tao", "chúng mày", "tụi mày", "bọn mày", "chúng nó", "tụi nó",
    "bọn", "lũ", "thằng", "con", "nó"
]

# Cập nhật Regex Pattern ưu tiên match từ dài trước
INTENSIFIER_PATTERN = re.compile(r'\b(?:' + '|'.join(map(re.escape, sorted(INTENSIFIER_SARCASTIC_LIST, key=len, reverse=True))) + r')\b', re.IGNORECASE)
BAD_WORDS_PATTERN = re.compile(r'\b(?:' + '|'.join(map(re.escape, sorted(BAD_WORDS_LIST, key=len, reverse=True))) + r')\b', re.IGNORECASE)
AGGRESSIVE_PRONOUN_PATTERN = re.compile(r'\b(?:' + '|'.join(map(re.escape, sorted(AGGRESSIVE_PRONOUNS_LIST, key=len, reverse=True))) + r')\b', re.IGNORECASE)


def debug(msg: str) -> None:
    print(f"[DEBUG][FeatureBuilder] {msg}")

def get_default_data_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset-vihsd"))

def normalize_whitespace(text: str) -> str:
    normalized = unicodedata.normalize("NFC", str(text)).strip()
    return re.sub(r"\s+", " ", normalized)

def count_emoji_features(text: str) -> int:
    return len(EMOJI_TAG_PATTERN.findall(text)) + len(EMOJI_ALIAS_PATTERN.findall(text))

def count_bad_words(text: str) -> int:
    clean_text = text.replace('_', ' ').lower()
    return len(BAD_WORDS_PATTERN.findall(clean_text))

def count_aggressive_pronouns(text: str) -> int:
    clean_text = text.replace('_', ' ').lower()
    return len(AGGRESSIVE_PRONOUN_PATTERN.findall(clean_text))

def count_laugh_signals(text: str) -> int:
    # Bắt chữ haha, hehe, kkk
    text_laugh = len(re.findall(r'(haha+|hehe+|hihi+|kkk+|hé hé|hô hô|há há)', text.lower()))
    # Bắt emoji cười cợt
    emoji_laugh = len(re.findall(r'(TEARS_OF_JOY|ROLLING_ON_THE_FLOOR|GRINNING_SQUINTING|SMIRKING)', text.upper()))
    # Bắt dấu ngoặc đóng lặp lại (=))) hoặc :)))
    paren_laugh = len(re.findall(r'(\={1,}\)+|\:{1,}\)+)', text))
    return text_laugh + emoji_laugh + paren_laugh

def count_elongated_words(text: str) -> int:
    return len(ELONGATED_PATTERN.findall(text))

def count_sarcastic_punctuation(text: str) -> int:
    return len(SARCASTIC_PUNCT_PATTERN.findall(text))

def count_scare_quotes(text: str) -> int:
    return len(QUOTED_WORD_PATTERN.findall(text))

def count_intensifier_words(text: str) -> int:
    clean_text = text.replace('_', ' ').lower()
    return len(INTENSIFIER_PATTERN.findall(clean_text))

def extract_feature_row(text: str) -> dict:
    tokens = [tok for tok in text.split(" ") if tok]
    num_tokens = len(tokens)
    num_chars = len(text)

    # Đếm ký tự để tính tỷ lệ
    alpha_count = sum(1 for ch in text if ch.isalpha())
    upper_count = sum(1 for ch in text if ch.isalpha() and ch.isupper())

    # Chạy các hàm trích xuất
    emoji_count = count_emoji_features(text)
    bad_word_count = count_bad_words(text)
    agg_pronoun_count = count_aggressive_pronouns(text)
    laugh_count = count_laugh_signals(text)
    sarcastic_punct_count = count_sarcastic_punctuation(text)
    scare_quote_count = count_scare_quotes(text)
    intensifier_count = count_intensifier_words(text)
    elongated_count = count_elongated_words(text)

    # Đóng gói đúng 10 đặc trưng
    return {
        "feat_log_num_tokens": float(np.log1p(num_tokens)),
        "feat_upper_ratio": float(upper_count / max(alpha_count, 1)),
        "feat_emoji_density": float(emoji_count / max(num_tokens, 1)),
        "feat_bad_word_density": float(bad_word_count / max(num_tokens, 1)),
        "feat_aggressive_pronoun": float(agg_pronoun_count / max(num_tokens, 1)),
        "feat_laugh_density": float(laugh_count / max(num_tokens, 1)),
        "feat_sarcastic_punct": float(sarcastic_punct_count / max(num_chars, 1)),
        "feat_scare_quotes": float(scare_quote_count / max(num_tokens, 1)),
        "feat_intensifier_words": float(intensifier_count / max(num_tokens, 1)),
        "feat_elongated_ratio": float(elongated_count / max(num_tokens, 1)),
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

    # Sắp xếp đúng trình tự cột
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