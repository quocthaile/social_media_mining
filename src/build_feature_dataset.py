import argparse
import os
import re
import unicodedata
import numpy as np
import pandas as pd

DEFAULT_INPUT_PREFIX = "preprocessed"
DEFAULT_OUTPUT_PREFIX = "features"

# ĐÃ TINH GỌN XUỐNG CÒN 5 ĐẶC TRƯNG CỐT LÕI
FEATURE_COLUMNS = [
    "feat_log_num_tokens",      # [Kích thước] Dấu hiệu của bài viết lập luận / thù ghét dài
    "feat_punct_density",       # [Cấu trúc] Dấu hiệu của sự phẫn nộ, hỗn loạn
    "feat_upper_ratio",         # [Cấu trúc] Dấu hiệu la hét (CAPSLOCK)
    "feat_bad_word_density",    # [Từ vựng] Tín hiệu chửi tục / thoá mạ (Đã đánh trọng số)
    "feat_aggressive_pronoun",  # [Từ vựng] Tín hiệu công kích cá nhân
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

AGGRESSIVE_PRONOUNS_LIST = [
    "mày", "tao", "chúng mày", "tụi mày", "bọn mày", "chúng nó", "tụi nó",
    "bọn", "lũ", "thằng", "con", "nó"
]

# =================================================================
# KỸ THUẬT: OBFUSCATION MAP & LEXICAL WEIGHTING
# =================================================================
OBFUSCATION_MAP = {
    # Tier 1
    r"đm": r"[đd]\s*[\.\*_\-]?\s*[k]?\s*m\b",
    r"cc": r"\bc\s*[\.\*_\-]?\s*c\b",
    r"vcl": r"v\s*[\.\*_\-]?\s*[ck]\s*[\.\*_\-]?\s*l\b",
    r"vl": r"\bv\s*[\.\*_\-]?\s*l\b",
    
    # Tier 2
    r"địt": r"[đd]\s*[\.\*_\-i1!ịĩỉ]\s*t\b",
    r"đụ": r"[đd]\s*[\.\*_\-uụũủ]\b",
    r"lồn": r"l\s*[\.\*_\-o0òõọôồốỗộơờớỡợ]\s*n\b",
    r"cặc": r"c\s*[\.\*_\-aăâặằắẵẳ]\s*[ck]\b",
    r"ngu": r"ng\s*[\.\*_\-u0úùũụủ]\b",
    r"chó": r"ch\s*[\.\*_\-o0òõọôồốỗộơờớỡợ]\b",
    
    # Tier 3
    r"súc vật": r"s\s*[\.\*_\-uúùũụủ]?\s*c\s+v\s*[\.\*_\-aâăậầấẫẩặằắẵẳ]?\s*t\b",
}

TIER_3_WORDS = ["đảng", "cộng sản", "phản động", "nghiệt súc", "súc vật", "đĩ điếm", "đĩ thoã", "rác rưởi", "đáp cứt"]
TIER_2_WORDS = ["lồn", "đéo", "địt", "cặc", "đĩ", "điếm", "đụ", "tinh trùng", "bê đê", "óc chó", "chó", "ngu"]

def build_weighted_patterns():
    patterns_with_weights = []
    
    def add_pattern(word_list, weight):
        sorted_words = sorted(word_list, key=len, reverse=True)
        for w in sorted_words:
            if w in OBFUSCATION_MAP:
                patterns_with_weights.append((re.compile(OBFUSCATION_MAP[w], re.IGNORECASE), weight))
            else:
                patterns_with_weights.append((re.compile(r'\b' + re.escape(w) + r'\b', re.IGNORECASE), weight))

    add_pattern(TIER_3_WORDS, 2.0)
    add_pattern(TIER_2_WORDS, 1.0)
    tier_1_list = [w for w in BAD_WORDS_LIST if w not in TIER_3_WORDS and w not in TIER_2_WORDS]
    add_pattern(tier_1_list, 0.5)
             
    return patterns_with_weights

WEIGHTED_BAD_WORDS_PATTERNS = build_weighted_patterns()
SORTED_AGGRESSIVE = sorted(AGGRESSIVE_PRONOUNS_LIST, key=len, reverse=True)
AGGRESSIVE_PRONOUN_PATTERN = re.compile(r'\b(?:' + '|'.join(map(re.escape, SORTED_AGGRESSIVE)) + r')\b', re.IGNORECASE)

def calculate_weighted_bad_words(text: str) -> float:
    total_score = 0.0
    matched_indices = set()
    for pattern, weight in WEIGHTED_BAD_WORDS_PATTERNS:
        for match in pattern.finditer(text):
            overlap = False
            for i in range(match.start(), match.end()):
                if i in matched_indices:
                    overlap = True
                    break
            if not overlap:
                total_score += weight
                for i in range(match.start(), match.end()):
                    matched_indices.add(i)
    return total_score

def normalize_whitespace(text: str) -> str:
    normalized = unicodedata.normalize("NFC", str(text)).strip()
    return re.sub(r"\s+", " ", normalized)

def compute_feature_signals(text: str) -> dict:
    clean_text = text.replace("_", " ").lower()
    return {
        "punct_count": sum(1 for ch in text if unicodedata.category(ch).startswith("P")),
        "upper_count": sum(1 for ch in text if ch.isalpha() and ch.isupper()),
        "bad_word_count": calculate_weighted_bad_words(clean_text),
        "agg_pronoun_count": len(AGGRESSIVE_PRONOUN_PATTERN.findall(clean_text)),
    }

def extract_feature_row(text: str) -> dict:
    text = str(text)
    tokens = [tok for tok in text.split(" ") if tok]
    num_tokens = len(tokens)
    num_chars = len(text)

    safe_tokens = max(num_tokens, 1)
    safe_chars = max(num_chars, 1)
    safe_alpha = max(sum(1 for ch in text if ch.isalpha()), 1)

    signals = compute_feature_signals(text)

    return {
        "feat_log_num_tokens": float(np.log1p(num_tokens)),
        "feat_punct_density": float(signals["punct_count"] / safe_chars),
        "feat_upper_ratio": float(signals["upper_count"] / safe_alpha),
        
        # Mật độ phi tuyến
        "feat_bad_word_density": float(np.log1p((signals["bad_word_count"] / safe_tokens) * 10)),
        "feat_aggressive_pronoun": float(np.log1p((signals["agg_pronoun_count"] / safe_tokens) * 10)),
    }

# --- CÁC HÀM build_one_split VÀ process_dataset GIỮ NGUYÊN NHƯ CŨ ---
def build_one_split(input_path: str, output_path: str) -> None:
    print(f"[DEBUG] Reading: {input_path}")
    df = pd.read_csv(input_path)

    required_cols = {"clean_text", "label_id"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{input_path} must contain columns: {required_cols}")

    token_text = df["clean_text"].fillna("").astype(str).map(normalize_whitespace)
    feature_df = token_text.map(extract_feature_row).apply(pd.Series)

    out_df = pd.DataFrame({
        "tokens_text": token_text,
        "transformer_text": token_text.map(lambda t: t.replace("_", " ")),
    })

    out_df = pd.concat([out_df, feature_df], axis=1)
    out_df["label_id"] = pd.to_numeric(df["label_id"], errors="raise").astype(int)

    ordered_cols = ["tokens_text", "transformer_text", *FEATURE_COLUMNS, "label_id"]
    out_df = out_df[ordered_cols]

    print(f"[DEBUG] Writing: {output_path}")
    out_df.to_csv(output_path, index=False)

def process_dataset(data_dir: str, input_prefix: str, output_prefix: str) -> None:
    for split in ["train", "dev", "test"]:
        input_path = os.path.join(data_dir, f"{input_prefix}_{split}.csv")
        output_path = os.path.join(data_dir, f"{output_prefix}_{split}.csv")
        build_one_split(input_path, output_path)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset-vihsd")))
    parser.add_argument("--input_prefix", type=str, default=DEFAULT_INPUT_PREFIX)
    parser.add_argument("--output_prefix", type=str, default=DEFAULT_OUTPUT_PREFIX)
    args = parser.parse_args()
    process_dataset(args.data_dir, args.input_prefix, args.output_prefix)

if __name__ == "__main__":
    main()