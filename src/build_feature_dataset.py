import argparse
import os
import re
import unicodedata
import numpy as np
import pandas as pd

DEFAULT_INPUT_PREFIX = "preprocessed"
DEFAULT_OUTPUT_PREFIX = "features"

FEATURE_COLUMNS = [
    "feat_log_num_tokens",              # ĐẶC TRƯNG: Logarithm của số lượng token
    "feat_log_num_chars",               # ĐẶC TRƯNG: Logarithm của số lượng token
    "feat_avg_token_len",               # ĐẶC TRƯNG: Độ dài trung bình của token
    "feat_emoji_density",
    "feat_punct_density",
    "feat_upper_ratio",                 # ĐẶC TRƯNG: Tỉ lệ chữ hoa
    "feat_bad_word_density",            # ĐẶC TRƯNG: Tỉ lệ từ tục tĩu
    "feat_exclamation_density",         # ĐẶC TRƯNG: Tỉ lệ dấu chấm than
    "feat_allcaps_ratio",               # ĐẶC TRƯNG: Tỉ lệ chữ hoa toàn bộ
    "feat_laugh_density",               # Tín hiệu cười cợt mỉa mai
    "feat_aggressive_pronoun",          # ĐẶC TRƯNG: Đại từ công kích
    "feat_sarcastic_punct",             # ĐẶC TRƯNG: Dấu câu mỉa mai
    "feat_scare_quotes",                # ĐẶC TRƯNG: Dấu ngoặc kép mỉa mai
    "feat_intensifier_words",           # ĐẶC TRƯNG: Từ cường điệu mỉa mai
]

EMOJI_TAG_PATTERN = re.compile(r"\bEMOJI_[A-Z_]+\b")
EMOJI_ALIAS_PATTERN = re.compile(r":\s*[a-z0-9_+\-]+\s*:", re.IGNORECASE)
ELONGATED_PATTERN = re.compile(r'(.)\1{2,}')
SARCASTIC_PUNCT_PATTERN = re.compile(r'(\?{2,})|(!{2,})|(\?!|\!\?)|(\.{3,})')
QUOTED_WORD_PATTERN = re.compile(r'["\'“‘]([^"\'”’\s]+)["\'”’]')

BAD_WORDS_LIST = [
    "lồn", "đéo", "địt", "đkm", "vcl", "cặc", "ngu", "chó", "đĩ", 
    "điếm", "đảng", "cộng sản", "phản động", "cc", "cđm", "vl", 
    "đm", "dkm", "đĩ điếm", "đĩ thoã", "coin card", "éo", "nham lon",
    "lủ chó", "nhảm lồn", "vãi lồn", "vãi cả lồn", "địt mẹ", "đụ", 
    "đụ má", "con card", "concard", "củ cặc", "xạo lồn", "tinh trùng", 
    "bê đê", "ml", "sml", "óc chó", "đực rựa", "đm", "cmm", "dcm",
    "nghiệt súc", "súc vật", "rác rưởi", "đáp cứt", "ngu học"
]

# ĐẶC TRƯNG MỚI: DANH SÁCH ĐẠI TỪ CÔNG KÍCH / CHIA PHE PHÁI
AGGRESSIVE_PRONOUNS_LIST = [
    "mày", "tao", "chúng mày", "tụi mày", "bọn mày", "chúng nó", "tụi nó",
    "bọn", "lũ", "thằng", "con", "nó"
]

INTENSIFIER_SARCASTIC_LIST = [
    "quá cơ", "lắm cơ", "ghê", "cơ à", "hộ cái", "giùm cái", "quá chừng", "quá trời", "thế cơ", "chứ", ":v", ":V"
]

POSITIVE_SARCASTIC_LIST = [
    "tuyệt vời", "giỏi", "đỉnh", "hay quá", "xuất sắc", "thông minh",
    "khen", "hoan hô", "tuyệt", "hảo", "nhất bạn", "số 1", "thiên tài",
    "đáng tuyên dương", "hảo hán", "đỉnh cao", "xuất chúng", "tốt đẹp"
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

# Phân cấp từ vựng
TIER_3_WORDS = ["đảng", "cộng sản", "phản động", "nghiệt súc", "súc vật", "đĩ điếm", "đĩ thoã", "rác rưởi", "đáp cứt"]
TIER_2_WORDS = ["lồn", "đéo", "địt", "cặc", "đĩ", "điếm", "đụ", "tinh trùng", "bê đê", "óc chó", "chó", "ngu"]
# Tier 1 là các từ còn lại trong BAD_WORDS_LIST

def build_weighted_patterns():
    """Biên dịch danh sách Regex đi kèm trọng số, sắp xếp theo từ dài xuống ngắn."""
    patterns_with_weights = []
    
    # Hàm con hỗ trợ đẩy regex
    def add_pattern(word_list, weight):
        # Ưu tiên các chuỗi dài (như "vãi cả lồn") để tránh bị bắt nhầm bởi chuỗi ngắn (như "lồn")
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

def calculate_weighted_bad_words(text: str) -> float:
    """Tính tổng điểm (Toxic Score) dựa trên trọng số từ vựng, tránh đếm trùng lắp (overlap)."""
    total_score = 0.0
    matched_indices = set()
    
    # Quét qua từng pattern (đã được sắp xếp chuỗi dài trước)
    for pattern, weight in WEIGHTED_BAD_WORDS_PATTERNS:
        for match in pattern.finditer(text):
            # Kiểm tra xem từ này đã bị khớp bởi regex dài hơn trước đó chưa
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

# 2. Xây dựng danh sách các biểu thức lách luật
obf_patterns = list(OBFUSCATION_MAP.values())

# 3. Lọc các từ không nằm trong Obfuscation Map (để khớp chính xác)
# Tránh trùng lặp với các Regex mạnh đã định nghĩa ở trên
EXCLUDED_ANCHORS = ["đm", "địt", "đụ", "lồn", "cặc", "cc", "vcl", "vl", "súc vật", "ngu", "chó"]
exact_words = [w for w in BAD_WORDS_LIST if w not in EXCLUDED_ANCHORS]

# 4. Sắp xếp các từ khóa khớp chính xác theo chiều dài để tham lam (longest match)
exact_patterns = [r'\b' + re.escape(w) + r'\b' for w in sorted(exact_words, key=len, reverse=True)]

# 5. Hợp nhất toàn bộ Regex
FINAL_BAD_WORDS_PATTERNS = obf_patterns + exact_patterns
SORTED_BAD_WORDS = sorted(FINAL_BAD_WORDS_PATTERNS, key=len, reverse=True)
BAD_WORDS_PATTERN = re.compile(r'(?:' + '|'.join(SORTED_BAD_WORDS) + r')', re.IGNORECASE)


# SORTED_BAD_WORDS = sorted(BAD_WORDS_LIST, key=len, reverse=True)
# BAD_WORDS_PATTERN = re.compile(r'\b(?:' + '|'.join(map(re.escape, SORTED_BAD_WORDS)) + r')\b', re.IGNORECASE)

SORTED_AGGRESSIVE = sorted(AGGRESSIVE_PRONOUNS_LIST, key=len, reverse=True)
AGGRESSIVE_PRONOUN_PATTERN = re.compile(r'\b(?:' + '|'.join(map(re.escape, SORTED_AGGRESSIVE)) + r')\b', re.IGNORECASE)

SORTED_INTENSIFIER = sorted(INTENSIFIER_SARCASTIC_LIST, key=len, reverse=True)
INTENSIFIER_PATTERN = re.compile(r'\b(?:' + '|'.join(map(re.escape, INTENSIFIER_SARCASTIC_LIST)) + r')\b', re.IGNORECASE)

SORTED_POSITIVE = sorted(POSITIVE_SARCASTIC_LIST, key=len, reverse=True)
POSITIVE_PATTERN = re.compile(r'\b(?:' + '|'.join(map(re.escape, SORTED_INTENSIFIER)) + r')\b', re.IGNORECASE)

def debug(msg: str) -> None:
    print(f"[DEBUG][FeatureBuilder] {msg}")

def get_default_data_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset-vihsd"))

def normalize_whitespace(text: str) -> str:
    normalized = unicodedata.normalize("NFC", str(text)).strip()
    return re.sub(r"\s+", " ", normalized)

def compute_feature_signals(text: str, tokens: list[str]) -> dict:
    clean_text = text.replace("_", " ").lower()

    text_laugh = len(re.findall(r"(haha+|hehe+|hihi+|kkk+|hé hé|hô hô|há há)", clean_text))
    emoji_laugh = len(re.findall(r"(TEARS_OF_JOY|ROLLING_ON_THE_FLOOR|GRINNING_SQUINTING|SMIRKING)", text.upper()))
    paren_laugh = len(re.findall(r"(\={1,}\)+|\:{1,}\)+)", text))

    return {
        "emoji_count": len(EMOJI_TAG_PATTERN.findall(text)) + len(EMOJI_ALIAS_PATTERN.findall(text)),
        "punct_count": sum(1 for ch in text if unicodedata.category(ch).startswith("P")),
        "upper_count": sum(1 for ch in text if ch.isalpha() and ch.isupper()),
        "bad_word_count": calculate_weighted_bad_words(clean_text),
        "exclamation_count": sum(1 for ch in text if ch in ("!", "?")),
        "allcaps_count": sum(1 for t in tokens if t.isupper() and len(t) > 1),
        "laugh_count": text_laugh + emoji_laugh + paren_laugh,
        "agg_pronoun_count": len(AGGRESSIVE_PRONOUN_PATTERN.findall(clean_text)),
        "sarcastic_punct_count": len(SARCASTIC_PUNCT_PATTERN.findall(text)),
        "scare_quote_count": len(QUOTED_WORD_PATTERN.findall(text)),
        "intensifier_count": len(INTENSIFIER_PATTERN.findall(clean_text)),
    }

def extract_feature_row(text: str) -> dict:
    text = str(text)
    tokens = [tok for tok in text.split(" ") if tok]
    num_tokens = len(tokens)
    num_chars = len(text)

    # Đảm bảo an toàn toán học (tránh lỗi ZeroDivisionError)
    safe_tokens = max(num_tokens, 1)
    safe_chars = max(num_chars, 1)
    safe_alpha = max(sum(1 for ch in text if ch.isalpha()), 1)

    avg_token_len = float(np.mean([len(tok) for tok in tokens])) if num_tokens > 0 else 0.0

    # Gộp toàn bộ phép đếm tín hiệu vào một bước để giảm hàm rời rạc.
    signals = compute_feature_signals(text, tokens)

    # Trích xuất và định tuyến đặc trưng
    return {
        # =================================================================
        # NHÓM 1: KHÔNG GIAN LOGARITHM (Kích thước văn bản)
        # =================================================================
        "feat_log_num_tokens": float(np.log1p(num_tokens)),
        "feat_log_num_chars": float(np.log1p(num_chars)),
        "feat_avg_token_len": float(avg_token_len),
        
        # =================================================================
        # NHÓM 2: TỈ LỆ TUYẾN TÍNH (Cấu trúc & Định dạng)
        # =================================================================
        "feat_punct_density": float(signals["punct_count"] / safe_chars),
        "feat_upper_ratio": float(signals["upper_count"] / safe_alpha),
        "feat_exclamation_density": float(signals["exclamation_count"] / safe_chars),
        "feat_allcaps_ratio": float(signals["allcaps_count"] / safe_tokens),
        
        # =================================================================
        # NHÓM 3: MẬT ĐỘ PHI TUYẾN (Tín hiệu từ vựng / Cảm xúc)
        # Khôi phục không gian liên tục để cung cấp cường độ cho PhoBERT
        # =================================================================
        "feat_emoji_density": float(np.log1p((signals["emoji_count"] / safe_tokens) * 10)),
        "feat_bad_word_density": float(np.log1p((signals["bad_word_count"] / safe_tokens) * 10)),
        "feat_laugh_density": float(np.log1p((signals["laugh_count"] / safe_tokens) * 10)),
        "feat_aggressive_pronoun": float(np.log1p((signals["agg_pronoun_count"] / safe_tokens) * 10)),
        "feat_sarcastic_punct": float(np.log1p((signals["sarcastic_punct_count"] / safe_chars) * 10)),
        "feat_scare_quotes": float(np.log1p((signals["scare_quote_count"] / safe_tokens) * 10)),
        "feat_intensifier_words": float(np.log1p((signals["intensifier_count"] / safe_tokens) * 10)),
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