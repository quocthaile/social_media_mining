import os
import argparse
import pandas as pd
import re
import unicodedata
import emoji
from underthesea import word_tokenize
from typing import Optional
from preprocessing_dic import ABBREV_DICT_RAW, COMPOUND_DICT_RAW, SLANG_DICT_RAW, STOPWORDS


SPECIAL_TOKEN_PREFIXES = ("EMOJI_", "PUNC_", "EMOTICON_")
LEXICON_TOKEN_PATTERN = re.compile(r"[\w_+-]+|[^\w\s]", flags=re.UNICODE)
NEGATION_KEEP_TOKENS = {"không", "chẳng", "chả", "đừng", "chưa"}


SLANG_DICT  = {
    unicodedata.normalize('NFC', k): unicodedata.normalize('NFC', v).replace(' ', '_')
    for k, v in SLANG_DICT_RAW.items()
}
ABBREV_DICT = {
    unicodedata.normalize('NFC', k): unicodedata.normalize('NFC', v).replace(' ', '_')
    for k, v in ABBREV_DICT_RAW.items()
}
COMPOUND_DICT = {
    unicodedata.normalize('NFC', k): unicodedata.normalize('NFC', v).replace(' ', '_')
    for k, v in COMPOUND_DICT_RAW.items()
}

COMPOUND_PATTERNS = [
    (
        re.compile(
            r'(?<!\w)' + r'\s+'.join(re.escape(part) for part in phrase.split()) + r'(?!\w)',
            flags=re.IGNORECASE,
        ),
        replacement,
    )
    for phrase, replacement in sorted(COMPOUND_DICT.items(), key=lambda item: len(item[0].split()), reverse=True)
]

# ==========================================
# CÁC HÀM TIỀN XỬ LÝ
# ==========================================

def unicode_normalization(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize('NFC', text)

def remove_noise(text: str) -> str:
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[@#]\w+', '', text)
    return text

def is_word_token(token: str) -> bool:
    if not token:
        return False
    return all(ch.isalnum() or ch in {'_', '-'} for ch in token)

def is_special_token(token: str) -> bool:
    return token.startswith(SPECIAL_TOKEN_PREFIXES)

def tokenize_keep_punctuation(text: str) -> list[str]:
    """Tách token giữ dấu câu riêng để tránh miss-match kiểu 'ko?' hoặc 'đc.'"""
    return LEXICON_TOKEN_PATTERN.findall(text)

def apply_lexicon(tokens: list[str], lexicon: dict[str, str]) -> list[str]:
    """Áp dụng map từ điển trên token chữ, bỏ qua token kỹ thuật (EMOJI_/PUNC_)."""
    normalized_tokens = []
    for token in tokens:
        if is_word_token(token) and not is_special_token(token):
            replacement = lexicon.get(token, token)
            normalized_tokens.extend(replacement.split())
        else:
            normalized_tokens.append(token)
    return normalized_tokens

def separate_emoji_boundaries(text: str) -> str:
    emoji_spans = emoji.emoji_list(text)
    if not emoji_spans:
        return text
    pieces = []
    cursor = 0
    for item in emoji_spans:
        start = item["match_start"]
        end = item["match_end"]
        if start > cursor:
            pieces.append(text[cursor:start])
        pieces.append(f" {item['emoji']} ")
        cursor = end
    if cursor < len(text):
        pieces.append(text[cursor:])
    return ''.join(pieces)

def emoji_to_fallback_tag(emoji_text: str) -> str:
    alias = emoji.demojize(emoji_text, delimiters=("", ""))
    alias = re.sub(r'[^0-9a-zA-Z_]+', '_', alias).strip('_')
    if not alias:
        return "EMOJI_KHAC"
    return f"EMOJI_ALIAS_{alias.upper()}"

def extract_emoji_features(text: str) -> str:
    text = separate_emoji_boundaries(text)
    emoji_spans = emoji.emoji_list(text)
    if not emoji_spans:
        return text
    result = []
    cursor = 0
    for item in emoji_spans:
        start = item["match_start"]
        end = item["match_end"]
        symbol = item["emoji"]
        if start > cursor:
            result.append(text[cursor:start])
        result.append(f" {emoji_to_fallback_tag(symbol)} ")
        cursor = end
    if cursor < len(text):
        result.append(text[cursor:])
    return ''.join(result)

def normalize_lengthened_words(text: str) -> str:
    def _punct_with_intensity(symbol: str, tag_prefix: str):
        def _repl(match: re.Match) -> str:
            level = min(len(match.group(0)), 3)
            return f" {symbol} {tag_prefix}_{level} "
        return _repl
    # Lớp 1a: Giữ tín hiệu cảm xúc của dấu câu lặp bằng intensity tag
    text = re.sub(r'\.{3,}', _punct_with_intensity('…', 'PUNC_ELLIPSIS'), text)
    text = re.sub(r'!{2,}', _punct_with_intensity('!', 'PUNC_EXCLAM'), text)
    text = re.sub(r'\?{2,}', _punct_with_intensity('?', 'PUNC_QUESTION'), text)
    # Lớp 1b: Các dấu câu lặp khác thu gọn còn 1 ký tự
    text = re.sub(r'([,;:\)\(\]\[><~\-=])\1+', r' \1 ', text)
    # Lớp 2: cắt chữ cái (Latin + tiếng Việt có dấu) lặp ≥ 3 lần về còn 1
    text = re.sub(
        r'([a-zàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ])\1{2,}',
        r'\1', text)
    return text

def replace_slang_and_abbreviations_cased(
    text: str,
    slang_dict: dict[str, str] = SLANG_DICT,
    abbrev_dict: dict[str, str] = ABBREV_DICT,
) -> str:
    tokens = text.split()
    processed_tokens = []
    
    for token in tokens:
        token_lower = token.lower()
        if token_lower in abbrev_dict:
            processed_tokens.append(abbrev_dict[token_lower])
        elif token_lower in slang_dict:
            processed_tokens.append(slang_dict[token_lower])
        else:
            processed_tokens.append(token)
    return " ".join(processed_tokens)

def replace_compound_words(text: str) -> str:
    """Gộp các cụm từ ghép phổ biến thành một token có dấu underscore."""
    for pattern, replacement in COMPOUND_PATTERNS:
        text = pattern.sub(replacement, text)
    return text

def segment_text(text: str, remove_stopwords: bool = True) -> str:
    """Phân đoạn từ (underthesea), có thể bật/tắt loại bỏ stopwords."""
    tokenized_text = word_tokenize(text, format="text")
    if isinstance(tokenized_text, list):
        words = tokenized_text
    else:
        words = tokenized_text.split()

    if not remove_stopwords:
        return ' '.join(words)

    filtered = [
        w for w in words
        if not (
            w.lower() in STOPWORDS
            and w.lower() not in NEGATION_KEEP_TOKENS
            and not is_special_token(w))
    ]
    return ' '.join(filtered)

def run_quick_regression_checks(remove_stopwords: bool = True) -> None:
    """Self-test nhanh cho các ca social text dễ tách sai."""
    samples = [
        "Ko??? dcm!!!",
        "Được anh ưi :)))",
        "Cắt cho trẻ trâu bớt thui mà 😂😂😂",
        "Thế đấy. làm j bọn nó :v",
        "đừng chửi nữa!!!",
        "Mọi người đừng quên chúc mừng năm mới",
    ]

    print("\n--- SELF TEST: QUICK REGRESSION SAMPLES ---")
    for idx, sample in enumerate(samples, start=1):
        text = unicode_normalization(sample)
        text = text.lower()
        text = remove_noise(text)
        text = extract_emoji_features(text)
        text = normalize_lengthened_words(text)
        text = replace_slang_and_abbreviations_cased(text)
        text = replace_compound_words(text)
        text = segment_text(text, remove_stopwords=remove_stopwords)
        text = re.sub(r'\s+', ' ', text).strip()

        print(f"[{idx}] IN : {sample}")
        print(f"    OUT: {text}")


# ==========================================
# PIPELINE & LOGGING
# ==========================================

def run_pipeline(df: pd.DataFrame, text_column: str, remove_stopwords: bool = True) -> pd.DataFrame:
    """Thực thi pipeline 7 bước và in log."""
    print("  [1/7] Đồng nhất bảng mã Unicode NFC (fix NFD/mixed)...")
    df['clean_text'] = df[text_column].apply(unicode_normalization)

    print("  [2/7] Lọc nhiễu kỹ thuật (URL, @mention, #hashtag)...")
    df['clean_text'] = df['clean_text'].apply(remove_noise)

    print("  [3/7] Trích xuất đặc trưng Emoji (demojize tiếng Anh + khoảng trắng)...")
    df['clean_text'] = df['clean_text'].apply(extract_emoji_features)

    print("  [4/7] Chuẩn hóa ký tự kéo dài (dấu câu + chữ cái)...")
    df['clean_text'] = df['clean_text'].apply(normalize_lengthened_words)

    print("  [5/7] Chuẩn hóa Teencode: ABBREV_DICT → SLANG_DICT...")
    df['clean_text'] = df['clean_text'].apply(replace_slang_and_abbreviations_cased)

    print("  [6/7] Gộp từ ghép phổ biến thành token underscore...")
    df['clean_text'] = df['clean_text'].apply(replace_compound_words)

    if remove_stopwords:
        print("  [7/7] Phân đoạn từ + Loại bỏ Stopwords (mất chút thời gian)...")
    else:
        print("  [7/7] Phân đoạn từ (Giữ nguyên stopwords)...")
    df['clean_text'] = df['clean_text'].apply(
        lambda x: segment_text(x, remove_stopwords=remove_stopwords)
    )

    print("  [*] Dọn khoảng trắng thừa...")
    df['clean_text'] = df['clean_text'].apply(
        lambda x: re.sub(r'\s+', ' ', str(x)).strip()
    )
    return df


def process_dataset(dataset_dir: str | None = None, remove_stopwords: bool = True):
    """Hàm main: đọc origin_tran/dev/test → xử lý → lưu preprocessed_v2_*.csv."""
    TEXT_COLUMN = 'free_text'

    if dataset_dir is None:
        dataset_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'dataset-vihsd')
        )

    try:
        print(f"--- BẮT ĐẦU ĐỌC DỮ LIỆU TỪ: {dataset_dir} ---")
        df_train = pd.read_csv(os.path.join(dataset_dir, 'origin_train.csv'))
        df_dev   = pd.read_csv(os.path.join(dataset_dir, 'origin_dev.csv'))
        df_test  = pd.read_csv(os.path.join(dataset_dir, 'origin_test.csv'))

        print(f"\n▶ ĐANG XỬ LÝ TẬP TRAIN ({len(df_train)} dòng)...")
        df_train = run_pipeline(df_train, TEXT_COLUMN, remove_stopwords=remove_stopwords)

        print(f"\n▶ ĐANG XỬ LÝ TẬP VALIDATION/DEV ({len(df_dev)} dòng)...")
        df_dev = run_pipeline(df_dev, TEXT_COLUMN, remove_stopwords=remove_stopwords)

        print(f"\n▶ ĐANG XỬ LÝ TẬP TEST ({len(df_test)} dòng)...")
        df_test = run_pipeline(df_test, TEXT_COLUMN, remove_stopwords=remove_stopwords)

        print("\n--- ĐANG LƯU KẾT QUẢ ---")
        out_train = os.path.join(dataset_dir, 'preprocessed_train.csv')
        out_dev   = os.path.join(dataset_dir, 'preprocessed_dev.csv')
        out_test  = os.path.join(dataset_dir, 'preprocessed_test.csv')

        df_train[['clean_text', 'label_id']].to_csv(out_train, index=False)
        df_dev[['clean_text', 'label_id']].to_csv(out_dev, index=False)
        df_test[['clean_text', 'label_id']].to_csv(out_test, index=False)

        print(f" HOÀN TẤT! {out_train}, {out_dev}, {out_test}")

    except Exception as e:
        import traceback
        print(f"\n [LỖI]: {e}")
        traceback.print_exc()


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='preprocessing — đầy đủ 4 fix: Unicode / Lexicon / Lengthening / Emoji'
    )
    parser.add_argument(
        '--dataset_dir', '-d',
        help='Đường dẫn thư mục dataset (mặc định: ../dataset-vihsd)',
        default=None
    )
    parser.add_argument(
        '--self_test',
        action='store_true',
        help='Chạy nhanh một bộ regression sample trước khi xử lý dataset'
    )
    parser.add_argument(
        '--remove_stopwords',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Bật/tắt loại bỏ stopwords ở bước phân đoạn từ (mặc định: bật)'
    )
    args = parser.parse_args()

    if args.self_test:
        run_quick_regression_checks(remove_stopwords=args.remove_stopwords)

    process_dataset(args.dataset_dir, remove_stopwords=args.remove_stopwords)
