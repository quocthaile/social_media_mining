"""
preprocessing_v2.py
===================
Phiên bản cải tiến của preprocessing.py với đầy đủ các fix:

FIX 1 — Unicode Normalization:
    - Normalize cả key lẫn value của SLANG_DICT / ABBREV_DICT về NFC khi khởi động
      → tránh miss-match khi input NFD khác key NFC

FIX 2 — Lexicon Normalization:
    - Tách thành 2 từ điển riêng: SLANG_DICT (từ lóng/tục) và ABBREV_DICT (viết tắt)
    - Bổ sung thêm ~25 entry từ phân tích vocab tự động
    - Áp dụng ABBREV_DICT trước, SLANG_DICT sau

FIX 3 — Lengthening Normalization:
    - Thêm Lớp 1: cắt dấu câu lặp   :))) → :)   ... → .   !!! → !
    - Giữ Lớp 2: cắt chữ cái lặp    gìiiii → gì (regex cũ)

FIX 4 — Emoji Polarity Extraction:
    - Chèn khoảng trắng quanh emoji trước khi demojize()
      → tránh 😂😡 dính thành 1 token :face_with_tears_of_joy::enraged_face:
    - Map emoji phổ biến sang tag phân cực tiếng Việt (EMOJI_POLARITY)
      thay vì dùng tên tiếng Anh trung tính của demojize()
"""

import os
import argparse
import pandas as pd
import re
import unicodedata
import emoji
from underthesea import word_tokenize
from typing import Optional

# ==========================================
# BẢNG MAP EMOJI → TAG PHÂN CỰC [FIX 4]
# ==========================================
# Ưu tiên: tra bảng này trước; emoji không có trong bảng
# mới fallback về demojize() tiếng Anh.
EMOJI_POLARITY = {
    # Tích cực — vui / yêu
    "😀": "EMOJI_TICH_CUC", "😁": "EMOJI_TICH_CUC", "😄": "EMOJI_TICH_CUC",
    "😆": "EMOJI_TICH_CUC", "😊": "EMOJI_TICH_CUC", "🙂": "EMOJI_TICH_CUC",
    "😍": "EMOJI_TICH_CUC", "🥰": "EMOJI_TICH_CUC", "😘": "EMOJI_TICH_CUC",
    "😂": "EMOJI_TICH_CUC", "🤣": "EMOJI_TICH_CUC", "😅": "EMOJI_TICH_CUC",
    "😎": "EMOJI_TICH_CUC", "🤩": "EMOJI_TICH_CUC", "😇": "EMOJI_TICH_CUC",
    "❤":  "EMOJI_TICH_CUC", "❤️": "EMOJI_TICH_CUC", "🧡": "EMOJI_TICH_CUC",
    "💛": "EMOJI_TICH_CUC", "💚": "EMOJI_TICH_CUC", "💙": "EMOJI_TICH_CUC",
    "💜": "EMOJI_TICH_CUC", "🖤": "EMOJI_TICH_CUC", "💕": "EMOJI_TICH_CUC",
    "💞": "EMOJI_TICH_CUC", "💓": "EMOJI_TICH_CUC", "💗": "EMOJI_TICH_CUC",
    "👍": "EMOJI_TICH_CUC", "👏": "EMOJI_TICH_CUC", "🙌": "EMOJI_TICH_CUC",
    "✅": "EMOJI_TICH_CUC", "🎉": "EMOJI_TICH_CUC", "🎊": "EMOJI_TICH_CUC",
    "🌟": "EMOJI_TICH_CUC", "⭐": "EMOJI_TICH_CUC", "🔥": "EMOJI_TICH_CUC",

    # Tiêu cực nhẹ — buồn / thất vọng
    "😢": "EMOJI_TIEU_CUC", "😭": "EMOJI_TIEU_CUC", "😞": "EMOJI_TIEU_CUC",
    "😔": "EMOJI_TIEU_CUC", "😟": "EMOJI_TIEU_CUC", "😕": "EMOJI_TIEU_CUC",
    "🙁": "EMOJI_TIEU_CUC", "😣": "EMOJI_TIEU_CUC", "😩": "EMOJI_TIEU_CUC",
    "😰": "EMOJI_TIEU_CUC", "😱": "EMOJI_TIEU_CUC", "😓": "EMOJI_TIEU_CUC",
    "😒": "EMOJI_TIEU_CUC", "😑": "EMOJI_TIEU_CUC", "🙄": "EMOJI_TIEU_CUC",

    # Tiêu cực mạnh — tức giận / thù địch / xúc phạm
    "😡": "EMOJI_GIAN_DU",  "🤬": "EMOJI_GIAN_DU",  "😠": "EMOJI_GIAN_DU",
    "😤": "EMOJI_GIAN_DU",  "👿": "EMOJI_GIAN_DU",  "😈": "EMOJI_GIAN_DU",
    "🖕": "EMOJI_XUC_PHAM", "💩": "EMOJI_XUC_PHAM", "🤮": "EMOJI_XUC_PHAM",
    "🤢": "EMOJI_XUC_PHAM", "💀": "EMOJI_XUC_PHAM", "☠️": "EMOJI_XUC_PHAM",
    "👎": "EMOJI_TIEU_CUC",
}

# ==========================================
# TỪ ĐIỂN TỪ LÓNG (SLANG) [FIX 2]
# Chỉ chứa: từ tục, từ biến âm cố tình, teencode mang nghĩa xúc phạm
# ==========================================
_SLANG_DICT_RAW = {
    # Nhóm từ tục / xúc phạm trực tiếp
    "dell":    "đếch",
    "đell":    "đếch",
    "đéo":     "đếch",
    "deo":     "đếch",
    "eo":      "éo",
    "đm":      "địt_mẹ",
    "dm":      "địt_mẹ",
    "đmm":     "địt_mẹ_mày",
    "dmm":     "địt_mẹ_mày",
    "dcm":     "địt_cụ_mày",
    "cc":      "cục_cứt",
    "cl":      "cái_lồn",
    "clm":     "cái_lồn_má",
    "clgv":    "cái_lồn_gì_vậy",
    "clgt":    "cái_lồn_gì_thế",
    "clg":     "cái_lồn_gì",
    "lol":     "lồn",
    "loz":     "lồn",
    "lz":      "lồn",
    "l":       "lồn",
    "ml":      "mặt_lồn",
    "cmm":     "con_mẹ_mày",
    "cmn":     "con_mẹ_nó",
    "cmnr":    "con_mẹ_nó_rồi",
    "xamlol":  "xàm_lồn",
    "xamlone": "xàm_lồn",
    "sml":     "sấp_mặt_lồn",
    "vl":      "vãi_lồn",
    "vcl":     "vãi_cả_lồn",
    "vkl":     "vãi_cả_lồn",
    "vll":     "vãi_lồn_luôn",
    "vlon":    "vãi_lồn",
    "vailon":  "vãi_lồn",
    "vcđ":     "vãi_cả_đái",
    "cđ":      "con_đĩ",
    "đjs":     "địt_mẹ",
    "dhs":     "đéo_hiểu_sao",
    "xl":      "xạo_lồn",
    "éo":      "đéo",
    "đ":       "đéo",
    "đesss":   "đéo",
    "moẹ":     "mẹ",

    # Nhóm biến âm cố tình (teencode âm thanh)
    "thỳ":     "thì",
    "dề":      "về",
    "ge":      "ghê",
    "thik":    "thích",
    "ik":      "đi",
    "jeenh":   "kênh",
    "zời":     "trời",
    "zị":      "vậy",
    "zay":     "vậy",
    "ưiiiii":  "ơi",
    "qá":      "quá",
    "qa":      "quá",
    "roày":    "rồi",
    "jui":     "rồi",
    "ni":      "này",
    "nà":      "này",
    "lun":     "luôn",
    "ah":      "à",
    "kkk":     "haha",
    "khong":   "không",
}

# ==========================================
# TỪ ĐIỂN VIẾT TẮT (ABBREVIATION) [FIX 2]
# Chỉ chứa: chữ viết tắt, ký hiệu internet, teencode viết tắt
# ==========================================
_ABBREV_DICT_RAW = {
    # Phủ định
    "ko":    "không",
    "k":     "không",
    "hk":    "không",
    "hok":   "không",
    "kg":    "không",

    # Đại từ / xưng hô
    "m":     "mày",
    "t":     "tao",
    "e":     "em",
    "tui":   "tôi",
    "mk":    "mình",
    "ck":    "chồng",
    "vk":    "vợ",

    # Động từ / trạng từ
    "dc":    "được",
    "đc":    "được",
    "vs":    "với",
    "v":     "vậy",
    "r":     "rồi",
    "j":     "gì",
    "z":     "vậy",
    "lm":    "làm",
    "ok":    "được_rồi",
    "iu":    "yêu",
    "ak":    "à",
    "cx":    "cũng",
    "bit":   "biết",
    "ht":    "hết",
    "nc":    "nói_chuyện",
    "trc":   "trước",
    "tr":    "trước",
    "nhe":   "nhé",
    "roi":   "rồi",
    "gi":    "gì",
    "luon":  "luôn",
    "dg":    "đang",

    # Viết tắt không dấu — từ ngữ thông thường
    "nguoi": "người",
    "vn":    "việt_nam",

    # Mạng xã hội / internet
    "mn":    "mọi_người",
    "mng":   "mọi_người",
    "ae":    "anh_em",
    "ac":    "anh_chị",
    "ad":    "admin",
    "fb":    "facebook",
    "ib":    "nhắn_tin",
    "inb":   "nhắn_tin",
    "rep":   "trả_lời",
    "cmt":   "bình_luận",
    "stt":   "trạng_thái",
    "tus":   "trạng_thái",
    "wed":   "web",

    # Viết tắt thông dụng khác
    "ny":    "người_yêu",
    "bh":    "bao_giờ",
    "hqua":  "hôm_qua",
    "hqa":   "hôm_qua",
    "ntn":   "như_thế_nào",
    "bt":    "bình_thường",
    "bth":   "bình_thường",
    "bn":    "bạn",
    "qtrong":"quan trọng",
    "tgian": "thời_gian",
    "ng":    "người",
    "ngta":  "người_ta",
    "đg":    "đang",
    "trk":   "trước",
    "h":     "giờ",
    "kp":    "không_phải",
    "in4":   "thông_tin",
    "sud":   "đăng_ký",
    "cacban":"các bạn",
    "dlv":   "dư_luận_viên",
    "hvb":   "hồng_vệ_binh",
    "3":     "ba",
}

# FIX 1 — Normalize tất cả key/value về NFC ngay khi khởi động
# → Tránh miss-match nếu có key NFD lẫn vào khi chỉnh sửa file
SLANG_DICT  = {
    unicodedata.normalize('NFC', k): unicodedata.normalize('NFC', v)
    for k, v in _SLANG_DICT_RAW.items()
}
ABBREV_DICT = {
    unicodedata.normalize('NFC', k): unicodedata.normalize('NFC', v)
    for k, v in _ABBREV_DICT_RAW.items()
}

# ==========================================
# STOPWORDS
# ==========================================
STOPWORDS = set([
    # Nhóm gốc
    "và", "là", "như", "thì", "mà", "nếu", "có", "các", "những", "của",
    "cho", "đi", "này", "cái", "nó", "rồi", "lại", "ra", "hay", "còn",
    "phải", "mình", "ơi", "nào", "thế", "sao", "ai", "ở", "đâu", "đó",
    "để", "thôi", "vậy", "với", "chỉ", "cả", "đã", "vào", "nên", "nữa",
    "từ", "khi", "đến", "trong", "vì", "cứ", "sau", "con", "gì", "rất",
    "quá", "đang", "mới", "hơn", "luôn", "được", "không", "hoặc", "làm",
    "thấy", "bị", "nhé", "nha", "hả", "thật",

    # Nhóm bổ sung — động từ / phó từ trung tính (từ phân tích vocab)
    "cũng",  "biết",  "muốn",  "nghe",  "nhìn",
    "đúng",  "vẫn",   "chưa",  "nói",   "học",
    "xem",   "cần",   "nhớ",   "nhiều", "chứ",
    "lên",   "xuống", "vô",    "sẽ",    "đây",
    "đấy",   "lắm",   "hết",   "theo",  "nhau",
    "vừa",   "thêm",  "một",   "cùng",  "tất",
])

# ==========================================
# CÁC HÀM TIỀN XỬ LÝ
# ==========================================

def unicode_normalization(text: str) -> str:
    """
    [FIX 1] Bước 1: Đồng nhất bảng mã về chuẩn Unicode NFC.
    Xử lý lỗi dấu thanh Tổ hợp/Dựng sẵn — rào cản kinh điển trong NLP tiếng Việt.
    Phát hiện: 541/24048 dòng train (2.25%) bị lỗi NFD.
    """
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize('NFC', text)


def remove_noise(text: str) -> str:
    """Bước 3: Lọc nhiễu kỹ thuật — URL, @mention, #hashtag."""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[@#]\w+', '', text)
    return text


def extract_emoji_features(text: str) -> str:
    """
    [FIX 4] Bước 4: Khai thác đặc trưng phân cực Emoji.

    Thay đổi so với v1:
      - Chèn khoảng trắng quanh mỗi emoji trước khi xử lý
        → tránh 😂😡 bị ghép thành 1 token dính
      - Map emoji phổ biến sang tag phân cực tiếng Việt (EMOJI_POLARITY)
        thay vì dùng tên tiếng Anh của demojize() (ít ngữ nghĩa hơn)
      - Emoji không có trong bảng → fallback về demojize()
    """
    result = []
    for char in text:
        if emoji.is_emoji(char):
            tag = EMOJI_POLARITY.get(char)
            if tag:
                result.append(f' {tag} ')       # tag phân cực rõ ràng
            else:
                result.append(f' {emoji.demojize(char)} ')  # fallback
        else:
            result.append(char)
    return ''.join(result)


def normalize_lengthened_words(text: str) -> str:
    """
    [FIX 3] Bước 5: Chuẩn hóa ký tự / dấu câu kéo dài.

    Thay đổi so với v1:
      Lớp 1 (MỚI): Dấu câu lặp lại (510+ lần trong dataset)
                   :))) → :)   ... → .   !!! → !   ??? → ?
      Lớp 2 (GIỮ): Chữ cái tiếng Việt lặp lại
                   gìiiii → gì   buồnnnn → buồn
    """
    # Lớp 1: cắt dấu câu lặp (≥ 2 lần) về còn 1
    text = re.sub(r'([!?,.:;\)\(\]\[><~\-=])\1+', r'\1', text)

    # Lớp 2: cắt chữ cái (Latin + tiếng Việt có dấu) lặp ≥ 3 lần về còn 1
    text = re.sub(
        r'([a-zàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ])\1{2,}',
        r'\1', text
    )
    return text


def replace_slang_and_abbreviations(text: str) -> str:
    """
    [FIX 2] Bước 6: Chuẩn hóa từ lóng và viết tắt — 2 từ điển riêng biệt.

    Thay đổi so với v1:
      - ABBREV_DICT (viết tắt) áp dụng TRƯỚC
      - SLANG_DICT  (từ lóng)  áp dụng SAU
      Lý do: viết tắt thường là prefix của từ lóng (vd: "dc" → "được",
      không nhầm với "dcm" → "địt cụ mày" vì "dcm" khớp exact match)
    """
    words = text.split()
    # Áp dụng ABBREV_DICT trước
    words = [ABBREV_DICT.get(w, w) for w in words]
    # Áp dụng SLANG_DICT sau (cần re-split vì value có thể là cụm từ)
    words = [SLANG_DICT.get(w, w) for w in ' '.join(words).split()]
    return ' '.join(words)


def segment_and_remove_stopwords(text: str) -> str:
    """Bước 7: Phân đoạn từ (underthesea) và loại bỏ từ dừng."""
    tokenized_text = word_tokenize(text, format="text")
    if isinstance(tokenized_text, list):
        words = tokenized_text
    else:
        words = tokenized_text.split()
    # words = tokenized_text.split()

    filtered = [w for w in words if w.lower() not in STOPWORDS]
    return ' '.join(filtered)


# ==========================================
# PIPELINE & LOGGING
# ==========================================

def run_pipeline(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Thực thi pipeline 7 bước và in log."""
    print("  [1/7] Đồng nhất bảng mã Unicode NFC (fix NFD/mixed)...")
    df['clean_text'] = df[text_column].apply(unicode_normalization)

    # print("  [2/7] Chuyển về chữ thường...")
    # df['clean_text'] = df['clean_text'].str.lower()

    print("  [3/7] Lọc nhiễu kỹ thuật (URL, @mention, #hashtag)...")
    df['clean_text'] = df['clean_text'].apply(remove_noise)

    print("  [4/7] Trích xuất đặc trưng phân cực Emoji (có khoảng trắng + tag cực)...")
    df['clean_text'] = df['clean_text'].apply(extract_emoji_features)

    print("  [5/7] Chuẩn hóa ký tự kéo dài (dấu câu + chữ cái)...")
    df['clean_text'] = df['clean_text'].apply(normalize_lengthened_words)

    print("  [6/7] Chuẩn hóa Teencode: ABBREV_DICT → SLANG_DICT...")
    df['clean_text'] = df['clean_text'].apply(replace_slang_and_abbreviations)

    print("  [7/7] Phân đoạn từ + Loại bỏ Stopwords (mất chút thời gian)...")
    df['clean_text'] = df['clean_text'].apply(segment_and_remove_stopwords)

    print("  [*] Dọn khoảng trắng thừa...")
    df['clean_text'] = df['clean_text'].apply(
        lambda x: re.sub(r'\s+', ' ', str(x)).strip()
    )
    return df


def process_dataset(dataset_dir: str | None = None):
    """Hàm main: đọc train/dev/test → xử lý → lưu preprocessed_v2_*.csv."""
    TEXT_COLUMN = 'free_text'

    if dataset_dir is None:
        dataset_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'dataset-vihsd')
        )

    try:
        print(f"--- BẮT ĐẦU ĐỌC DỮ LIỆU TỪ: {dataset_dir} ---")
        df_train = pd.read_csv(os.path.join(dataset_dir, 'train.csv'))
        df_dev   = pd.read_csv(os.path.join(dataset_dir, 'dev.csv'))
        df_test  = pd.read_csv(os.path.join(dataset_dir, 'test.csv'))

        print(f"\n▶ ĐANG XỬ LÝ TẬP TRAIN ({len(df_train)} dòng)...")
        df_train = run_pipeline(df_train, TEXT_COLUMN)

        print(f"\n▶ ĐANG XỬ LÝ TẬP VALIDATION/DEV ({len(df_dev)} dòng)...")
        df_dev = run_pipeline(df_dev, TEXT_COLUMN)

        print(f"\n▶ ĐANG XỬ LÝ TẬP TEST ({len(df_test)} dòng)...")
        df_test = run_pipeline(df_test, TEXT_COLUMN)

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
        description='preprocessing_v2 — đầy đủ 4 fix: Unicode / Lexicon / Lengthening / Emoji'
    )
    parser.add_argument(
        '--dataset_dir', '-d',
        help='Đường dẫn thư mục dataset (mặc định: ../dataset-vihsd)',
        default=None
    )
    args = parser.parse_args()
    process_dataset(args.dataset_dir)
