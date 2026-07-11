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

FIX 4 — Emoji Feature Extraction:
        - Chèn khoảng trắng quanh emoji trước khi demojize()
            → tránh 😂😡 dính thành 1 token
        - Chuẩn hóa emoji về token trung tính EMOJI_ALIAS_* để đưa vào mô hình
"""

import os
import argparse
import pandas as pd
import re
import unicodedata
import emoji
from underthesea import word_tokenize
from typing import Optional


SPECIAL_TOKEN_PREFIXES = ("EMOJI_", "PUNC_", "EMOTICON_")
LEXICON_TOKEN_PATTERN = re.compile(r"[\w_+-]+|[^\w\s]", flags=re.UNICODE)
NEGATION_KEEP_TOKENS = {"không", "chẳng", "chả", "đừng", "chưa"}

# ==========================================
# CÁC HÀM TIỀN XỬ LÝ
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
    "khongg":  "không",
    "khonggg": "không",
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
    "b":     "bạn",
    "a":     "anh",

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
    "bnhiu":  "bao_nhiêu",
    "ddi":   "đi",
}

# ==========================================
# TỪ ĐIỂN TỪ GHÉP / CỤM TỪ THƯỜNG GẶP TRONG train.csv + dev.csv
# Giữ nguyên thành một token bằng dấu underscore trước bước loại stopwords.
# ==========================================
_COMPOUND_DICT_RAW = {
    "nhà nước": "nhà_nước",
    "chính phủ": "chính_phủ",
    "quốc gia": "quốc_gia",
    "nhân dân": "nhân_dân",
    "dư luận viên": "dư_luận_viên",
    "nhà báo": "nhà_báo",
    "báo công an": "báo_công_an",
    "bộ đội": "bộ_đội",
    "công an": "công_an",
    "cách ly": "cách_ly",
    "khâu trang": "khẩu_trang",
    "khẩu trang": "khẩu_trang",
    "táo quân": "táo_quân",
    "năm mới": "năm_mới",
    "chúc mừng": "chúc_mừng",
    "mừng năm mới": "mừng_năm_mới",
    "năm nay": "năm_nay",
    "năm sau": "năm_sau",
    "hôm nay": "hôm_nay",
    "hôm qua": "hôm_qua",
    "bây giờ": "bây_giờ",
    "lần đầu": "lần_đầu",
    "cuối cùng": "cuối_cùng",
    "ngày càng": "ngày_càng",
    "nói chung": "nói_chung",
    "thật sự": "thật_sự",
    "rất thuyết phục": "rất_thuyết_phục",
    "cực kỳ": "cực_kỳ",
    "cực kì": "cực_kì",
    "hợp lý": "hợp_lý",
    "tuyệt vời": "tuyệt_vời",
    "chắc chắn": "chắc_chắn",
    "không biết": "không_biết",
    "không phải": "không_phải",
    "không cần": "không_cần",
    "không có": "không_có",
    "chỉ có": "chỉ_có",
    "có vẻ": "có_vẻ",
    "được rồi": "được_rồi",
    "làm sao": "làm_sao",
    "như thế": "như_thế",
    "thế nào": "thế_nào",
    "bao giờ": "bao_giờ",
    "xin lỗi": "xin_lỗi",
    "xin phép": "xin_phép",
    "xin video": "xin_video",
    "xin link": "xin_link",
    "nghe nói": "nghe_nói",
    "xem video": "xem_video",
    "link video": "link_video",
    "nhắn tin": "nhắn_tin",
    "trả lời": "trả_lời",
    "bình luận": "bình_luận",
    "đăng ký": "đăng_ký",
    "bình thường": "bình_thường",
    "thời gian": "thời_gian",
    "thành công": "thành_công",
    "công nhận": "công_nhận",
    "phát triển": "phát_triển",
    "quảng cáo": "quảng_cáo",
    "gia đình": "gia_đình",
    "nước ngoài": "nước_ngoài",
    "người dân": "người_dân",
    "nhiều người": "nhiều_người",
    "mấy thằng": "mấy_thằng",
    "mấy đứa": "mấy_đứa",
    "lãnh đạo": "lãnh_đạo",
    "chính quyền": "chính_quyền",
    "chương trình": "chương_trình",
    "câu chuyện": "câu_chuyện",
    "bản thân": "bản_thân",
    "quan tâm": "quan_tâm",
    "suy nghĩ": "suy_nghĩ",
    "cẩn thận": "cẩn_thận",
    "quan trọng": "quan_trọng",
    "khẩu nghiệp": "khẩu_nghiệp",
    "cộng đồng": "cộng_đồng",
    "cộng sản": "cộng_sản",
    "phản động": "phản_động",
    "tham nhũng": "tham_nhũng",
    "trẻ trâu": "trẻ_trâu",
    "huyền thoại": "huyền_thoại",
    "danh hài": "danh_hài",
    "giải trí": "giải_trí",
    "cuộc sống": "cuộc_sống",
    "sức khỏe": "sức_khỏe",
    "mùa dịch": "mùa_dịch",
    "dịch bệnh": "dịch_bệnh",
    "tình hình": "tình_hình",
    "tương lai": "tương_lai",
    "may mắn": "may_mắn",
    "yên tâm": "yên_tâm",
    "đạo đức": "đạo_đức",
    "giao thông": "giao_thông",
    "hình ảnh": "hình_ảnh",
    "cảm giác": "cảm_giác",
    "khả năng": "khả_năng",
    "thần kinh": "thần_kinh",
    "phong cách": "phong_cách",
    "câu nói": "câu_nói",
    "hợp tác": "hợp_tác",
    "ngày xưa": "ngày_xưa",
    "nước mắt": "nước_mắt",
    "người việt": "người_việt",
    "sài gòn": "sài_gòn",
    "sân bay": "sân_bay",
    "máy bay": "máy_bay",
    "fan cứng": "fan_cứng",
    "xạo lồn": "xạo_lồn",
    "tấu hài": "tấu_hài",
    "cánh hoa": "cánh_hoa",
    "vàng ngọc": "vàng_ngọc",
    "cao lãnh": "cao_lãnh",
    "trung thu": "trung_thu",
    "trả nợ": "trả_nợ",
    "không gian": "không_gian",
    "thể hiện": "thể_hiện",
    "sản phẩm": "sản_phẩm",
    "người khác": "người_khác",
    "nói chuyện": "nói_chuyện",
    "xin chào": "xin_chào",
    "về việt nam": "về_việt_nam",
    "trên mạng": "trên_mạng",
    "chụp ảnh": "chụp_ảnh",
    "làm việc": "làm_việc",
    "đi học": "đi_học",
    "đi làm": "đi_làm",
    "đọc báo": "đọc_báo",
    "xem phim": "xem_phim",
    "chốt đơn": "chốt_đơn",
    "like dạo": "like_dạo",
    "bị chặn": "bị_chặn",
    "bảo vệ": "bảo_vệ",
    "kiểm soát": "kiểm_soát",
    "thuyết phục": "thuyết_phục",
    "hợp nhóm": "hợp_nhóm",
    "công khai": "công_khai",
    "bị phạt": "bị_phạt",
    "quá hay": "quá_hay",
    "quá đẹp": "quá_đẹp",
    "quá tốt": "quá_tốt",
    "bao nhiêu": "bao_nhiêu",
    "xét nghiệm": "xét_nghiệm",
    "văn hóa": "văn_hóa",
    "văn hoá": "văn_hoá",
    "đồng chí": "đồng_chí",
    "nước hoa": "nước_hoa",
    "chủ tịch": "chủ_tịch",
    "kinh nghiệm": "kinh_nghiệm",
    "nổi tiếng": "nổi_tiếng",
    "thanh niên": "thanh_niên",
    "phân biệt": "phân_biệt",
    "đồng lòng": "đồng_lòng",
    "bỏ phiếu": "bỏ_phiếu",
    "bán hàng": "bán_hàng",
    "chụp hình": "chụp_hình",
    "việt nam": "việt_nam",
    "đất nước": "đất_nước",
    "virus corona": "virus_corona",
    "covid 19": "covid_19"
}
# FIX 1 — Normalize tất cả key/value về NFC ngay khi khởi động
# → Tránh miss-match nếu có key NFD lẫn vào khi chỉnh sửa file
SLANG_DICT  = {
    unicodedata.normalize('NFC', k): unicodedata.normalize('NFC', v).replace(' ', '_')
    for k, v in _SLANG_DICT_RAW.items()
}
ABBREV_DICT = {
    unicodedata.normalize('NFC', k): unicodedata.normalize('NFC', v).replace(' ', '_')
    for k, v in _ABBREV_DICT_RAW.items()
}
COMPOUND_DICT = {
    unicodedata.normalize('NFC', k): unicodedata.normalize('NFC', v).replace(' ', '_')
    for k, v in _COMPOUND_DICT_RAW.items()
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
# STOPWORDS
# ==========================================
STOPWORDS = set([
    # Nhóm gốc
    "và", "là", "như", "thì", "mà", "nếu", "có", "các", "những", "của",
    "cho", "đi", "này", "cái", "nó", "rồi", "lại", "ra", "còn",
    "phải", "mình", "ơi", "nào", "thế", "sao", "ai", "ở", "đâu", "đó",
    "để", "thôi", "vậy", "với", "chỉ", "cả", "đã", "vào", "nên", "nữa",
    "từ", "khi", "đến", "trong", "vì", "cứ", "sau", "con", "gì", "rất",
    "quá", "đang", "mới", "hơn", "luôn", "được", "hoặc", "làm",
    "thấy", "bị", "nhé", "nha", "hả", "thật",

    # Nhóm bổ sung — động từ / phó từ trung tính (từ phân tích vocab)
    "cũng",  "biết",  "muốn",
    "đúng",  "vẫn",   "cần", "chứ",
       "sẽ",    "đây",
    "đấy",   "lắm",   "hết",   "theo",  "nhau",
    "vừa",   "thêm",  "tất",
])

# ==========================================
# CÁC HÀM TIỀN XỬ LÝ
# ==========================================

VIETNAMESE_ACCENT_REPLACEMENTS = {
    "oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa",
    "oè": "òe", "oé": "óe", "oẻ": "ỏe", "oẽ": "õe", "oẹ": "ọe",
    "uỳ": "ùy", "uý": "úy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy",
    "hoà": "hòa", "hoá": "hóa", "hoả": "hỏa", "hoã": "hõa", "hoạ": "họa",
    "toà": "tòa", "toá": "tóa", "toả": "tỏa", "toã": "tõa", "toạ": "tọa",
    "loà": "lòa", "loá": "lóa", "loả": "lỏa", "loã": "lõa", "loạ": "lọa",
    "xoà": "xòa", "xoá": "xóa", "xoả": "xỏa", "xoã": "xõa", "xoạ": "xọa",
    "đoà": "đòa", "đoá": "đóa", "đoả": "đỏa", "đoã": "đõa", "đoạ": "đọa",
    "khoà": "khòa", "khoá": "khóa", "khoả": "khỏa", "khoã": "khõa", "khoạ": "khọa",
    "ngoà": "ngòa", "ngoá": "ngóa", "ngoả": "ngỏa", "ngoã": "ngõa", "ngoạ": "ngọa",
    "thoà": "thòa", "thoá": "thóa", "thoả": "thỏa", "thoã": "thõa", "thoạ": "thọa",
    "thuỷ": "thủy", "thuý": "thúy", "thuỳ": "thùy", "thuỹ": "thũy", "thuỵ": "thụy",
    "huỷ": "hủy", "huý": "húy", "huỳ": "hùy", "huỹ": "hũy", "huỵ": "hụy",
    "luỷ": "lủy", "luý": "lúy", "luỳ": "lùy", "luỹ": "lũy", "luỵ": "lụy",
}

def normalize_vietnamese_accents(text: str) -> str:
    """Chuẩn hóa kiểu gõ dấu cũ sang mới (hoà -> hòa, uý -> úy)."""
    for old, new in VIETNAMESE_ACCENT_REPLACEMENTS.items():
        text = text.replace(old, new)
        text = text.replace(old.upper(), new.upper())
        text = text.replace(old.capitalize(), new.capitalize())
    return text

def unicode_normalization(text: str) -> str:
    """
    [FIX 1] Bước 1: Đồng nhất bảng mã về chuẩn Unicode NFC và chuẩn hóa kiểu gõ dấu tiếng Việt.
    Xử lý lỗi dấu thanh Tổ hợp/Dựng sẵn — rào cản kinh điển trong NLP tiếng Việt.
    Phát hiện: 541/24048 dòng train (2.25%) bị lỗi NFD.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFC', text)
    text = normalize_vietnamese_accents(text)
    return text


def remove_noise(text: str) -> str:
    """Bước 3: Lọc nhiễu kỹ thuật — URL, @mention, #hashtag."""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[@#]\w+', '', text)
    return text


def is_word_token(token: str) -> bool:
    """Token chữ/số/underscore/hyphen để áp dụng từ điển chuẩn hóa."""
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
            token_lower = token.lower()
            replacement = lexicon.get(token_lower, token)
            if replacement == token:
                normalized_tokens.append(token)
            else:
                normalized_tokens.extend(replacement.split())
        else:
            normalized_tokens.append(token)
    return normalized_tokens


def separate_emoji_boundaries(text: str) -> str:
    """Tách emoji thô khỏi chữ ở biên token trước khi gán nhãn emoji."""
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
    """Fallback demojize thành token an toàn, không dùng dấu ':' để tránh vỡ token."""
    alias = emoji.demojize(emoji_text, delimiters=("", ""))
    alias = re.sub(r'[^0-9a-zA-Z_]+', '_', alias).strip('_')
    if not alias:
        return "EMOJI_KHAC"
    return f"EMOJI_ALIAS_{alias.upper()}"


def extract_emoji_features(text: str) -> str:
    """
        [FIX 4] Bước 4: Khai thác đặc trưng emoji trung tính.

    Thay đổi so với v1:
      - Chèn khoảng trắng quanh mỗi emoji trước khi xử lý
        → tránh 😂😡 bị ghép thành 1 token dính
            - Chuyển toàn bộ emoji sang token EMOJI_ALIAS_* bằng demojize()
    """
    text = separate_emoji_boundaries(text)

    # Xử lý theo cụm emoji (grapheme) để không tách rời variation selector như '❤️'
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
    """
    [FIX 3] Bước 5: Chuẩn hóa ký tự / dấu câu kéo dài.

    Thay đổi so với v1:
      Lớp 1 (MỚI): Dấu câu lặp lại (510+ lần trong dataset)
                   :))) → :)   ... → .   !!! → !   ??? → ?
      Lớp 2 (GIỮ): Chữ cái tiếng Việt lặp lại
                   gìiiii → gì   buồnnnn → buồn
    """
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

    # Lớp 2a: cắt chữ cái lặp ≥ 2 lần về còn 1 (ngoại trừ chữ 'o')
    text = re.sub(
        r'([a-np-zàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ])\1+',
        r'\1', text, flags=re.IGNORECASE
    )
    # Lớp 2b: cắt chữ 'o' lặp ≥ 3 lần về còn 2 (để giữ xoong, coong)
    text = re.sub(
        r'(o|ò|ó|ỏ|õ|ọ|ô|ồ|ố|ổ|ỗ|ộ|ơ|ờ|ớ|ở|ỡ|ợ)\1{2,}',
        r'\1\1', text, flags=re.IGNORECASE
    )
    return text


# def replace_slang_and_abbreviations(text: str) -> str:
#     """
#     [FIX 2] Bước 6: Chuẩn hóa từ lóng và viết tắt — 2 từ điển riêng biệt.

#     Thay đổi so với v1:
#       - ABBREV_DICT (viết tắt) áp dụng TRƯỚC
#       - SLANG_DICT  (từ lóng)  áp dụng SAU
#       Lý do: viết tắt thường là prefix của từ lóng (vd: "dc" → "được",
#       không nhầm với "dcm" → "địt cụ mày" vì "dcm" khớp exact match)
#     """
#     words = tokenize_keep_punctuation(text)
#     # Áp dụng ABBREV_DICT trước
#     words = apply_lexicon(words, ABBREV_DICT)
#     # Áp dụng SLANG_DICT sau
#     words = apply_lexicon(words, SLANG_DICT)
#     return ' '.join(words)
def replace_slang_and_abbreviations_cased(
    text: str,
    slang_dict: dict[str, str] = SLANG_DICT,
    abbrev_dict: dict[str, str] = ABBREV_DICT,
) -> str:
    """
    Hàm thay thế từ lóng/viết tắt nhưng BẢO TOÀN cấu trúc viết hoa/thường 
    của các từ ngữ khác trong câu, sử dụng bộ tách từ giữ dấu câu.
    """
    words = tokenize_keep_punctuation(text)
    words = apply_lexicon(words, abbrev_dict)
    words = apply_lexicon(words, slang_dict)
    return " ".join(words)


def replace_compound_words(text: str) -> str:
    """Gộp các cụm từ ghép phổ biến thành một token có dấu underscore."""
    for pattern, replacement in COMPOUND_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def segment_and_remove_stopwords(text: str, remove_stopwords: bool = False) -> str:
    """Bước 7: Phân đoạn từ (underthesea) và loại bỏ từ dừng nếu remove_stopwords=True."""
    tokenized_text = word_tokenize(text, format="text")
    if isinstance(tokenized_text, list):
        words = tokenized_text
    else:
        words = tokenized_text.split()

    if remove_stopwords:
        filtered = [
            w for w in words
            if not (
                w.lower() in STOPWORDS
                and w.lower() not in NEGATION_KEEP_TOKENS
                and not is_special_token(w)
            )
        ]
        return ' '.join(filtered)
    return ' '.join(words)


def run_quick_regression_checks() -> None:
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
        text = segment_and_remove_stopwords(text)
        text = re.sub(r'\s+', ' ', text).strip()

        print(f"[{idx}] IN : {sample}")
        print(f"    OUT: {text}")


# ==========================================
# PIPELINE & LOGGING
# ==========================================

def run_pipeline(df: pd.DataFrame, text_column: str, remove_stopwords: bool = False) -> pd.DataFrame:
    """Thực thi pipeline 8 bước và in log."""
    print("  [1/7] Đồng nhất bảng mã Unicode NFC (fix NFD/mixed)...")
    df['clean_text'] = df[text_column].apply(unicode_normalization)

    # print("  [2/7] Chuyển về chữ thường...")
    # df['clean_text'] = df['clean_text'].str.lower()

    print("  [3/7] Lọc nhiễu kỹ thuật (URL, @mention, #hashtag)...")
    df['clean_text'] = df['clean_text'].apply(remove_noise)

    print("  [4/7] Trích xuất đặc trưng Emoji (demojize tiếng Anh + khoảng trắng)...")
    df['clean_text'] = df['clean_text'].apply(extract_emoji_features)

    print("  [5/7] Chuẩn hóa ký tự kéo dài (dấu câu + chữ cái)...")
    df['clean_text'] = df['clean_text'].apply(normalize_lengthened_words)

    print("  [6/7] Chuẩn hóa Teencode: ABBREV_DICT → SLANG_DICT...")
    df['clean_text'] = df['clean_text'].apply(replace_slang_and_abbreviations_cased)

    print("  [7/8] Gộp từ ghép phổ biến thành token underscore...")
    df['clean_text'] = df['clean_text'].apply(replace_compound_words)

    print(f"  [8/8] Phân đoạn từ (loại bỏ stopwords: {remove_stopwords})...")
    df['clean_text'] = df['clean_text'].apply(lambda x: segment_and_remove_stopwords(x, remove_stopwords=remove_stopwords))

    print("  [*] Dọn khoảng trắng thừa...")
    df['clean_text'] = df['clean_text'].apply(
        lambda x: re.sub(r'\s+', ' ', str(x)).strip()
    )
    return df


def process_dataset(dataset_dir: str | None = None, remove_stopwords: bool = False):
    """Hàm main: đọc origin_tran/dev/test → xử lý → lưu preprocessed_*.csv."""
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
        description='preprocessing_v2 — đầy đủ 4 fix: Unicode / Lexicon / Lengthening / Emoji'
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
        action='store_true',
        help='Loại bỏ stopwords (mặc định: False để có hiệu năng Transformer tốt nhất)'
    )
    args = parser.parse_args()

    if args.self_test:
        run_quick_regression_checks()

    process_dataset(args.dataset_dir, remove_stopwords=args.remove_stopwords)
