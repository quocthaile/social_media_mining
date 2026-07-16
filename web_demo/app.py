import os
import sys

# Monkeypatch tokenizers.models.Unigram for Python 3.14 compatibility
try:
    import tokenizers.models
    original_unigram = tokenizers.models.Unigram
    tokenizers.models.Unigram = lambda *args, **kwargs: original_unigram(list(kwargs.pop('vocab').items()), *args, **kwargs) if 'vocab' in kwargs and isinstance(kwargs['vocab'], dict) else original_unigram(*args, **kwargs)
except Exception:
    pass

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
import emoji
from underthesea import word_tokenize

SPECIAL_TOKEN_PREFIXES = ("EMOJI_", "PUNC_", "EMOTICON_")
LEXICON_TOKEN_PATTERN = re.compile(r"[\w_+-]+|[^\w\s]", flags=re.UNICODE)
NEGATION_KEEP_TOKENS = {"không", "chẳng", "chả", "đừng", "chưa"}

_SLANG_DICT_RAW = {
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

_ABBREV_DICT_RAW = {
    "ko":    "không",
    "k":     "không",
    "hk":    "không",
    "hok":   "không",
    "kg":    "không",
    "m":     "mày",
    "t":     "tao",
    "e":     "em",
    "tui":   "tôi",
    "mk":    "mình",
    "ck":    "chồng",
    "vk":    "vợ",
    "b":     "bạn",
    "a":     "anh",
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
    "nguoi": "người",
    "vn":    "việt_nam",
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
    "gia định": "gia_đình",
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

STOPWORDS = set([
    "và", "là", "như", "thì", "mà", "nếu", "có", "các", "những", "của",
    "cho", "đi", "này", "cái", "nó", "rồi", "lại", "ra", "còn",
    "phải", "mình", "ơi", "nào", "thế", "sao", "ai", "ở", "đâu", "đó",
    "để", "thôi", "vậy", "với", "chỉ", "cả", "đã", "vào", "nên", "nữa",
    "từ", "khi", "đến", "trong", "vì", "cứ", "sau", "con", "gì", "rất",
    "quá", "đang", "mới", "hơn", "luôn", "được", "hoặc", "làm",
    "thấy", "bị", "nhé", "nha", "hả", "thật",
    "cũng",  "biết",  "muốn",
    "đúng",  "vẫn",   "cần", "chứ",
    "sẽ",    "đây",
    "đấy",   "lắm",   "hết",   "theo",  "nhau",
    "vừa",   "thêm",  "tất",
])

VIETNAMESE_ACCENT_REPLACEMENTS = {
    "oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa",
    "oè": "òe", "oé": "óe", "oẻ": "ỏe", "oẽ": "õe", "oẹ": "ọe",
    "uỳ": "ùy", "uý": "úy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy",
    "hoà": "hòa", "hoá": "hóa", "hoả": "hỏa", "hoã": "hõa", "hoạ": "họa",
    "toà": "tòa", "toá": "tóa", "toả": "tòa", "toã": "tõa", "toạ": "tọa",
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
    for old, new in VIETNAMESE_ACCENT_REPLACEMENTS.items():
        text = text.replace(old, new)
        text = text.replace(old.upper(), new.upper())
        text = text.replace(old.capitalize(), new.capitalize())
    return text

def unicode_normalization(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFC', text)
    text = normalize_vietnamese_accents(text)
    return text

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
    return LEXICON_TOKEN_PATTERN.findall(text)

def apply_lexicon(tokens: list[str], lexicon: dict[str, str]) -> list[str]:
    normalized_tokens = []
    for token in tokens:
        if is_word_token(token) and not is_special_token(token):
            token_lower = token.lower()
            if token_lower in lexicon:
                rep = lexicon[token_lower]
                if token.isupper():
                    rep = rep.upper()
                elif token[0].isupper():
                    rep = rep.capitalize()
                normalized_tokens.append(rep)
            else:
                normalized_tokens.append(token)
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
        return "EMOJI_UNK"
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
            return f" {tag_prefix}_{level} "
        return _repl
    text = re.sub(r'\.{3,}', _punct_with_intensity('…', 'PUNC_ELLIPSIS'), text)
    text = re.sub(r'!{2,}', _punct_with_intensity('!', 'PUNC_EXCLAM'), text)
    text = re.sub(r'\?{2,}', _punct_with_intensity('?', 'PUNC_QUESTION'), text)
    text = re.sub(r'([,;:\)\(\]\[><~\-=])\1+', r' \1 ', text)
    text = re.sub(
        r'([a-np-zàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ])\1+',
        r'\1', text, flags=re.IGNORECASE
    )
    text = re.sub(
        r'(o|ò|ó|ỏ|õ|ọ|ô|ồ|ố|ổ|ỗ|ộ|ơ|ờ|ớ|ở|ỡ|ợ)\1{2,}',
        r'\1\1', text, flags=re.IGNORECASE
    )
    return text

def replace_slang_and_abbreviations_cased(
    text: str,
    slang_dict: dict[str, str] = SLANG_DICT,
    abbrev_dict: dict[str, str] = ABBREV_DICT,
) -> str:
    words = tokenize_keep_punctuation(text)
    words = apply_lexicon(words, abbrev_dict)
    words = apply_lexicon(words, slang_dict)
    return " ".join(words)

def replace_compound_words(text: str) -> str:
    for pattern, replacement in COMPOUND_PATTERNS:
        text = pattern.sub(replacement, text)
    return text

def segment_and_remove_stopwords(text: str, remove_stopwords: bool = False) -> str:
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

# FEATURE EXTRACTION FOR WEB_DEMO
EMOJI_TAG_PATTERN = re.compile(r"\bEMOJI_[A-Z_]+\b")
EMOJI_ALIAS_PATTERN = re.compile(r":\s*[a-z0-9_+\-]+\s*:", re.IGNORECASE)
ELONGATED_PATTERN = re.compile(r'(.)\1{2,}')

BAD_WORDS_LIST = [
    "lồn", "đéo", "địt", "đkm", "vcl", "cặc", "ngu", "chó", "đĩ", 
    "điếm", "đảng", "cộng sản", "phản động", "cc", "cđm", "vl", 
    "đm", "dkm", "đĩ điếm", "đĩ thoã", "coin card", "éo", "nham lon",
    "lủ chó", "nhảm lồn", "vãi lồn", "vãi cả lồn", "địt mẹ", "đụ", 
    "đụ má", "con card", "concard", "củ cặc", "xạo lồn", "tinh trùng", 
    "bê đê", "ml", "sml", "óc chó", "đực rựa", "đm", "cmm", "dcm"
]
BAD_WORDS_PATTERN = re.compile(r'\b(?:' + '|'.join(map(re.escape, BAD_WORDS_LIST)) + r')\b', re.IGNORECASE)

def count_emoji_features(text: str) -> int:
    return len(EMOJI_TAG_PATTERN.findall(text)) + len(EMOJI_ALIAS_PATTERN.findall(text))

def count_punctuation(text: str) -> int:
    return sum(1 for ch in text if unicodedata.category(ch).startswith("P"))

def count_bad_words(text: str) -> int:
    clean_text = text.replace('_', ' ').lower()
    return len(BAD_WORDS_PATTERN.findall(clean_text))

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
    }

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
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

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
                try:
                    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
                except Exception:
                    tokenizer = AutoTokenizer.from_pretrained(checkpoint["model_name"])
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

    def predict_batch(self, texts, batch_size=32):
        """Suy luận (Inference) theo lô (batch) để tối ưu hiệu năng."""
        if not self.model:
            return []

        results = []
        # 1. Tiền xử lý
        cleaned_texts = []
        remove_sw = True if self.model_type in ["gru", "textcnn"] else False
        
        for text in texts:
            clean_text = unicode_normalization(text)
            clean_text = remove_noise(clean_text)
            clean_text = extract_emoji_features(clean_text)
            clean_text = normalize_lengthened_words(clean_text)
            clean_text = replace_slang_and_abbreviations_cased(clean_text)
            clean_text = replace_compound_words(clean_text)
            clean_text = segment_and_remove_stopwords(clean_text, remove_stopwords=remove_sw).strip()
            cleaned_texts.append(clean_text)

        feature_columns = self.checkpoint["feature_columns"]
        max_len = self.checkpoint["args"].get("max_len", 128)
        use_threshold_moving = self.checkpoint.get("args", {}).get("use_threshold_moving", True)

        # 2. Xử lý theo từng batch
        for idx in range(0, len(texts), batch_size):
            batch_orig = texts[idx:idx+batch_size]
            batch_clean = cleaned_texts[idx:idx+batch_size]

            # Trích xuất đặc trưng meta
            batch_meta = []
            for ct in batch_clean:
                meta_feat = extract_feature_row(ct)
                batch_meta.append([meta_feat[col] for col in feature_columns])
            meta_tensor = torch.tensor(batch_meta, dtype=torch.float, device=device)

            if self.model_type == "transformer":
                transformer_texts = [ct.replace("_", " ") for ct in batch_clean]
                inputs = self.tokenizer(
                    transformer_texts,
                    truncation=True,
                    padding="max_length",
                    max_length=max_len,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

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
                pad_idx = self.vocab.get("<pad>", 0)
                unk_idx = self.vocab.get("<unk>", 1)

                for ct in batch_clean:
                    tokens = ct.split()
                    token_ids = [self.vocab.get(t, unk_idx) for t in tokens[:max_len]]
                    if len(token_ids) < max_len:
                        token_ids += [pad_idx] * (max_len - len(token_ids))
                    batch_tokens.append(token_ids)

                input_tensor = torch.tensor(batch_tokens, dtype=torch.long, device=device)
                with torch.no_grad():
                    logits = self.model(input_tensor, meta_tensor)

            probs = torch.softmax(logits, dim=1).cpu().numpy()

            # 3. Phân lớp & Highlight từ độc hại cho từng phần tử
            for i, p in enumerate(probs):
                orig_text = batch_orig[i]
                ct = batch_clean[i]

                if use_threshold_moving:
                    if p[2] > 0.25:
                        pred_label_id = 2
                    elif p[1] > 0.25:
                        pred_label_id = 1
                    else:
                        pred_label_id = int(np.argmax(p))
                else:
                    pred_label_id = int(np.argmax(p))

                highlighted_text, detected_bad_words = highlight_bad_words(orig_text)
                meta_feat_single = extract_feature_row(ct)

                results.append({
                    "raw_text": orig_text,
                    "clean_text": ct,
                    "highlighted_text": highlighted_text,
                    "detected_bad_words": detected_bad_words,
                    "label_id": pred_label_id,
                    "label_name": label_names[pred_label_id],
                    "probs": {
                        "clean": float(p[0]),
                        "offensive": float(p[1]),
                        "hate": float(p[2]),
                    },
                    "meta_features": {k: float(v) for k, v in meta_feat_single.items()},
                })

        return results

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

                raw_texts = df[text_col].fillna("").astype(str).tolist()
                
                # Chạy dự đoán dạng batch để tăng tốc
                batch_results = manager.predict_batch(raw_texts)
                
                results = []
                for res in batch_results:
                    results.append({
                        "text": res["raw_text"],
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

    results = manager.predict_batch(texts)
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
