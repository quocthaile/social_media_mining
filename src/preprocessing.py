import os
import argparse
import pandas as pd
import re
import unicodedata
from underthesea import word_tokenize
import emoji

# ==========================================
# CẤU HÌNH TỪ ĐIỂN VÀ THAM SỐ
# ==========================================
# 1. Từ điển từ lóng và viết tắt (Cập nhật theo ngữ liệu thực tế)
SLANG_DICT = {
    "dell": "đếch", "đéo": "đếch", "đm": "địt mẹ", "địt": "chửi thề",
    "cc": "cục cức", "lol": "lồn", "eo": "éo",
    "ko": "không", "k": "không", "khong": "không",
    "dc": "được", "đc": "được",
    "vs": "với", "3": "ba", "m": "mày", "t": "tao"
}

# 2. Danh sách stopwords tiếng Việt (Cơ bản)
STOPWORDS = set(["và", "là", "như", "thì", "mà", "nếu", "có", "các", "những", "của"])

# ==========================================
# CÁC HÀM TIỀN XỬ LÝ (THEO ĐỀ XUẤT BÁO CÁO)
# ==========================================
def unicode_normalization(text):
    """Giai đoạn 2: Đồng nhất bảng mã về chuẩn Unicode NFC"""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize('NFC', text)

def remove_noise(text):
    """Giai đoạn 1: Lọc nhiễu kỹ thuật (URL, hashtag, mention tags)"""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#\w+', '', text)
    return text

def extract_emoji_features(text):
    """Giai đoạn 2: Trích xuất đặc trưng Emoji thành token văn bản"""
    # Bỏ tham số language='vi' để tránh lỗi. Sử dụng alias mặc định (Tiếng Anh)
    return emoji.demojize(text)

def normalize_lengthened_words(text):
    """Giai đoạn 2: Chuẩn hóa từ kéo dài (ví dụ: gìiiiii -> gì)"""
    return re.sub(r'([a-zàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ])\1{2,}', r'\1', text)

def replace_slang_and_abbreviations(text):
    """Giai đoạn 2: Thay thế từ lóng và viết tắt (Teencode)"""
    words = text.split()
    normalized_words = [SLANG_DICT.get(word, word) for word in words]
    return " ".join(normalized_words)

def segment_and_remove_stopwords(text):
    """Giai đoạn 3: Phân đoạn từ và loại bỏ từ dừng"""
    # Tokenize bằng underthesea
    tokenized_text = word_tokenize(text, format="text")
    
    # Loại bỏ stopwords
    words = tokenized_text.split()
    filtered_words = [word for word in words if word.lower() not in STOPWORDS]
    
    return " ".join(filtered_words)

# ==========================================
# QUẢN LÝ PIPELINE & LOGGING
# ==========================================
def run_pipeline(df, text_column):
    """Thực thi pipeline theo từng bước và in log ra màn hình"""
    print("  [1/7] Đang đồng nhất bảng mã Unicode...")
    df['clean_text'] = df[text_column].apply(unicode_normalization)
    
    print("  [2/7] Đang chuyển đổi chữ in thường...")
    df['clean_text'] = df['clean_text'].str.lower()
    
    print("  [3/7] Đang lọc nhiễu kỹ thuật (URLs, Mentions, Hashtags)...")
    df['clean_text'] = df['clean_text'].apply(remove_noise)
    
    print("  [4/7] Đang trích xuất đặc trưng phân cực Emoji...")
    df['clean_text'] = df['clean_text'].apply(extract_emoji_features)
    
    print("  [5/7] Đang chuẩn hóa các từ kéo dài ký tự...")
    df['clean_text'] = df['clean_text'].apply(normalize_lengthened_words)
    
    print("  [6/7] Đang chuẩn hóa Teencode, từ lóng và viết tắt...")
    df['clean_text'] = df['clean_text'].apply(replace_slang_and_abbreviations)
    
    print("  [7/7] Đang phân đoạn từ và loại bỏ Stopwords (Tiến trình này mất chút thời gian)...")
    df['clean_text'] = df['clean_text'].apply(segment_and_remove_stopwords)
    
    print("  [*] Dọn dẹp khoảng trắng thừa...")
    df['clean_text'] = df['clean_text'].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())
    
    return df

def process_dataset(dataset_dir=None):
    """Hàm Main chạy toàn bộ quy trình cho Train, Dev, Test.

    Nếu `dataset_dir` không được cung cấp, mặc định là thư mục
    dataset-vihsd nằm song song với thư mục `src`.
    Kết quả lưu về cùng thư mục với tiền tố `preprocessed_`.
    """
    TEXT_COLUMN = 'free_text'

    # Xác định thư mục dữ liệu mặc định
    if dataset_dir is None:
        dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset-vihsd'))

    try:
        print(f"--- BẮT ĐẦU ĐỌC DỮ LIỆU TỪ: {dataset_dir} ---")
        train_path = os.path.join(dataset_dir, 'train.csv')
        dev_path = os.path.join(dataset_dir, 'dev.csv')
        test_path = os.path.join(dataset_dir, 'test.csv')

        df_train = pd.read_csv(train_path)
        df_dev = pd.read_csv(dev_path)
        df_test = pd.read_csv(test_path)

        print(f"\n▶ ĐANG XỬ LÝ TẬP TRAIN ({len(df_train)} dòng)...")
        df_train = run_pipeline(df_train, TEXT_COLUMN)

        print(f"\n▶ ĐANG XỬ LÝ TẬP VALIDATION/DEV ({len(df_dev)} dòng)...")
        df_dev = run_pipeline(df_dev, TEXT_COLUMN)

        print(f"\n▶ ĐANG XỬ LÝ TẬP TEST ({len(df_test)} dòng)...")
        df_test = run_pipeline(df_test, TEXT_COLUMN)

        # Lưu kết quả, đặt tiền tố preprocessed_
        print("\n--- ĐANG LƯU KẾT QUẢ ---")
        out_train = os.path.join(dataset_dir, 'preprocessed_train.csv')
        out_dev = os.path.join(dataset_dir, 'preprocessed_dev.csv')
        out_test = os.path.join(dataset_dir, 'preprocessed_test.csv')

        df_train[['clean_text', 'label_id']].to_csv(out_train, index=False)
        df_dev[['clean_text', 'label_id']].to_csv(out_dev, index=False)
        df_test[['clean_text', 'label_id']].to_csv(out_test, index=False)

        print(f"✅ TIỀN XỬ LÝ HOÀN TẤT! Kết quả: {out_train}, {out_dev}, {out_test}")

    except Exception as e:
        print(f"\n❌ [LỖI NGHIÊM TRỌNG]: {e}")

# Chạy chương trình
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess dataset files in dataset directory')
    parser.add_argument('--dataset_dir', '-d', help='Path to dataset directory (default: ../dataset-vihsd)', default=None)
    args = parser.parse_args()
    process_dataset(args.dataset_dir)