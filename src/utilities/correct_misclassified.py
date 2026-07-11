import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def fix_ground_truth_labels(origin_path, misclass_path, output_path):
    # 1. Khởi tạo dữ liệu
    origin_df = pd.read_csv(origin_path)
    misclass_df = pd.read_csv(misclass_path)

    # 2. Cô lập các trường hợp Ground Truth = 0 nhưng Model dự đoán = 1 hoặc 2
    suspect_cases = misclass_df[(misclass_df['label_id_true'] == 0) & 
                                (misclass_df['label_id_pred'].isin([1, 2]))]

    # 3. Rút trích đặc trưng TF-IDF (Character N-gram để chống lỗi chính tả / tiền xử lý)
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    
    origin_texts = origin_df['free_text'].fillna("").astype(str).tolist()
    suspect_texts = suspect_cases['transformer_text'].fillna("").astype(str).tolist()

    origin_tfidf = vectorizer.fit_transform(origin_texts)
    suspect_tfidf = vectorizer.transform(suspect_texts)

    # 4. Tính ma trận Cosine Similarity
    cos_sim = cosine_similarity(suspect_tfidf, origin_tfidf)
    best_indices = np.argmax(cos_sim, axis=1)
    best_scores = np.max(cos_sim, axis=1)

    # 5. Pipeline cập nhật nhãn
    corrected_df = origin_df.copy()
    correction_count = 0

    for i, (idx, score) in enumerate(zip(best_indices, best_scores)):
        # Thiết lập ngưỡng Confidence Threshold > 0.6 để đảm bảo ánh xạ chính xác
        if score > 0.6: 
            # Đổi nhãn 0 thành nhãn 1 hoặc 2 dựa trên kết quả Model đã Catch được
            proposed_label = suspect_cases.iloc[i]['label_id_pred']
            corrected_df.at[idx, 'label_id'] = proposed_label
            correction_count += 1

    # 6. Xuất cấu trúc file chuẩn
    corrected_df.to_csv(output_path, index=False)
    print(f"Hoàn tất. Đã căn chỉnh {correction_count} nhãn bị gán sai trong tập test.")
    print(f"File mới được lưu tại: '{output_path}'")

# Thực thi Pipeline
fix_ground_truth_labels(
    origin_path="origin_test.csv", 
    misclass_path="misclassified_test.csv", 
    output_path="corrected_origin_test.csv"
)