import itertools
import subprocess
import re
import sys
import pandas as pd
import time
import argparse
from datetime import datetime
from pathlib import Path
import os

SRC_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent

# ==========================================
# KHÔNG GIAN TÌM KIẾM CHO TỪNG KIẾN TRÚC
# ==========================================
CONFIGS = {
    "textcnn": {
        "script": "train_textcnn.py",
        "grid": {
            "kernel_sizes": ["2,3,4", "3,4,5"],
            "num_filters": [128, 256],
            "lr": [1e-3, 5e-4]
        }
    },
    "gru": {
        "script": "train_gru.py",
        "grid": {
            "hidden_size": [128, 256],
            "num_layers": [1, 2],
            "lr": [1e-3, 5e-4]
        }
    },
    "transformer": {
        "script": "train_phobert.py", # Tệp master pipeline cho Transformer
        "grid": {
            # Quét thử nghiệm chéo các kiến trúc BERT khác nhau
            "model_name": [
                "vinai/phobert-base",
                "bert-base-multilingual-cased",
                "xlm-roberta-base",
                "distilbert-base-multilingual-cased"
            ],
            "lr": [2e-5, 3e-5],
            "batch_size": [16, 32]
        }
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="Master Grid Search for All Models")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        choices=["textcnn", "gru", "transformer"],
        help="Chọn kiến trúc để chạy Grid Search"
    )
    return parser.parse_args()

def run_grid_search(target_model):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] KHỞI ĐỘNG GRID SEARCH CHO: {target_model.upper()}")
    
    config = CONFIGS[target_model]
    script_name = config["script"]
    param_grid = config["grid"]

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    total_runs = len(combinations)
    print(f"Tổng số cấu hình cần kiểm thử: {total_runs}\n" + "="*50)

    results = []
    
    # Biểu thức chính quy bắt F1-macro dựa trên hàm print_confusion_and_scores chung
    metric_pattern = re.compile(r"F1-macro:\s*([0-9\.]+)")

    for idx, combination in enumerate(combinations, start=1):
        params = dict(zip(keys, combination))
        print(f"\n[{idx}/{total_runs}] Đang huấn luyện với cấu hình: {params}")
        
        # Chuyển đổi params dictionary thành tham số dòng lệnh CLI
        cmd = [sys.executable, str(SRC_DIR / script_name)]
        for k, v in params.items():
            cmd.extend([f"--{k}", str(v)])
            
        # Thêm các tham số cố định để ép epoch chạy nhanh hoặc thử nghiệm
        if target_model == "transformer":
            cmd.extend(["--epochs", "5"]) # Transformer hội tụ rất nhanh
        else:
            cmd.extend(["--epochs", "15"])
            
        start_time = time.time()

        try:
            # Khởi chạy tiến trình con, gom I/O để phân tích
            custom_env = os.environ.copy()
            custom_env["PYTHONIOENCODING"] = "utf-8"
            custom_env["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + custom_env.get("PYTHONPATH", "")
            process = subprocess.run(
                cmd,
                cwd=str(SRC_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=custom_env,
                check=True
            )
            
            output = process.stdout
            
            # Quét đầu ra để trích xuất F1-Macro
            matches = metric_pattern.findall(output)
            if matches:
                f1_score = float(matches[-1]) # Lấy giá trị F1-macro cuối cùng ở Test set
            else:
                print(f"  [CẢNH BÁO] Không tìm thấy điểm F1 trong log hệ thống.")
                f1_score = 0.0
                
        except subprocess.CalledProcessError as e:
            # e.stderr cũng cần áp dụng cơ chế đọc an toàn nếu có lỗi
            error_msg = e.stderr if e.stderr else "Lỗi không xác định."
            print(f"  [LỖI TIẾN TRÌNH] Huấn luyện thất bại:\n{error_msg}")
            f1_score = 0.0

        execution_time = time.time() - start_time
        print(f"  -> Hoàn tất trong {execution_time:.2f}s | F1-Macro: {f1_score:.4f}")
        
        result_entry = {**params, "f1_macro": f1_score, "time_seconds": round(execution_time, 2)}
        results.append(result_entry)

    print("\n" + "="*50 + "\n[HOÀN TẤT] BẢNG XẾP HẠNG SIÊU THAM SỐ:")
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="f1_macro", ascending=False).reset_index(drop=True)
    
    output_csv = OUTPUT_DIR / f"{target_model}_grid_search_results.csv"
    df_results.to_csv(output_csv, index=False)
    
    print(df_results.to_string())
    print(f"\nĐã xuất báo cáo chi tiết ra tệp: {output_csv}")
    print(f"CẤU HÌNH TỐI ƯU NHẤT: {df_results.iloc[0].to_dict()}")

if __name__ == "__main__":
    args = parse_args()
    run_grid_search(args.model)