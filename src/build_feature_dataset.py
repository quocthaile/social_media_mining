import argparse
import os
import re
import unicodedata

import numpy as np
import pandas as pd


DEFAULT_INPUT_PREFIX = "preprocessed"
DEFAULT_OUTPUT_PREFIX = "features"

FEATURE_COLUMNS = [
    "feat_log_num_tokens",
    "feat_log_num_chars",
    "feat_avg_token_len",
    "feat_emoji_density",
    "feat_punct_density",
    "feat_upper_ratio",
    "feat_digit_ratio",
]


def debug(msg: str) -> None:
    print(f"[DEBUG][FeatureBuilder] {msg}")


def get_default_data_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset-vihsd"))


def normalize_whitespace(text: str) -> str:
    normalized = unicodedata.normalize("NFC", str(text)).strip()
    return re.sub(r"\s+", " ", normalized)


def count_emoji_aliases(text: str) -> int:
    # Matches both compact form (:smile:) and spaced form (: smile :).
    matches = re.findall(r":\s*[a-z0-9_+\-]+\s*:", text.lower())
    return len(matches)


def count_punctuation(text: str) -> int:
    return sum(1 for ch in text if unicodedata.category(ch).startswith("P"))


def extract_feature_row(text: str) -> dict:
    tokens = [tok for tok in text.split(" ") if tok]
    num_tokens = len(tokens)
    num_chars = len(text)

    avg_token_len = float(np.mean([len(tok) for tok in tokens])) if num_tokens > 0 else 0.0
    emoji_count = count_emoji_aliases(text)
    punct_count = count_punctuation(text)

    alpha_count = sum(1 for ch in text if ch.isalpha())
    upper_count = sum(1 for ch in text if ch.isalpha() and ch.isupper())
    digit_count = sum(1 for ch in text if ch.isdigit())

    feat = {
        "feat_log_num_tokens": float(np.log1p(num_tokens)),
        "feat_log_num_chars": float(np.log1p(num_chars)),
        "feat_avg_token_len": float(avg_token_len),
        "feat_emoji_density": float(emoji_count / max(num_tokens, 1)),
        "feat_punct_density": float(punct_count / max(num_chars, 1)),
        "feat_upper_ratio": float(upper_count / max(alpha_count, 1)),
        "feat_digit_ratio": float(digit_count / max(num_chars, 1)),
    }
    return feat


def build_one_split(input_path: str, output_path: str) -> None:
    debug(f"Reading: {input_path}")
    df = pd.read_csv(input_path)

    required_cols = {"clean_text", "label_id"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{input_path} must contain columns: {required_cols}")

    token_text = df["clean_text"].fillna("").astype(str).map(normalize_whitespace)
    transformer_text = token_text.map(lambda t: t.replace("_", " "))
    feature_df = token_text.map(extract_feature_row).apply(pd.Series)

    out_df = pd.DataFrame(
        {
            "tokens_text": token_text,
            "transformer_text": transformer_text,
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
    split_names = ["train", "dev", "test"]
    for split in split_names:
        input_name = f"{input_prefix}_{split}.csv"
        output_name = f"{output_prefix}_{split}.csv"
        input_path = os.path.join(data_dir, input_name)
        output_path = os.path.join(data_dir, output_name)
        build_one_split(input_path, output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build unified feature datasets from preprocessed CSV files "
            "for GRU, TextCNN, PhoBERT, BERT-cased, and DistilBERT-cased."
        )
    )
    parser.add_argument("--data_dir", type=str, default=get_default_data_dir())
    parser.add_argument("--input_prefix", type=str, default=DEFAULT_INPUT_PREFIX)
    parser.add_argument("--output_prefix", type=str, default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    debug(f"Arguments: {vars(args)}")
    process_dataset(
        data_dir=args.data_dir,
        input_prefix=args.input_prefix,
        output_prefix=args.output_prefix,
    )
    debug("All splits processed successfully")


if __name__ == "__main__":
    main()