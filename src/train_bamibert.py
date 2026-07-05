"""Fine-tune BamiBERT (Qualcomm AI Research) on the ViHSD hate speech dataset.

BamiBERT is a BERT-based ("base"-architecture, 12 Transformer layers) language
model for Vietnamese, pre-trained from scratch by Qualcomm AI Research on a
clean, 129GB general-domain Vietnamese text corpus for 20 epochs using the
RoBERTa pre-training recipe (dynamic masking, no next-sentence-prediction).

Key properties (see paper / model card below):
- Extended context length of up to 2048 tokens (vs. 256 for PhoBERT).
- Operates directly on raw, un-segmented Vietnamese text (no external word
  segmenter such as VnCoreNLP/RDRSegmenter is required, unlike PhoBERT).
- Vietnamese-specific byte-level BPE tokenizer extended from PhoGPT's
  tokenizer (vocabulary size 20,481 token types incl. <mask>).
- Reported as new SOTA (or near-SOTA) among "base"-sized Vietnamese encoders
  across 8 Vietnamese benchmarks, ranking #1 on 11/15 metrics and #2 on 3/15,
  notably surpassing PhoBERT on ViNLI (+3.01 Accuracy / +3.10 F1) and on
  UIT-ViSFD aspect-based sentiment classification (+5.48 F1).

References:
- Paper: "BamiBERT: A New BERT-based Language Model for Vietnamese"
  Nguyen, Pham, Tran, Nguyen (2026). arXiv:2607.02259.
  https://arxiv.org/abs/2607.02259
- Model card / weights: https://huggingface.co/Qualcomm-AI-Research/BamiBERT

Because BamiBERT consumes raw (non-word-segmented) text, this script reuses
the "transformer_text" feature column (underscores from word segmentation
replaced back with spaces) that is already produced by build_feature_dataset*.py
for the other multilingual/cased transformer baselines.
"""

from train_phobert import run_experiment


if __name__ == "__main__":
    print("[DEBUG][BamiBERT] Wrapper started")
    print("[DEBUG][BamiBERT] Delegating to shared feature-augmented transformer pipeline")
    run_experiment(
        default_model_name="Qualcomm-AI-Research/BamiBERT",
        default_output_subdir="bamibert",
        run_name="BamiBERT",
        default_text_column="transformer_text"
    )
    print("[DEBUG][BamiBERT] Wrapper finished")
