from train_phobert import run_experiment


if __name__ == "__main__":
    print("[DEBUG][DistilBERT-cased] Wrapper started")
    print("[DEBUG][DistilBERT-cased] Delegating to shared feature-augmented transformer pipeline")
    run_experiment(
        default_model_name="distilbert-base-multilingual-cased",
        default_output_subdir="distilbert_cased",
        run_name="DistilBERT cased",
    )
    print("[DEBUG][DistilBERT-cased] Wrapper finished")
