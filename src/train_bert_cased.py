from train_phobert import run_experiment


if __name__ == "__main__":
    print("[DEBUG][BERT-cased] Wrapper started")
    print("[DEBUG][BERT-cased] Delegating to shared feature-augmented transformer pipeline")
    run_experiment(
        default_model_name="bert-base-multilingual-cased",
        default_output_subdir="bert_cased",
        run_name="BERT cased",
    )
    print("[DEBUG][BERT-cased] Wrapper finished")
