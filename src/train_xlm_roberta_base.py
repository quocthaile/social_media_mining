from train_phobert import run_experiment


if __name__ == "__main__":
    print("[DEBUG][XLM-R-base] Wrapper started")
    print("[DEBUG][XLM-R-base] Delegating to shared feature-augmented transformer pipeline")
    run_experiment(
        default_model_name="xlm-roberta-base",
        default_output_subdir="xlm_roberta_base",
        run_name="XLM-R base",
        default_text_column="transformer_text",
        model_prefix="xlmr"
    )
    print("[DEBUG][XLM-R-base] Wrapper finished")