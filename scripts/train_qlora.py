"""QLoRA supervised fine-tuning for JSONL instruction data.

Each input row must contain either `messages` (chat format) or `text`.
The base model and adapter are user-configurable; no weights are bundled.
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="JSONL file with messages or text")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--output", default="artifacts/qlora-adapter")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit("Install optional dependencies with `pip install -r requirements-training.txt`.") from exc

    if not torch.cuda.is_available():
        raise SystemExit("QLoRA training requires a CUDA GPU with a supported bitsandbytes build.")
    os.makedirs(args.output, exist_ok=True)
    data = load_dataset("json", data_files=args.dataset, split="train")
    if not data or not ({"text", "messages"} & set(data.column_names)):
        raise SystemExit("Dataset must contain at least one row with a `text` or `messages` field.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map="auto",
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=data,
        peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                               target_modules="all-linear", task_type="CAUSAL_LM"),
        args=SFTConfig(output_dir=args.output, num_train_epochs=args.epochs,
                       max_length=args.max_length, per_device_train_batch_size=1,
                       gradient_accumulation_steps=8, learning_rate=2e-4,
                       logging_steps=10, save_strategy="epoch", report_to="none"),
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
