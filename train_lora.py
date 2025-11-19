import os
import json
import argparse
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import evaluate


# ============== Configuration ==============
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="models/mt5-small")
    max_source_length: int = field(default=384)
    max_target_length: int = field(default=128)
    
    # LoRA config
    lora_r: int = field(default=8, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})


@dataclass
class DataArguments:
    input_file: str = field(default="data/processed_dataset.json")
    output_dir: str = field(default="data/splits")
    train_ratio: float = field(default=0.8)
    val_ratio: float = field(default=0.1)
    test_ratio: float = field(default=0.1)
    max_samples: Optional[int] = field(default=None)
    seed: int = field(default=42)


# ============== Data Loading & Splitting ==============
def load_and_split_dataset(data_args: DataArguments):
    print(f"\n{'='*60}")
    print(f"Loading dataset from: {data_args.input_file}")
    
    samples = []
    with open(data_args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if data_args.max_samples:
            data = data[:data_args.max_samples]
        
        for item in data:
            content = item.get("content", "").strip()
            summary = item.get("summary", "").strip()
            if content and summary:
                samples.append({"document": content, "summary": summary})
    
    print(f"Total valid samples: {len(samples)}")
    
    dataset = Dataset.from_list(samples)
    dataset = dataset.shuffle(seed=data_args.seed)
    
    train_size = int(len(dataset) * data_args.train_ratio)
    val_size = int(len(dataset) * data_args.val_ratio)
    
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, train_size + val_size))
    test_dataset = dataset.select(range(train_size + val_size, len(dataset)))
    
    print(f"Split sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    
    os.makedirs(data_args.output_dir, exist_ok=True)
    dataset_dict.save_to_disk(data_args.output_dir)
    print(f"Saved splits to: {data_args.output_dir}")
    print(f"{'='*60}\n")
    
    return dataset_dict


# ============== Preprocessing ==============
def preprocess_function(examples, tokenizer, model_args, prefix="summarize: "):
    inputs = [prefix + doc for doc in examples["document"]]
    targets = examples["summary"]
    
    # DEBUG: Print first example
    if len(inputs) > 0 and not hasattr(preprocess_function, '_printed'):
        print(f"\n[DEBUG Preprocess]")
        print(f"  Input (first 100 chars): {inputs[0][:100]}...")
        print(f"  Target: {targets[0]}")
        preprocess_function._printed = True
    
    model_inputs = tokenizer(
        inputs,
        max_length=model_args.max_source_length,
        truncation=True,
        padding=False,
        return_attention_mask=True
    )
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=model_args.max_target_length,
            truncation=True,
            padding=False
        )
    
    # DEBUG: Print tokenized output
    if len(labels["input_ids"]) > 0 and not hasattr(preprocess_function, '_printed_tokens'):
        print(f"  Label tokens (first 20): {labels['input_ids'][0][:]}")
        print(f"  Decoded label: {tokenizer.decode(labels['input_ids'][0][:], skip_special_tokens=False)}")
        preprocess_function._printed_tokens = True
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# ============== Metrics ==============
def compute_metrics(eval_pred, tokenizer, metric):
    """Compute ROUGE metrics with error handling"""
    predictions, labels = eval_pred
    
    # Fix OverflowError: Clean predictions before decoding
    # Replace negative values and out-of-vocab tokens with pad_token_id
    predictions = np.where(predictions < 0, tokenizer.pad_token_id, predictions)
    predictions = np.where(predictions >= len(tokenizer), tokenizer.pad_token_id, predictions)
    
    # Decode predictions
    try:
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    except Exception as e:
        print(f"Warning: Error decoding predictions: {e}")
        # Fallback: decode each individually and skip errors
        decoded_preds = []
        for pred in predictions:
            try:
                decoded_preds.append(tokenizer.decode(pred, skip_special_tokens=True))
            except:
                decoded_preds.append("")
    
    # Decode labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = ["\n".join(pred.strip().split()) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.strip().split()) for label in decoded_labels]
    
    result = metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=False
    )
    
    result = {key: value * 100 for key, value in result.items()}
    result = {k: round(v, 2) for k, v in result.items()}
    
    return result


# ============== Main Training Function ==============
def main():
    parser = argparse.ArgumentParser(description="Fine-tune mT5 with LoRA for Vietnamese summarization")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="models/mt5-small")
    parser.add_argument("--max_source_length", type=int, default=384)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank (4, 8, 16)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    
    # Data arguments
    parser.add_argument("--input_file", type=str, default="data/processed_dataset.json")
    parser.add_argument("--output_dir", type=str, default="data/splits")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--load_splits", action="store_true")
    
    # Training arguments
    parser.add_argument("--output_model_dir", type=str, default="./models/mt5-lora")
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=50, help="Warmup steps (will auto-adjust if too high)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    
    args = parser.parse_args()
    
    # ============== Check GPU ==============
    print("\n" + "="*60)
    print("DEVICE INFORMATION")
    print("="*60)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_count = torch.cuda.device_count()
        print(f"✓ GPU Available: YES")
        print(f"✓ GPU Count: {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"✓ GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        print(f"✓ CUDA Version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        gpu_count = 0
        print(f"⚠ GPU Available: NO")
        print(f"⚠ Using CPU")
    
    print("="*60 + "\n")
    
    # Create arguments objects
    model_args = ModelArguments(
        model_name_or_path=args.model_name,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    data_args = DataArguments(
        input_file=args.input_file,
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )
    
    # ============== Load Dataset ==============
    if args.load_splits and os.path.exists(args.output_dir):
        print(f"Loading existing splits from: {args.output_dir}")
        dataset_dict = DatasetDict.load_from_disk(args.output_dir)
    else:
        dataset_dict = load_and_split_dataset(data_args)
    
    # ============== Load Model & Tokenizer ==============
    print(f"Loading model and tokenizer: {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)
    
    # ============== Setup LoRA ==============
    print("\n" + "="*60)
    print("LORA CONFIGURATION")
    print("="*60)
    
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=["q", "v", "k", "o", "wi_0", "wi_1", "wo"],  # Attention + FFN layers
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Enable gradient checkpointing BEFORE moving to device (important for LoRA)
    if args.gradient_checkpointing:
        model.enable_input_require_grads()  # Required for gradient checkpointing with LoRA
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled (with input_require_grads)")
    
    model = model.to(device)
    print(f"Model moved to: {device}")
    
    print("="*60 + "\n")
    
    # ============== Preprocess Datasets ==============
    print("\n[DEBUG] Tokenizing datasets...")
    print(f"  Train samples: {len(dataset_dict['train'])}")
    
    tokenized_datasets = dataset_dict.map(
        lambda x: preprocess_function(x, tokenizer, model_args),
        batched=True,
        remove_columns=dataset_dict["train"].column_names,
        desc="Tokenizing"
    )
    
    print(f"\n[DEBUG] Tokenization complete")
    print(f"  First tokenized sample:")
    print(f"    input_ids length: {len(tokenized_datasets['train'][0]['input_ids'])}")
    print(f"    labels length: {len(tokenized_datasets['train'][0]['labels'])}")
    
    # ============== Setup Data Collator ==============
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8 if args.fp16 else None
    )
    
    # ============== Load ROUGE Metric ==============
    rouge_metric = evaluate.load("rouge")
    
    # ============== Calculate Training Steps ==============
    num_train_samples = len(tokenized_datasets["train"])
    steps_per_epoch = num_train_samples // (args.per_device_train_batch_size * args.gradient_accumulation_steps * max(gpu_count, 1))
    total_steps = steps_per_epoch * args.num_train_epochs
    
    # Giảm validation set xuống 5000 samples để eval nhanh hơn
    if len(tokenized_datasets["validation"]) > 5000:
        print(f"\n⚡ Reducing validation set from {len(tokenized_datasets['validation'])} to 5000 samples for faster eval")
        tokenized_datasets["validation"] = tokenized_datasets["validation"].select(range(5000))
    
    # CRITICAL FIX: Smart warmup adjustment based on dataset size
    if total_steps < 20:
        # Very small datasets (< 20 steps): NO WARMUP
        warmup_steps_to_use = 0
        print(f"\n⚠ Very small dataset detected ({total_steps} total steps)")
        print(f"✓ WARMUP DISABLED - Learning will start immediately with full LR\n")
    elif total_steps < 100:
        # Small datasets (20-100 steps): Minimal warmup (5%)
        warmup_steps_to_use = max(1, int(0.05 * total_steps))
        print(f"\n⚠ Small dataset detected ({total_steps} total steps)")
        print(f"✓ Using minimal warmup: {warmup_steps_to_use} steps (5%)\n")
    else:
        # Larger datasets: Use warmup for stability
        # Default 10% warmup, with minimum 50 steps for very large datasets
        recommended_warmup = max(int(0.1 * total_steps), min(50, total_steps // 20))
        
        if args.warmup_steps == 50:  # Using default value
            warmup_steps_to_use = recommended_warmup
            print(f"\n✓ Using recommended warmup: {warmup_steps_to_use} steps ({warmup_steps_to_use/total_steps*100:.1f}%)")
        else:
            # User specified warmup - validate it
            max_safe_warmup = int(0.2 * total_steps)  # Allow up to 20%
            warmup_steps_to_use = min(args.warmup_steps, max_safe_warmup)
            
            if warmup_steps_to_use != args.warmup_steps:
                print(f"\n⚠ WARNING: Requested warmup ({args.warmup_steps}) too high!")
                print(f"⚠ Total steps: {total_steps}, Max safe warmup: {max_safe_warmup}")
                print(f"✓ Adjusted warmup to: {warmup_steps_to_use}\n")
            else:
                print(f"\n✓ Using custom warmup: {warmup_steps_to_use} steps ({warmup_steps_to_use/total_steps*100:.1f}%)\n")
    
    print("\n" + "="*60)
    print("TRAINING STEPS CALCULATION")
    print("="*60)
    print(f"Total training samples: {num_train_samples}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps ({args.num_train_epochs} epochs): {total_steps}")
    print(f"Warmup steps: {warmup_steps_to_use} ({warmup_steps_to_use/total_steps*100:.1f}% of total)")
    print("="*60 + "\n")
    
    # ============== Training Arguments ==============
    use_fp16 = args.fp16  # Use FP16 if requested
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_model_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps_to_use,
        warmup_ratio=0.0,  # Disable warmup_ratio to use warmup_steps
        max_grad_norm=1.0,
        
        # Evaluation & Saving
        eval_strategy="steps",
        eval_steps=1000,  # Tăng từ 500 -> 1000 (eval ít hơn = nhanh hơn)
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        greater_is_better=True,
        eval_on_start=False,  # Bỏ qua eval đầu tiên
        
        # Generation config
        predict_with_generate=True,
        generation_max_length=model_args.max_target_length,
        generation_num_beams=4,
        
        # Optimization
        fp16=use_fp16,
        fp16_full_eval=False,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        
        # Logging
        logging_dir=f"{args.output_model_dir}/logs",
        logging_steps=100,
        logging_first_step=True,
        logging_nan_inf_filter=True,
        report_to=["tensorboard"],
        disable_tqdm=False,  # Keep main progress bar
        
        # Misc
        seed=42,
        remove_unused_columns=True,
        dataloader_pin_memory=True,  # Tăng tốc data loading
        dataloader_num_workers=4,    # Parallel data loading
        dataloader_prefetch_factor=2, # Prefetch batches
        push_to_hub=False
    )
    
    # ============== Initialize Trainer ==============
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer, rouge_metric),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )
    
    # ============== Train ==============
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION (LoRA)")
    print("="*60)
    print(f"Model: {model_args.model_name_or_path}")
    print(f"LoRA rank: {model_args.lora_r}")
    print(f"LoRA alpha: {model_args.lora_alpha}")
    print(f"Device: {device}")
    print(f"Mixed Precision (FP16): {use_fp16}")
    print(f"Batch Size (per device): {args.per_device_train_batch_size}")
    print(f"Gradient Accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective Batch Size: {args.per_device_train_batch_size * args.gradient_accumulation_steps * max(gpu_count, 1)}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Warmup Steps: {warmup_steps_to_use} (out of {total_steps} total)")
    print(f"Warmup Ratio: {warmup_steps_to_use/total_steps*100:.1f}%")
    print(f"Epochs: {args.num_train_epochs}")
    print("="*60 + "\n")
    
    print("\n[DEBUG] Starting training...")
    print(f"  Global step before: {trainer.state.global_step}")
    
    train_result = trainer.train()
    
    print(f"\n[DEBUG] Training completed")
    print(f"  Global step after: {trainer.state.global_step}")
    
    # Save model
    trainer.save_model()
    trainer.save_state()
    
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # ============== Evaluate on Test Set ==============
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60 + "\n")
    
    test_results = trainer.evaluate(
        eval_dataset=tokenized_datasets["test"],
        max_length=model_args.max_target_length,
        num_beams=4,
        metric_key_prefix="test"
    )
    
    trainer.log_metrics("test", test_results)
    trainer.save_metrics("test", test_results)
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Model saved to: {args.output_model_dir}")
    print("="*60 + "\n")
    
    print("Test Set Results:")
    for key, value in test_results.items():
        print(f"  {key}: {value:.2f}")


if __name__ == "__main__":
    main()
