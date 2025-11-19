import os
import json
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate


# ============== CPO Configuration ==============
@dataclass
class CPOArguments:
    """Arguments for CPO training"""
    beta: float = field(default=0.1, metadata={"help": "CPO beta parameter (loss weight)"})
    label_smoothing: float = field(default=0.0, metadata={"help": "Label smoothing factor"})
    max_length: int = field(default=128, metadata={"help": "Max generation length"})
    temperature: float = field(default=1.0, metadata={"help": "Generation temperature"})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="models/mt5-small")
    pretrained_lora_path: Optional[str] = field(default=None, metadata={"help": "Path to pretrained LoRA model"})
    max_source_length: int = field(default=384)
    max_target_length: int = field(default=128)
    
    # LoRA config
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)


@dataclass
class DataArguments:
    input_file: str = field(default="data/processed_dataset.json")
    preference_file: Optional[str] = field(default=None, metadata={"help": "JSON file with preference pairs"})
    output_dir: str = field(default="data/cpo_splits")
    max_samples: Optional[int] = field(default=None)
    generate_synthetic_pairs: bool = field(default=True)
    seed: int = field(default=42)


# ============== Generate Synthetic Preference Pairs ==============
def create_negative_summary(summary: str, method: str = "truncate") -> str:
    """Create a negative summary from a positive one"""
    words = summary.split()
    
    if method == "truncate":
        # Cắt ngắn quá mức (mất thông tin)
        return " ".join(words[:len(words)//3])
    
    elif method == "repeat":
        # Lặp lại từ (không tự nhiên)
        if len(words) > 5:
            return " ".join(words[:5] + words[2:4] + words[2:4] + words[5:])
        return summary
    
    elif method == "shuffle":
        # Xáo trộn thứ tự (mất mạch lạc)
        import random
        random.seed(42)
        shuffled = words.copy()
        random.shuffle(shuffled)
        return " ".join(shuffled[:len(words)])
    
    elif method == "verbose":
        # Thêm từ dài dòng
        return summary + " " + " ".join(words[:3])
    
    return summary


def generate_preference_pairs(dataset: Dataset, num_negatives: int = 2) -> Dataset:
    """Generate preference pairs with positive and negative summaries"""
    preference_data = []
    
    for item in dataset:
        document = item["document"]
        positive_summary = item["summary"]
        
        # Tạo nhiều negative summaries với các phương pháp khác nhau
        methods = ["truncate", "repeat", "shuffle", "verbose"]
        negative_summaries = []
        
        for i in range(min(num_negatives, len(methods))):
            neg_summary = create_negative_summary(positive_summary, method=methods[i])
            negative_summaries.append(neg_summary)
        
        preference_data.append({
            "document": document,
            "chosen": positive_summary,  # Tóm tắt tốt (ground truth)
            "rejected": negative_summaries  # Danh sách tóm tắt xấu
        })
    
    return Dataset.from_list(preference_data)


# ============== CPO Loss Function ==============
class CPOTrainer(Seq2SeqTrainer):
    """Custom Trainer with CPO loss"""
    
    def __init__(self, *args, cpo_beta=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.cpo_beta = cpo_beta
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to remove rejected_input_ids during evaluation"""
        # Remove rejected_input_ids before evaluation/generation
        inputs.pop("rejected_input_ids", None)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute CPO loss:
        L = -log(sigmoid(beta * (log p(y_chosen|x) - log p(y_rejected|x))))
        """
        # Extract rejected_input_ids if present (model doesn't accept this argument)
        rejected_input_ids = inputs.pop("rejected_input_ids", None)
        
        # CPO contrastive loss (nếu có rejected summaries)
        if rejected_input_ids is not None:
            # Forward pass for chosen
            chosen_outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"]
            )
            chosen_loss = chosen_outputs.loss
            
            # Forward pass for rejected
            rejected_outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=rejected_input_ids
            )
            rejected_loss = rejected_outputs.loss
            
            # CPO loss: prefer chosen over rejected
            # Lower loss is better, so we want chosen_loss < rejected_loss
            # logits_diff = rejected_loss - chosen_loss (should be positive)
            logits_diff = rejected_loss - chosen_loss
            
            # Clamp to avoid numerical issues
            logits_diff = torch.clamp(logits_diff, min=-10, max=10)
            cpo_loss = -torch.nn.functional.logsigmoid(self.cpo_beta * logits_diff)
            
            # Combine: supervised loss on chosen + CPO preference loss
            loss = chosen_loss + cpo_loss
            
            # Ensure loss is valid (handle multi-GPU case)
            # Loss might be a tensor with multiple values (one per GPU)
            if loss.dim() > 0:
                loss = loss.mean()  # Average across GPUs
            
            loss_item = loss.item()
            if np.isnan(loss_item) or np.isinf(loss_item):
                print(f"Warning: Invalid loss detected. chosen_loss={chosen_loss.mean().item():.4f}, rejected_loss={rejected_loss.mean().item():.4f}, cpo_loss={cpo_loss.mean().item():.4f}")
                loss = chosen_loss.mean() if chosen_loss.dim() > 0 else chosen_loss  # Fallback to just supervised loss
            
            outputs = chosen_outputs
        else:
            # Standard supervised loss only
            outputs = model(**inputs)
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss


# ============== Data Loading & Preprocessing ==============
def load_and_prepare_cpo_data(data_args: DataArguments):
    """Load and prepare data for CPO training"""
    print(f"\n{'='*60}")
    
    # Check if we should load from existing splits (from LoRA stage)
    if os.path.exists("data/splits/dataset_dict.json"):
        print(f"Loading dataset from existing LoRA splits: data/splits")
        from datasets import load_from_disk
        dataset_dict = load_from_disk("data/splits")
        
        # Use the same train/val/test split as LoRA
        train_dataset = dataset_dict["train"]
        val_dataset = dataset_dict["validation"]
        test_dataset = dataset_dict["test"]
        
        # Apply max_samples if specified
        if data_args.max_samples:
            train_dataset = train_dataset.select(range(min(data_args.max_samples, len(train_dataset))))
            print(f"Limited to {len(train_dataset)} training samples")
        
        print(f"Using LoRA splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    else:
        # Fallback to loading from JSON
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
        
        # Split dataset
        train_size = int(len(dataset) * 0.8)
        val_size = int(len(dataset) * 0.1)
        
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, train_size + val_size))
        test_dataset = dataset.select(range(train_size + val_size, len(dataset)))
        
        print(f"Split sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Generate preference pairs for CPO
    if data_args.generate_synthetic_pairs:
        print("\nGenerating synthetic preference pairs...")
        train_dataset = generate_preference_pairs(train_dataset, num_negatives=2)
        val_dataset = generate_preference_pairs(val_dataset, num_negatives=2)
        test_dataset = generate_preference_pairs(test_dataset, num_negatives=2)
        print(f"✓ Created preference pairs with chosen/rejected summaries")
    
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    
    os.makedirs(data_args.output_dir, exist_ok=True)
    dataset_dict.save_to_disk(data_args.output_dir)
    print(f"Saved CPO splits to: {data_args.output_dir}")
    print(f"{'='*60}\n")
    
    return dataset_dict


def preprocess_cpo_function(examples, tokenizer, model_args, prefix="summarize: "):
    """Preprocess function for CPO with chosen/rejected pairs"""
    inputs = [prefix + doc for doc in examples["document"]]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=model_args.max_source_length,
        truncation=True,
        padding=False,
        return_attention_mask=True
    )
    
    # Check if we have CPO preference pairs or just regular summaries
    if "chosen" in examples:
        # CPO mode: tokenize chosen summaries (positive examples)
        with tokenizer.as_target_tokenizer():
            chosen_labels = tokenizer(
                examples["chosen"],
                max_length=model_args.max_target_length,
                truncation=True,
                padding=False
            )
        
        model_inputs["labels"] = chosen_labels["input_ids"]
        
        # Tokenize rejected summaries (negative examples)
        if "rejected" in examples and examples["rejected"]:
            # Handle rejected as list or single value
            rejected_texts = [rej[0] if isinstance(rej, list) else rej for rej in examples["rejected"]]
            
            with tokenizer.as_target_tokenizer():
                rejected_labels = tokenizer(
                    rejected_texts,
                    max_length=model_args.max_target_length,
                    truncation=True,
                    padding=False
                )
            
            model_inputs["rejected_input_ids"] = rejected_labels["input_ids"]
    else:
        # Standard mode: just use summary as labels
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["summary"],
                max_length=model_args.max_target_length,
                truncation=True,
                padding=False
            )
        
        model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs


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


class CPODataCollator:
    """Custom data collator for CPO that handles rejected_input_ids properly"""
    
    def __init__(self, tokenizer, model=None, padding=True, max_length=None, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.model = model
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features):
        import torch
        
        # Check if we have rejected_input_ids
        has_rejected = "rejected_input_ids" in features[0]
        
        # Extract rejected_input_ids if present
        rejected_ids = None
        if has_rejected:
            rejected_ids = [f.pop("rejected_input_ids") for f in features]
        
        # Manual padding for each field
        batch = {}
        
        # Pad input_ids
        input_ids = [f["input_ids"] for f in features]
        input_ids_padded = self._pad_sequences(input_ids, self.tokenizer.pad_token_id)
        batch["input_ids"] = input_ids_padded
        
        # Pad attention_mask
        if "attention_mask" in features[0]:
            attention_masks = [f["attention_mask"] for f in features]
            attention_mask_padded = self._pad_sequences(attention_masks, 0)
            batch["attention_mask"] = attention_mask_padded
        
        # Pad labels
        if "labels" in features[0]:
            labels = [f["labels"] for f in features]
            labels_padded = self._pad_sequences(labels, self.tokenizer.pad_token_id)
            # Replace pad token with -100
            labels_padded = labels_padded.masked_fill(
                labels_padded == self.tokenizer.pad_token_id, -100
            )
            batch["labels"] = labels_padded
        
        # Pad rejected_input_ids if present
        if has_rejected and rejected_ids:
            rejected_padded = self._pad_sequences(rejected_ids, self.tokenizer.pad_token_id)
            batch["rejected_input_ids"] = rejected_padded
        
        return batch
    
    def _pad_sequences(self, sequences, pad_value):
        """Pad sequences to the same length"""
        import torch
        
        max_len = max(len(seq) for seq in sequences)
        
        # Apply pad_to_multiple_of if specified
        if self.pad_to_multiple_of is not None:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        padded = []
        for seq in sequences:
            padding_length = max_len - len(seq)
            padded_seq = seq + [pad_value] * padding_length
            padded.append(padded_seq)
        
        return torch.tensor(padded, dtype=torch.long)


# ============== Main Training Function ==============
def main():
    parser = argparse.ArgumentParser(description="Fine-tune mT5 with LoRA + CPO")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="models/mt5-small")
    parser.add_argument("--pretrained_lora_path", type=str, default="./models/mt5-lora-full/checkpoint-5500",
                        help="Path to pretrained LoRA checkpoint (optional)")
    parser.add_argument("--max_source_length", type=int, default=384)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    
    # CPO arguments
    parser.add_argument("--cpo_beta", type=float, default=0.1,
                        help="CPO beta parameter (higher = stronger preference)")
    parser.add_argument("--generate_synthetic_pairs", action="store_true", default=True)
    
    # Data arguments
    parser.add_argument("--input_file", type=str, default="data/processed_dataset.json")
    parser.add_argument("--output_dir", type=str, default="data/cpo_splits")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--load_splits", action="store_true")
    
    # Training arguments
    parser.add_argument("--output_model_dir", type=str, default="./models/mt5-lora-cpo")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    
    args = parser.parse_args()
    
    # ============== Device Info ==============
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
    
    # Create arguments
    model_args = ModelArguments(
        model_name_or_path=args.model_name,
        pretrained_lora_path=args.pretrained_lora_path,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    data_args = DataArguments(
        input_file=args.input_file,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        generate_synthetic_pairs=args.generate_synthetic_pairs
    )
    
    # ============== Load Dataset ==============
    if args.load_splits and os.path.exists(args.output_dir):
        print(f"Loading existing CPO splits from: {args.output_dir}")
        dataset_dict = DatasetDict.load_from_disk(args.output_dir)
    else:
        dataset_dict = load_and_prepare_cpo_data(data_args)
    
    # ============== Load Model & Tokenizer ==============
    print(f"Loading tokenizer: {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    
    if args.pretrained_lora_path:
        print(f"Loading pretrained LoRA model from: {args.pretrained_lora_path}")
        from peft import PeftModel
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)
        model = PeftModel.from_pretrained(base_model, args.pretrained_lora_path)
        print("✓ Loaded pretrained LoRA adapters")
        
        # Ensure LoRA adapters are trainable (not frozen)
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
        
        # Ensure model is in training mode
        model.train()
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable_params:,} || All params: {all_params:,} || Trainable%: {100 * trainable_params / all_params:.2f}%")
    else:
        print(f"Loading base model: {model_args.model_name_or_path}")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)
        
        # Setup LoRA
        print("\n" + "="*60)
        print("LORA CONFIGURATION")
        print("="*60)
        
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=["q", "v"],
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    model = model.to(device)
    print(f"Model moved to: {device}")
    
    # CRITICAL: Disable gradient checkpointing - causes NaN with LoRA + CPO
    if args.gradient_checkpointing:
        print("⚠ Warning: Gradient checkpointing disabled for stability (causes NaN with LoRA + CPO)")
        print("⚠ If you run out of memory, reduce batch_size or use smaller model")
    
    print("="*60 + "\n")
    
    # ============== Preprocess Datasets ==============
    print("Tokenizing datasets with CPO pairs...")
    
    # Check what columns exist and determine which to remove
    existing_columns = dataset_dict["train"].column_names
    print(f"Dataset columns: {existing_columns}")
    
    # Remove all original columns except those needed by the model
    # The preprocess function will create the required model input columns
    tokenized_datasets = dataset_dict.map(
        lambda x: preprocess_cpo_function(x, tokenizer, model_args),
        batched=True,
        remove_columns=existing_columns,
        desc="Tokenizing"
    )
    
    # Giảm validation set xuống 10k samples để eval nhanh hơn
    if len(tokenized_datasets["validation"]) > 10000:
        print(f"\n⚡ Reducing validation set from {len(tokenized_datasets['validation'])} to 10,000 samples for faster eval")
        tokenized_datasets["validation"] = tokenized_datasets["validation"].select(range(10000))
    
    # ============== Data Collator ==============
    data_collator = CPODataCollator(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8 if args.fp16 else None
    )
    
    # ============== Load Metrics ==============
    rouge_metric = evaluate.load("rouge")
    
    # ============== Training Arguments ==============
    num_train_samples = len(tokenized_datasets["train"])
    steps_per_epoch = num_train_samples // (args.per_device_train_batch_size * args.gradient_accumulation_steps * max(gpu_count, 1))
    total_steps = steps_per_epoch * args.num_train_epochs
    
    # Calculate warmup steps: 10% of total steps maximum
    calculated_warmup_steps = max(int(0.1 * total_steps), 1)
    warmup_steps_to_use = min(args.warmup_steps, calculated_warmup_steps)
    
    # Critical: Ensure warmup < total steps
    if warmup_steps_to_use >= total_steps:
        warmup_steps_to_use = max(1, int(0.1 * total_steps))
        print(f"⚠ Warning: Warmup steps adjusted to {warmup_steps_to_use} (10% of {total_steps} total steps)")
    
    print("\n" + "="*60)
    print("CPO TRAINING CONFIGURATION")
    print("="*60)
    print(f"Training samples: {num_train_samples}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps_to_use}")
    print(f"CPO Beta: {args.cpo_beta}")
    print("="*60 + "\n")
    
    # CRITICAL: Disable FP16 for stability (causes NaN with gradient checkpointing + LoRA)
    use_fp16 = False
    if args.fp16:
        print("⚠ Warning: FP16 disabled for stability (causes NaN with gradient checkpointing + LoRA)")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_model_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps_to_use,
        max_grad_norm=0.5,  # Reduce from 1.0 to 0.5 for more stability
        
        # Evaluation & Saving
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        greater_is_better=True,
        
        # Generation
        predict_with_generate=True,
        generation_max_length=model_args.max_target_length,
        generation_num_beams=4,
        
        # Optimization
        fp16=use_fp16,
        optim="adamw_torch",
        
        # Logging
        logging_dir=f"{args.output_model_dir}/logs",
        logging_steps=100,
        logging_first_step=True,
        report_to=["tensorboard"],
        
        seed=42,
        remove_unused_columns=False,  # Giữ rejected columns cho CPO
        dataloader_pin_memory=False,
        push_to_hub=False
    )
    
    # ============== Initialize CPO Trainer ==============
    trainer = CPOTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer, rouge_metric),
        cpo_beta=args.cpo_beta
    )
    
    # ============== Train ==============
    print("\n" + "="*60)
    print("STARTING CPO TRAINING")
    print("="*60 + "\n")
    
    train_result = trainer.train()
    
    # Save model
    trainer.save_model()
    trainer.save_state()
    
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # ============== Evaluate ==============
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
    print("CPO TRAINING COMPLETED!")
    print(f"Model saved to: {args.output_model_dir}")
    print("="*60 + "\n")
    
    print("Test Set Results:")
    for key, value in test_results.items():
        print(f"  {key}: {value:.2f}")


if __name__ == "__main__":
    main()
