import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from typing import List, Dict
import json


def load_model(model_path: str, base_model: str = "google/mt5-small", device: str = "cuda"):
    """Load model (base or LoRA)"""
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    if model_path == "base":
        # Base model without fine-tuning
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        print(f"✓ Loaded base model: {base_model}")
    else:
        # LoRA fine-tuned model
        base = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
        print(f"✓ Loaded LoRA model from: {model_path}")
    
    model.to(device)
    model.eval()
    
    return model, tokenizer


def generate_summary(
    text: str,
    model,
    tokenizer,
    max_source_length: int = 384,
    max_target_length: int = 128,
    num_beams: int = 4
):
    """Generate summary"""
    input_text = f"summarize: {text}"
    
    inputs = tokenizer(
        input_text,
        max_length=max_source_length,
        truncation=True,
        return_tensors="pt"
    )
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_target_length,
            num_beams=num_beams,
            early_stopping=True
        )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


def compare_models(
    text: str,
    models: Dict[str, tuple],
    max_source_length: int = 384,
    max_target_length: int = 128
):
    """Compare summaries from different models"""
    print("\n" + "="*80)
    print("INPUT TEXT:")
    print("="*80)
    print(text[:500] + "..." if len(text) > 500 else text)
    print()
    
    results = {}
    
    for model_name, (model, tokenizer) in models.items():
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name}")
        print("="*80)
        
        summary = generate_summary(
            text=text,
            model=model,
            tokenizer=tokenizer,
            max_source_length=max_source_length,
            max_target_length=max_target_length
        )
        
        print(summary)
        results[model_name] = summary
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80 + "\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare different models")
    parser.add_argument("--base_model", type=str, default="google/mt5-small")
    parser.add_argument("--lora_path", type=str, default="./models/mt5-lora-test")
    parser.add_argument("--cpo_path", type=str, default="./models/mt5-lora-cpo")
    parser.add_argument("--dpo_path", type=str, default="./models/mt5-lora-dpo")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--test_file", type=str, default=None,
                        help="JSON file with test examples")
    parser.add_argument("--max_source_length", type=int, default=384)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    
    # Setup device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"Device: {device}")
    print()
    
    # Load models
    models = {}
    
    print("Loading models...")
    
    # Base model
    base_model, base_tokenizer = load_model("base", args.base_model, device)
    models["Base mT5 (no fine-tuning)"] = (base_model, base_tokenizer)
    
    # LoRA SFT model (if exists)
    import os
    if os.path.exists(args.lora_path):
        lora_model, lora_tokenizer = load_model(args.lora_path, args.base_model, device)
        models["LoRA SFT"] = (lora_model, lora_tokenizer)
    else:
        print(f"⚠ LoRA model not found: {args.lora_path}")
    
    # LoRA + CPO model (if exists)
    if os.path.exists(args.cpo_path):
        cpo_model, cpo_tokenizer = load_model(args.cpo_path, args.base_model, device)
        models["LoRA + CPO"] = (cpo_model, cpo_tokenizer)
    else:
        print(f"⚠ CPO model not found: {args.cpo_path}")
    
    # LoRA + DPO model (if exists)
    if os.path.exists(args.dpo_path):
        dpo_model, dpo_tokenizer = load_model(args.dpo_path, args.base_model, device)
        models["LoRA + DPO"] = (dpo_model, dpo_tokenizer)
    else:
        print(f"⚠ DPO model not found: {args.dpo_path}")
    
    print()
    
    # Get input text
    if args.text:
        text = args.text
        compare_models(text, models, args.max_source_length, args.max_target_length)
    
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        compare_models(text, models, args.max_source_length, args.max_target_length)
    
    elif args.test_file:
        # Run on multiple test examples
        with open(args.test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        num_examples = min(5, len(test_data))
        print(f"\nTesting on {num_examples} examples from {args.test_file}\n")
        
        for i, item in enumerate(test_data[:num_examples]):
            print(f"\n{'#'*80}")
            print(f"EXAMPLE {i+1}/{num_examples}")
            print(f"{'#'*80}")
            
            text = item.get("content", "")
            ground_truth = item.get("summary", "")
            
            results = compare_models(text, models, args.max_source_length, args.max_target_length)
            
            print(f"\n{'='*80}")
            print("GROUND TRUTH SUMMARY:")
            print("="*80)
            print(ground_truth)
            print()
    
    else:
        # Default example
        example_text = """
        Việt Nam đang đẩy mạnh chuyển đổi số trong nhiều lĩnh vực, từ giáo dục, y tế 
        đến nông nghiệp và công nghiệp. Chính phủ đã ban hành nhiều chính sách hỗ trợ 
        doanh nghiệp và người dân tiếp cận công nghệ số. Các ứng dụng thanh toán điện tử, 
        thương mại điện tử đang phát triển mạnh mẽ. Tuy nhiên, vẫn còn nhiều thách thức 
        về hạ tầng, nguồn nhân lực và an ninh mạng cần được giải quyết.
        """
        
        compare_models(example_text.strip(), models, args.max_source_length, args.max_target_length)


if __name__ == "__main__":
    main()
