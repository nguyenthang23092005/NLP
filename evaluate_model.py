import argparse
import json
import os
from tqdm import tqdm
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate


def batch_generate(texts, model, tokenizer, batch_size=8, max_source_length=512, 
                   max_target_length=128, num_beams=4, device="cuda"):
    """Generate summaries in batches"""
    summaries = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating"):
        batch_texts = texts[i:i+batch_size]
        batch_inputs = [f"summarize: {text}" for text in batch_texts]
        
        inputs = tokenizer(
            batch_inputs,
            max_length=max_source_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_target_length,
                num_beams=num_beams,
                early_stopping=True
            )
        
        batch_summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        summaries.extend(batch_summaries)
    
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate model on test set")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/splits")
    parser.add_argument("--split", type=str, default="test", choices=["test", "validation", "train"])
    parser.add_argument("--output_file", type=str, default="predictions.jsonl")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    
    # Device with GPU info
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*60)
    print("DEVICE INFORMATION")
    print("="*60)
    if device == "cuda" and torch.cuda.is_available():
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"⚠ Using CPU (slower evaluation)")
    print("="*60 + "\n")
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    
    # Check if it's a LoRA model (has adapter_config.json)
    import os
    is_lora = os.path.exists(os.path.join(args.model_path, "adapter_config.json"))
    
    if is_lora:
        # LoRA model: load tokenizer from base model, then load LoRA adapters
        print("Detected LoRA model. Loading tokenizer from base model...")
        base_model_name = "google/mt5-small"
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        print(f"Loading base model: {base_model_name}")
        from peft import PeftModel
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base_model, args.model_path)
        
        # Merge LoRA weights for faster inference
        print("Merging LoRA weights...")
        model = model.merge_and_unload()
    else:
        # Full fine-tuned model
        print("Loading full fine-tuned model...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    
    model.to(device)
    model.eval()
    print(f"✓ Model loaded successfully\n")
    
    # Load dataset
    print(f"Loading dataset from: {args.data_path}")
    
    # Support both JSON files and dataset directories
    if args.data_path.endswith('.json'):
        # Load from JSON file
        with open(args.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Split data (80% train, 10% val, 10% test)
        from datasets import Dataset
        dataset_full = Dataset.from_list([
            {"document": item["content"], "summary": item["summary"]}
            for item in data if item.get("content") and item.get("summary")
        ])
        
        # Create splits
        total = len(dataset_full)
        train_size = int(0.8 * total)
        val_size = int(0.1 * total)
        
        if args.split == "train":
            dataset = dataset_full.select(range(train_size))
        elif args.split == "validation":
            dataset = dataset_full.select(range(train_size, train_size + val_size))
        else:  # test
            dataset = dataset_full.select(range(train_size + val_size, total))
    else:
        # Load from dataset directory
        try:
            dataset = load_from_disk(args.data_path)[args.split]
        except (OSError, FileNotFoundError) as e:
            print(f"⚠ Error loading from disk: {e}")
            print("Attempting to load from processed_dataset.json instead...\n")
            
            from datasets import load_dataset as hf_load_dataset
            full_dataset = hf_load_dataset('json', data_files='data/processed_dataset.json', split='train')
            
            # Rename columns to match expected format
            full_dataset = full_dataset.rename_column("content", "document")
            # summary column already exists
            
            # Create splits
            splits = full_dataset.train_test_split(test_size=0.2, seed=42)
            test_valid = splits['test'].train_test_split(test_size=0.5, seed=42)
            
            split_map = {
                'train': splits['train'],
                'test': test_valid['train'],
                'validation': test_valid['test']
            }
            
            dataset = split_map[args.split]
            print(f"✓ Loaded {len(dataset)} samples from {args.split} split\n")
    
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    print(f"Evaluating on {len(dataset)} samples from {args.split} split")
    
    # Extract texts and references
    documents = dataset["document"]
    references = dataset["summary"]
    
    # Generate predictions
    predictions = batch_generate(
        documents,
        model,
        tokenizer,
        batch_size=args.batch_size,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        num_beams=args.num_beams,
        device=device
    )
    
    # Save predictions
    print(f"\nSaving predictions to: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for doc, ref, pred in zip(documents, references, predictions):
            f.write(json.dumps({
                "document": doc,
                "reference": ref,
                "prediction": pred
            }, ensure_ascii=False) + "\n")
    
    # Compute multiple metrics
    print("\nComputing evaluation metrics...")
    
    # Format for metrics (add newlines)
    formatted_preds = ["\n".join(p.strip().split()) for p in predictions]
    formatted_refs = ["\n".join(r.strip().split()) for r in references]
    
    results = {}
    
    # 1. ROUGE scores
    print("  - Computing ROUGE...")
    rouge = evaluate.load("rouge")
    rouge_results = rouge.compute(
        predictions=formatted_preds,
        references=formatted_refs,
        use_stemmer=False
    )
    results.update(rouge_results)
    
    # 2. BLEU score
    print("  - Computing BLEU...")
    try:
        bleu = evaluate.load("bleu")
        bleu_results = bleu.compute(
            predictions=[p.split() for p in predictions],
            references=[[r.split()] for r in references]
        )
        results['bleu'] = bleu_results['bleu']
        results['bleu_precisions'] = bleu_results['precisions']
    except Exception as e:
        print(f"    Warning: BLEU computation failed: {e}")
    
    # 3. METEOR score
    print("  - Computing METEOR...")
    try:
        meteor = evaluate.load("meteor")
        meteor_results = meteor.compute(
            predictions=predictions,
            references=references
        )
        results['meteor'] = meteor_results['meteor']
    except Exception as e:
        print(f"    Warning: METEOR computation failed: {e}")
    
    # 4. BERTScore (semantic similarity)
    print("  - Computing BERTScore...")
    try:
        bertscore = evaluate.load("bertscore")
        bert_results = bertscore.compute(
            predictions=predictions,
            references=references,
            lang="vi",  # Vietnamese
            model_type="xlm-roberta-base",
            device=device
        )
        results['bertscore_precision'] = sum(bert_results['precision']) / len(bert_results['precision'])
        results['bertscore_recall'] = sum(bert_results['recall']) / len(bert_results['recall'])
        results['bertscore_f1'] = sum(bert_results['f1']) / len(bert_results['f1'])
    except Exception as e:
        print(f"    Warning: BERTScore computation failed: {e}")
    
    # 5. Custom metrics
    print("  - Computing custom metrics...")
    
    # Length statistics
    pred_lengths = [len(p.split()) for p in predictions]
    ref_lengths = [len(r.split()) for r in references]
    results['avg_pred_length'] = sum(pred_lengths) / len(pred_lengths)
    results['avg_ref_length'] = sum(ref_lengths) / len(ref_lengths)
    results['length_ratio'] = results['avg_pred_length'] / results['avg_ref_length']
    
    # Compression ratio
    doc_lengths = [len(d.split()) for d in documents]
    results['avg_doc_length'] = sum(doc_lengths) / len(doc_lengths)
    results['compression_ratio'] = results['avg_pred_length'] / results['avg_doc_length']
    
    # Vocabulary overlap
    def vocab_overlap(preds, refs):
        overlaps = []
        for p, r in zip(preds, refs):
            p_vocab = set(p.split())
            r_vocab = set(r.split())
            if len(r_vocab) > 0:
                overlaps.append(len(p_vocab & r_vocab) / len(r_vocab))
        return sum(overlaps) / len(overlaps) if overlaps else 0
    
    results['vocab_overlap'] = vocab_overlap(predictions, references)
    
    # Novel n-gram ratio (abstractiveness)
    def novel_ngrams(preds, docs, n=2):
        novel_ratios = []
        for p, d in zip(preds, docs):
            p_ngrams = set(zip(*[p.split()[i:] for i in range(n)]))
            d_ngrams = set(zip(*[d.split()[i:] for i in range(n)]))
            if len(p_ngrams) > 0:
                novel = len(p_ngrams - d_ngrams) / len(p_ngrams)
                novel_ratios.append(novel)
        return sum(novel_ratios) / len(novel_ratios) if novel_ratios else 0
    
    results['novel_bigrams'] = novel_ngrams(predictions, documents, n=2)
    results['novel_trigrams'] = novel_ngrams(predictions, documents, n=3)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION METRICS:")
    print("="*60)
    
    print("\n📊 ROUGE Scores:")
    for key in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']:
        if key in results:
            print(f"  {key:15s}: {results[key]*100:.2f}")
    
    print("\n📊 Other Automatic Metrics:")
    if 'bleu' in results:
        print(f"  {'BLEU':15s}: {results['bleu']*100:.2f}")
    if 'meteor' in results:
        print(f"  {'METEOR':15s}: {results['meteor']*100:.2f}")
    
    if 'bertscore_f1' in results:
        print("\n📊 BERTScore (Semantic Similarity):")
        print(f"  {'Precision':15s}: {results['bertscore_precision']*100:.2f}")
        print(f"  {'Recall':15s}: {results['bertscore_recall']*100:.2f}")
        print(f"  {'F1':15s}: {results['bertscore_f1']*100:.2f}")
    
    print("\n📊 Length & Compression:")
    print(f"  {'Avg Doc Length':20s}: {results['avg_doc_length']:.1f} words")
    print(f"  {'Avg Pred Length':20s}: {results['avg_pred_length']:.1f} words")
    print(f"  {'Avg Ref Length':20s}: {results['avg_ref_length']:.1f} words")
    print(f"  {'Length Ratio':20s}: {results['length_ratio']:.2f} (pred/ref)")
    print(f"  {'Compression Ratio':20s}: {results['compression_ratio']:.2%} (pred/doc)")
    
    print("\n📊 Abstractiveness:")
    print(f"  {'Vocab Overlap':20s}: {results['vocab_overlap']*100:.1f}%")
    print(f"  {'Novel Bigrams':20s}: {results['novel_bigrams']*100:.1f}%")
    print(f"  {'Novel Trigrams':20s}: {results['novel_trigrams']*100:.1f}%")
    
    print("="*60)
    
    # Save metrics
    metrics_file = args.output_file.replace(".jsonl", "_metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump({k: v*100 for k, v in results.items()}, f, indent=2)
    print(f"\nMetrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
