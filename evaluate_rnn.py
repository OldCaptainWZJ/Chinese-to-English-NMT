import argparse
import json
from pathlib import Path
import torch
from src.evaluator import Evaluator
from src.tokenizer import BilingualTokenizer
from src.models.rnn import build_model
from src.dataloader import TranslationDataset
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser(description='Run RNN Model Evaluation')
    
    # Model Configuration
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint (e.g., checkpoints/rnn_exp_concat_decay/best_model.pt)')
    parser.add_argument('--attention_type', type=str, default='concat', 
                       choices=['concat', 'general', 'dot'],
                       help='Must match the attention type used during training')
    
    # Data Configuration
    parser.add_argument('--test_file', type=str, default='dataset/test.jsonl',
                       help='Path to test file')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size (must be 1 for current decoding implementation)')
    
    # Experiment: Decoding Policy
    parser.add_argument('--decoding_method', type=str, default="greedy", 
                       choices=['greedy', 'beam'],
                       help='Decoding strategy to compare: greedy vs beam')
    
    # Output Configuration
    parser.add_argument('--output_file', type=str, default=None,
                       help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*50)
    print(f"Evaluating Model: {args.checkpoint}")
    print(f"Attention Type:   {args.attention_type}")
    print(f"Decoding Method:  {args.decoding_method}")
    print(f"Device:           {device}")
    print("="*50)
    
    # Path configuration
    BASE_DIR = Path(__file__).parent
    # Assuming tokenizer is stored in a common directory or relative to checkpoints
    TOKENIZER_DIR = BASE_DIR / "checkpoints" / "tokenizer"
    
    # 1. Load Tokenizer
    print("\n1. Loading Tokenizer...")
    tokenizer = BilingualTokenizer()
    if not TOKENIZER_DIR.exists():
        # Fallback if tokenizer is not in default path
        TOKENIZER_DIR = Path(args.checkpoint).parent.parent / "tokenizer"
    
    try:
        tokenizer.load_tokenizers(TOKENIZER_DIR)
    except Exception as e:
        print(f"Error loading tokenizer from {TOKENIZER_DIR}: {e}")
        return
    
    # 2. Load Test Data
    print("\n2. Loading Test Data...")
    test_dataset = TranslationDataset(args.test_file, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: TranslationDataset.collate_fn(batch, pad_token_id=0)
    )
    
    # 3. Build Model & Load Weights
    print("\n3. Building Model...")
    # Note: Hyperparameters (hidden_dim, etc.) must match training. 
    # If these vary, they should also be arguments or loaded from a config.
    model = build_model(
        src_vocab_size=tokenizer.sp_zh.get_piece_size(),
        tgt_vocab_size=tokenizer.sp_en.get_piece_size(),
        device=device,
        enc_emb_dim=256,
        dec_emb_dim=256,
        enc_hid_dim=512, # Consistent with trainer_rnn.py defaults
        dec_hid_dim=512, # Consistent with trainer_rnn.py defaults
        dropout=0.5,
        attention_type=args.attention_type # Crucial for Attention Experiment
    )
    
    print(f"Loading weights from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # 4. Evaluate
    # We pass the decoding method here to support the Decoding Policy experiment
    print(f"\n4. Running Evaluation using {args.decoding_method} decoding...")
    evaluator = Evaluator(model, tokenizer, device)
    
    # Evaluator.evaluate calls model.decode internally
    results = evaluator.evaluate(test_loader, method=args.decoding_method, verbose=True)
    
    # 5. Save Results
    if args.output_file is None:
        args.output_file = Path(args.checkpoint).parent.name  # default: name of the checkpoint sub-directory
    if args.output_file:
        output_path = BASE_DIR / "test_output" / args.output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result_data = {
            'checkpoint': args.checkpoint,
            'test_file': args.test_file,
            'attention_type': args.attention_type,
            'decoding_method': args.decoding_method,
            'bleu4': float(results['bleu4']),
            'num_samples': results['num_samples']
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()