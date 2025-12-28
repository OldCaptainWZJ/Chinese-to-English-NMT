from pathlib import Path
import torch
import os
import argparse

from src.tokenizer import BilingualTokenizer
from src.dataloader import create_data_loaders
from src.models.rnn import build_model
from src.trainer_rnn import RNNTrainer

def main(args):
    print("=" * 50)
    print("RNN Comparative Experiment Training")
    print(f"Attention Type: {args.attention_type}")
    print(f"TF Strategy: {args.tf_strategy} (Ratio: {args.tf_ratio})")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "dataset"
    MODEL_DIR = BASE_DIR / "checkpoints"
    # Create distinct directories for different experiments
    exp_name = f"{args.directory}_{args.attention_type}_{args.tf_strategy}"
    CHECKPOINT_DIR = MODEL_DIR / exp_name
    TOKENIZER_DIR = MODEL_DIR / "tokenizer"
    RAW_DIR = DATA_DIR
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("\n1. Loading Tokenizer...")
    tokenizer = BilingualTokenizer()
    tokenizer.load_tokenizers(TOKENIZER_DIR)
    
    print("\n2. Creating DataLoaders...")
    train_path = RAW_DIR / "train_100k.jsonl"
    val_path = RAW_DIR / "valid.jsonl"
    
    train_loader, val_loader = create_data_loaders(
        train_path=train_path,
        val_path=val_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size
    )
    
    print("\n3. Building Model...")
    model = build_model(
        src_vocab_size=tokenizer.sp_zh.get_piece_size(),
        tgt_vocab_size=tokenizer.sp_en.get_piece_size(),
        device=device,
        enc_emb_dim=256,
        dec_emb_dim=256,
        enc_hid_dim=512,
        dec_hid_dim=512,
        dropout=0.5,
        attention_type=args.attention_type # Pass attention type
    )
    
    print("\n4. Starting Training...")
    trainer = RNNTrainer(
        model=model,
        device=device,
        tokenizer=tokenizer, # Pass tokenizer for visual validation
        checkpoint_dir=CHECKPOINT_DIR,
        tf_strategy=args.tf_strategy,    # 'fixed' or 'decay'
        fixed_tf_ratio=args.tf_ratio     # Value for fixed strategy
    )

    if args.continue_checkpoint is not None:
        trainer.load_checkpoint(args.continue_checkpoint)
    
    trainer.train(train_loader, val_loader, num_epochs=args.num_epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RNN Comparative Experiments')
    parser.add_argument('--directory', type=str, default='experiment', help='Base directory name')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--continue_checkpoint', type=str, default=None)
    
    # Experiment 1: Attention Mechanism
    parser.add_argument('--attention_type', type=str, default='concat', 
                        choices=['concat', 'general', 'dot'],
                        help='Attention mechanism: concat (additive), general, or dot')

    # Experiment 2: Training Policy
    parser.add_argument('--tf_strategy', type=str, default='decay', 
                        choices=['decay', 'fixed'],
                        help='Teacher Forcing strategy')
    parser.add_argument('--tf_ratio', type=float, default=0.5, 
                        help='Fixed ratio for TF (if strategy is fixed). Use 1.0 for Teacher Forcing, 0.0 for Free Running')

    args = parser.parse_args()
    main(args)