import argparse
import torch
from pathlib import Path
from src.tokenizer import BilingualTokenizer
from src.dataloader import create_data_loaders
from src.models.transformer import build_transformer_model
from src.models.pretrained import build_pretrained_model
from src.trainer_transformer import TransformerTrainer

# 尝试导入 transformers 库用于 mT5
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "dataset"
    
    # 1. 准备分词器和数据
    if args.model_type == 'scratch':
        TOKENIZER_DIR = BASE_DIR / "checkpoints" / "tokenizer"
        tokenizer = BilingualTokenizer()
        tokenizer.load_tokenizers(TOKENIZER_DIR)
        zh_vocab = tokenizer.sp_zh.get_piece_size()
        en_vocab = tokenizer.sp_en.get_piece_size()
    else:
        # Pretrained mT5
        print("加载 mT5 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
        zh_vocab = tokenizer.vocab_size
        en_vocab = tokenizer.vocab_size

    # 2. 数据加载
    train_loader, val_loader = create_data_loaders(
        train_path=DATA_DIR / "train_100k.jsonl",
        val_path=DATA_DIR / "valid.jsonl",
        tokenizer=tokenizer,
        batch_size=args.batch_size
    )

    # 3. 构建模型
    print(f"构建模型: {args.model_type}")
    if args.model_type == 'scratch':
        save_dir = f"checkpoints/transformer_{args.norm_type}_{args.pos_emb}"
        model = build_transformer_model(
            src_vocab_size=zh_vocab,
            tgt_vocab_size=en_vocab,
            device=device,
            norm_type=args.norm_type,       # Ablation: layernorm vs rmsnorm
            pos_emb_type=args.pos_emb,      # Ablation: absolute vs relative
            d_model=args.d_model,           # Sensitivity: model scale
            n_head=8, num_layers=3
        )
    else:
        dir_name = Path(args.pretrained_model_name).name
        save_dir = f"checkpoints/pretrained_{dir_name}"
        model = build_pretrained_model(args.pretrained_model_name, device)
    if args.directory is not None:
        save_dir = save_dir + f"_{args.directory}"

    # 4. 训练
    trainer = TransformerTrainer(model, device, checkpoint_dir=save_dir)
    if args.checkpoint is not None:
        trainer.load_checkpoint(args.checkpoint)
    trainer.train(train_loader, val_loader, num_epochs=args.num_epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='scratch', choices=['scratch', 'pretrained'])
    parser.add_argument('--directory', type=str, default=None)
    
    # Ablation Arguments
    parser.add_argument('--norm_type', type=str, default='layernorm', choices=['layernorm', 'rmsnorm'])
    parser.add_argument('--pos_emb', type=str, default='absolute', choices=['absolute', 'relative'])
    
    # Sensitivity Arguments
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0005)
    
    # Pretrained Arguments
    parser.add_argument('--pretrained_model_name', type=str, default='mt5-base')
    parser.add_argument('--checkpoint', type=str, default=None)
    
    parser.add_argument('--num_epochs', type=int, default=10)
    
    args = parser.parse_args()
    main(args)