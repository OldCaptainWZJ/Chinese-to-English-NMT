import argparse
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset

# 引入原有模块
from src.tokenizer import BilingualTokenizer
from src.dataloader import TranslationDataset
from src.evaluator import Evaluator

# 引入新模块
from src.models.transformer import build_transformer_model
from src.models.pretrained import build_pretrained_model

# 尝试导入 transformers
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

class HFTokenizerWrapper:
    """
    将 HuggingFace Tokenizer 包装成类似 BilingualTokenizer 的接口，
    以便复用现有的 Evaluator 代码。
    """
    def __init__(self, hf_tokenizer):
        self.tokenizer = hf_tokenizer
        # T5 token IDs
        self.pad_id = hf_tokenizer.pad_token_id
        self.eos_id = hf_tokenizer.eos_token_id
        # T5 通常没有单独的 SOS，decoder start token 单独处理，这里主要用于 evaluator 识别特殊字符
        self.sos_id = hf_tokenizer.pad_token_id 

    def detokenize_en(self, token_ids):
        """解码英文 Token IDs"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def tokenize_zh(self, text):
        """用于兼容 Dataset 的检查，虽然实际逻辑在 Dataset 中已处理"""
        return self.tokenizer(text).input_ids

def main():
    parser = argparse.ArgumentParser(description='Transformer模型评估')
    
    # 核心配置
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='模型检查点路径 (.pt 文件)')
    parser.add_argument('--model_type', type=str, default='scratch', 
                       choices=['scratch', 'pretrained'], help='模型类型')
    parser.add_argument('--test_file', type=str, default='dataset/test.jsonl',
                       help='测试文件路径')
    parser.add_argument('--batch_size', type=int, default=1, 
                       help='推理通常设为1，或者小批量')
    parser.add_argument('--output_file', type=str, default=None,
                       help='结果输出文件名')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='仅评估前N个样本（用于快速测试）')

    # 架构消融参数 (必须与训练时一致，用于重建 Scratch 模型)
    parser.add_argument('--norm_type', type=str, default='layernorm', choices=['layernorm', 'rmsnorm'])
    parser.add_argument('--pos_emb', type=str, default='absolute', choices=['absolute', 'relative'])
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--pretrained_model_name', type=str, default='t5-base')

    args = parser.parse_args()
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 路径配置
    BASE_DIR = Path(__file__).parent
    
    # 1. 加载分词器 & 适配器
    print("1. 加载分词器...")
    if args.model_type == 'pretrained':
        if AutoTokenizer is None:
            raise ImportError("请安装 transformers: pip install transformers")
        raw_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
        tokenizer = HFTokenizerWrapper(raw_tokenizer)
        src_vocab_size = raw_tokenizer.vocab_size
        tgt_vocab_size = raw_tokenizer.vocab_size
    else:
        # From Scratch
        TOKENIZER_DIR = BASE_DIR / "checkpoints" / "tokenizer"
        tokenizer = BilingualTokenizer()
        tokenizer.load_tokenizers(TOKENIZER_DIR)
        src_vocab_size = tokenizer.sp_zh.get_piece_size()
        tgt_vocab_size = tokenizer.sp_en.get_piece_size()

    # 2. 加载测试数据
    print("2. 加载测试数据...")
    # 注意：如果使用的是修改后的 TranslationDataset，
    # 传入 HFTokenizerWrapper 时，Dataset 内部可能无法通过 type check 判断它是 HF。
    # 所以如果用的是 T5，最好传入原始的 raw_tokenizer 给 Dataset，
    # 而传给 Evaluator 的是 wrapper。
    
    dataset_tokenizer = raw_tokenizer if args.model_type == 'pretrained' else tokenizer

    test_dataset = TranslationDataset(args.test_file, dataset_tokenizer)
    
    if args.max_samples is not None:
        indices = list(range(min(args.max_samples, len(test_dataset))))
        test_dataset = Subset(test_dataset, indices)
        print(f"  截断测试集: 使用前 {len(test_dataset)} 个样本")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: TranslationDataset.collate_fn(batch, pad_token_id=0)
    )
    
    # 3. 构建模型
    print(f"3. 构建模型 ({args.model_type})...")
    if args.model_type == 'pretrained':
        model = build_pretrained_model(args.pretrained_model_name, device)
    else:
        model = build_transformer_model(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            device=device,
            d_model=args.d_model,
            n_head=8, 
            num_layers=3,
            norm_type=args.norm_type,
            pos_emb_type=args.pos_emb
        )
    
    # 4. 加载权重
    print(f"4. 加载检查点: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 兼容不同的保存格式
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint # 有时直接保存了 state_dict

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 5. 运行评估
    print("5. 开始评估...")
    # 实例化 Evaluator
    evaluator = Evaluator(model, tokenizer, device)
    
    # 修正 Evaluator 的特殊 token ID（针对 T5）
    if args.model_type == 'pretrained':
        evaluator.pad_id = tokenizer.pad_id
        evaluator.eos_id = tokenizer.eos_id
        # T5 不需要 trim SOS，因为 output 只有生成的序列
        evaluator.sos_id = -1 
    
    results = evaluator.evaluate(test_loader, method="greedy", verbose=True)
    
    # 6. 保存结果
    if args.output_file:
        output_path = BASE_DIR / "test_output" / args.output_file
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model_type': args.model_type,
                'bleu4': float(results['bleu4']),
                'num_samples': results['num_samples'],
                'checkpoint': args.checkpoint,
                'args': vars(args)
            }, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_path}")

if __name__ == "__main__":
    main()