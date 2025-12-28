import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from src.tokenizer import BilingualTokenizer
from src.models.rnn import build_model
from src.dataloader import TranslationDataset

class Evaluator:
    """模型评估器"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # 设置模型为评估模式
        self.model.eval()
        
        # 特殊标记ID
        self.pad_id = 0
        self.sos_id = 2
        self.eos_id = 3
        
    def decode_batch(self, batch, method="greedy") -> Tuple[List[str], List[str]]:
        """
        解码一个批次，生成翻译结果
        
        Args:
            batch: 数据批次
            
        Returns:
            hypotheses: 生成的翻译文本列表
            references: 参考翻译文本列表
        """
        with torch.no_grad():
            src_tokens = batch['src'].to(self.device)
            tgt_tokens = batch['tgt'].to(self.device)
            src_lengths = batch['src_mask'].sum(dim=1).to(self.device)
            tgt_lengths = batch['tgt_mask'].sum(dim=1).to(self.device)

            # batch_size = 1
            predictions = self.model.decode(src_tokens, src_lengths, tgt_tokens, method)
            predictions = torch.tensor(predictions).unsqueeze(0)

            # 转换为文本
            hypotheses = []
            references = []
            
            for i in range(predictions.size(0)):
                # 获取预测的token IDs，忽略padding和特殊标记
                pred_ids = predictions[i].cpu().tolist()
                ref_ids = tgt_tokens[i].cpu().tolist()
                
                # 去除padding和特殊标记
                pred_ids = self._trim_special_tokens(pred_ids)
                ref_ids = self._trim_special_tokens(ref_ids)
                
                # 解码为文本
                pred_text = self.tokenizer.detokenize_en(pred_ids)
                ref_text = self.tokenizer.detokenize_en(ref_ids)
                
                # 清理文本（去除多余空格）
                pred_text = self._clean_text(pred_text)
                ref_text = self._clean_text(ref_text)
                
                hypotheses.append(pred_text)
                references.append([ref_text])  # BLEU需要将参考文本包装在列表中
            
            return hypotheses, references
    
    def _trim_special_tokens(self, token_ids: List[int]) -> List[int]:
        """去除特殊标记（PAD, SOS, EOS）"""
        result = []
        for token_id in token_ids:
            if token_id == self.pad_id or token_id == self.sos_id:
                continue
            if token_id == self.eos_id:
                break
            result.append(token_id)
        return result
    
    def _clean_text(self, text: str) -> str:
        """清理文本，去除多余空格"""
        return ' '.join(text.split())
    
    def compute_bleu(self, hypotheses: List[str], references: List[List[str]], method='corpus') -> float:
        """
        计算BLEU-4分数
        
        Args:
            hypotheses: 生成的翻译文本列表
            references: 参考翻译文本列表
            method: 'corpus' (语料库级，标准方法) 或 'sentence' (句级平均)
            
        Returns:
            BLEU-4分数
        """
        from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
        
        if method == 'corpus':
            # 语料库级BLEU（标准方法）
            hypotheses_tokens = [hyp.split() for hyp in hypotheses]
            references_tokens = [[ref[0].split()] for ref in references]
            
            smoothie = SmoothingFunction().method4
            return corpus_bleu(
                references_tokens,
                hypotheses_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothie
            )
        
        else:
            # 句级BLEU平均
            bleu_scores = []
            smoothie = SmoothingFunction().method4
            
            for hyp, refs in zip(hypotheses, references):
                hyp_tokens = hyp.split()
                ref_tokens = [ref.split() for ref in refs]
                
                bleu = sentence_bleu(
                    ref_tokens, 
                    hyp_tokens, 
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=smoothie
                )
                bleu_scores.append(bleu)
            
            return np.mean(bleu_scores)
    
    def evaluate(self, test_loader, method="greedy", verbose=True) -> dict:
        """
        在测试集上评估模型
        
        Args:
            test_loader: 测试数据加载器
            verbose: 是否显示详细信息
            
        Returns:
            评估结果字典
        """
        print("开始评估模型...")
        
        all_hypotheses = []
        all_references = []
        
        # 遍历测试集
        progress_bar = tqdm(test_loader, desc="评估进度") if verbose else test_loader
        for batch_idx, batch in enumerate(progress_bar):
            hypotheses, references = self.decode_batch(batch, method)
            all_hypotheses.extend(hypotheses)
            all_references.extend(references)
            
            # 每100个batch显示一个示例
            if verbose and batch_idx % 100 == 0 and batch_idx > 0:
                print(f"\n示例 {batch_idx}:")
                print(f"  原文: {batch['src_text'][0]}")
                print(f"  参考: {references[0][0]}")
                print(f"  生成: {hypotheses[0]}")
        
        # 计算BLEU-4分数
        bleu_score = self.compute_bleu(all_hypotheses, all_references)
        
        # 打印一些统计信息
        if verbose:
            print(f"\n评估完成!")
            print(f"测试样本总数: {len(all_hypotheses)}")
            print(f"BLEU-4分数: {bleu_score:.4f}")
            
            # 打印几个随机示例
            print(f"\n随机示例:")
            indices = np.random.choice(len(all_hypotheses), min(3, len(all_hypotheses)), replace=False)
            for idx in indices:
                print(f"示例 {idx}:")
                print(f"  参考: {all_references[idx][0]}")
                print(f"  生成: {all_hypotheses[idx]}")
                print()
        
        return {
            'bleu4': bleu_score,
            'num_samples': len(all_hypotheses),
            'hypotheses': all_hypotheses,
            'references': all_references
        }


def main():
    """主评估函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RNN模型评估')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--max_samples', type=int, default=None, help='最大评估样本数（测试用）')
    args = parser.parse_args()
    
    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 路径配置
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "dataset"
    MODEL_DIR = BASE_DIR / "checkpoints"
    TOKENIZER_DIR = MODEL_DIR / "tokenizer"
    TEST_PATH = DATA_DIR / "test.jsonl"
    
    # 1. 加载分词器
    print("加载分词器...")
    tokenizer = BilingualTokenizer()
    tokenizer.load_tokenizers(TOKENIZER_DIR)
    
    # 2. 创建测试数据集
    print("创建测试数据集...")
    test_dataset = TranslationDataset(TEST_PATH, tokenizer)
    
    # 如果指定了最大样本数，则截断数据集（用于快速测试）
    if args.max_samples is not None:
        from torch.utils.data import Subset
        indices = list(range(min(args.max_samples, len(test_dataset))))
        test_dataset = Subset(test_dataset, indices)
        print(f"使用前 {len(test_dataset)} 个样本进行测试")
    
    # 创建数据加载器
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: TranslationDataset.collate_fn(batch, pad_token_id=0)
    )
    
    # 3. 获取词表大小并创建模型
    print("创建模型...")
    model = build_model(
        src_vocab_size=tokenizer.sp_zh.get_piece_size(),
        tgt_vocab_size=tokenizer.sp_en.get_piece_size(),
        device=device,
        enc_emb_dim=256,
        dec_emb_dim=256,
        enc_hid_dim=512,
        dec_hid_dim=512,
        dropout=0.5,
        teacher_forcing_ratio=0.0  # 评估时不使用teacher forcing
    )
    
    # 4. 加载模型权重
    print(f"加载检查点: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # 5. 创建评估器并进行评估
    evaluator = Evaluator(model, tokenizer, device)
    results = evaluator.evaluate(test_loader, verbose=True)
    
    # 6. 保存评估结果
    output_file = MODEL_DIR / "evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'bleu4': float(results['bleu4']),
            'num_samples': results['num_samples'],
            'timestamp': str(Path(__file__).stat().st_mtime)
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n评估结果已保存到: {output_file}")
    print(f"最终BLEU-4分数: {results['bleu4']:.4f}")


if __name__ == "__main__":
    # 安装必要的库
    import sys
    import subprocess
    
    # 检查nltk是否已安装
    try:
        import nltk
    except ImportError:
        print("正在安装nltk...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
        import nltk
    
    # 下载必要的nltk数据
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("下载nltk punkt数据...")
        nltk.download('punkt', quiet=True)
    
    main()