import json
from typing import List, Dict, Tuple
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
from src.tokenizer import BilingualTokenizer

class TranslationDataset(Dataset):
    """机器翻译数据集"""
    
    def __init__(self, data_path: str, tokenizer):
        self.tokenizer = tokenizer
        self.samples = []
        
        # 判断是否是 HF tokenizer
        self.is_hf = isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerBase)) if 'PreTrainedTokenizer' in globals() else False
        # 如果是 BilingualTokenizer，它没有 __class__ 检查，直接看方法名
        if hasattr(tokenizer, "tokenize_zh"):
            self.is_hf = False
        else:
            self.is_hf = True

        print("is_hf: ", self.is_hf)

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    zh_text = data['zh']
                    en_text = data['en']
                    
                    if not self.is_hf:
                        # 原有的逻辑
                        zh_ids = tokenizer.tokenize_zh(zh_text)
                        en_ids = tokenizer.tokenize_en(en_text)
                    else:
                        # 针对 T5/Pretrained: 预处理为 input_ids 和 labels
                        # T5 需要前缀
                        inputs = tokenizer("translate Chinese to English: " + zh_text, truncation=True, max_length=256)
                        targets = tokenizer(en_text, truncation=True, max_length=256)
                        zh_ids = inputs.input_ids
                        en_ids = targets.input_ids

                    self.samples.append({
                        'zh_ids': zh_ids,
                        'en_ids': en_ids,
                        'zh_text': zh_text,
                        'en_text': en_text
                    })
                except Exception as e:
                    continue
        print(f"加载 {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 转换为tensor
        zh_tensor = torch.tensor(sample['zh_ids'], dtype=torch.long)
        en_tensor = torch.tensor(sample['en_ids'], dtype=torch.long)
        
        return {
            'src': zh_tensor,  # 中文（源语言）
            'tgt': en_tensor,  # 英文（目标语言）
            'src_text': sample['zh_text'],
            'tgt_text': sample['en_text']
        }
    
    @staticmethod
    def collate_fn(batch, pad_token_id=0):
        """
        批量处理函数（填充序列）
        
        Args:
            batch: 批次数据
            pad_token_id: 填充token的ID
            
        Returns:
            填充后的批次数据
        """
        src_seqs = [item['src'] for item in batch]
        tgt_seqs = [item['tgt'] for item in batch]
        src_texts = [item['src_text'] for item in batch]
        tgt_texts = [item['tgt_text'] for item in batch]
        
        # 填充到最大长度
        src_padded = torch.nn.utils.rnn.pad_sequence(
            src_seqs, batch_first=True, padding_value=pad_token_id
        )
        tgt_padded = torch.nn.utils.rnn.pad_sequence(
            tgt_seqs, batch_first=True, padding_value=pad_token_id
        )
        
        # 创建注意力掩码
        return {
            'src': src_padded,
            'tgt': tgt_padded,
            'src_mask': (src_padded != pad_token_id).long(),
            'tgt_mask': (tgt_padded != pad_token_id).long(),
            'src_text': src_texts,
            'tgt_text': tgt_texts
        }
        

def create_data_loaders(train_path: str, val_path: str, tokenizer: BilingualTokenizer,
                       batch_size=32):
    """
    创建训练和验证数据加载器
    
    Args:
        train_path: 训练数据路径
        val_path: 验证数据路径
        tokenizer: 分词器
        batch_size: 批次大小
        
    Returns:
        train_loader, val_loader
    """
    # 创建数据集
    train_dataset = TranslationDataset(train_path, tokenizer)
    val_dataset = TranslationDataset(val_path, tokenizer)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: TranslationDataset.collate_fn(batch, pad_token_id=0)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: TranslationDataset.collate_fn(batch, pad_token_id=0)
    )
    
    print(f"训练集: {len(train_dataset)} 个样本")
    print(f"验证集: {len(val_dataset)} 个样本")
    
    return train_loader, val_loader