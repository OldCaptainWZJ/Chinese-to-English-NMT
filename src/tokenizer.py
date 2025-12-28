import os
import sentencepiece as spm
import jieba
import json
from typing import List, Tuple, Dict
from collections import Counter
import pandas as pd
from pathlib import Path

class BilingualTokenizer:
    """中英双语分词器（使用SentencePiece和Jieba）"""
    
    def __init__(self, vocab_size=20000, model_prefix="bpe"):
        """
        初始化分词器
        
        Args:
            vocab_size: 词表大小
            model_prefix: 模型文件前缀
        """
        self.vocab_size = vocab_size
        self.model_prefix = model_prefix
        self.sp_zh = None  # 中文SentencePiece模型
        self.sp_en = None  # 英文SentencePiece模型
        self.vocab_zh = {}
        self.vocab_en = {}
        self.idx2word_zh = {}
        self.idx2word_en = {}
        
    def prepare_corpus(self, data_path: str, output_dir: str):
        """
        准备训练分词器的语料
        
        Args:
            data_path: JSONL数据文件路径
            output_dir: 输出目录
        """
        print("准备训练语料...")
        
        # 读取数据
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 分离中英文句子
        zh_sentences = []
        en_sentences = []
        
        for line in lines:
            try:
                data = json.loads(line.strip())
                zh_sentences.append(data['zh'])
                en_sentences.append(data['en'])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"跳过无效行: {e}")
                continue
        
        # 保存到临时文件供SentencePiece训练
        temp_dir = Path(output_dir) / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存中文语料（使用Jieba分词）
        zh_corpus_path = temp_dir / "zh_corpus.txt"
        with open(zh_corpus_path, 'w', encoding='utf-8') as f:
            for sent in zh_sentences[:10000]:  # 使用部分数据训练分词器
                # 使用Jieba分词后用空格连接
                tokens = ' '.join(jieba.cut(sent.strip()))
                f.write(tokens + '\n')
        
        # 保存英文语料
        en_corpus_path = temp_dir / "en_corpus.txt"
        with open(en_corpus_path, 'w', encoding='utf-8') as f:
            for sent in en_sentences[:10000]:
                f.write(sent.strip() + '\n')
        
        return zh_corpus_path, en_corpus_path
    
    def train_tokenizers(self, zh_corpus_path: str, en_corpus_path: str, model_dir: str):
        """
        训练中英文SentencePiece模型
        
        Args:
            zh_corpus_path: 中文语料路径
            en_corpus_path: 英文语料路径
            model_dir: 模型保存目录
        """
        print("训练SentencePiece分词器...")
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练中文分词器（基于Jieba分词的文本）
        zh_model_prefix = model_dir / "zh_bpe"
        spm.SentencePieceTrainer.train(
            input=zh_corpus_path,
            model_prefix=str(zh_model_prefix),
            vocab_size=self.vocab_size,
            character_coverage=0.9995,
            model_type='bpe',
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=['<blank>'],
            unk_piece="<unk>",
            bos_piece="<s>",
            eos_piece="</s>",
            pad_piece="<pad>",
        )
        
        # 训练英文分词器
        en_model_prefix = model_dir / "en_bpe"
        spm.SentencePieceTrainer.train(
            input=en_corpus_path,
            model_prefix=str(en_model_prefix),
            vocab_size=self.vocab_size,
            character_coverage=1.0,
            model_type='bpe',
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=['<blank>'],
            unk_piece="<unk>",
            bos_piece="<s>",
            eos_piece="</s>",
            pad_piece="<pad>",
        )
        
        # 加载训练好的模型
        self.sp_zh = spm.SentencePieceProcessor()
        self.sp_en = spm.SentencePieceProcessor()
        self.sp_zh.load(f"{zh_model_prefix}.model")
        self.sp_en.load(f"{en_model_prefix}.model")
        
        # 构建词表映射
        self._build_vocab_mappings()
        
        print(f"中文词表大小: {self.sp_zh.get_piece_size()}")
        print(f"英文词表大小: {self.sp_en.get_piece_size()}")
    
    def _build_vocab_mappings(self):
        """构建词表ID映射"""
        # 中文词表
        for i in range(self.sp_zh.get_piece_size()):
            piece = self.sp_zh.id_to_piece(i)
            self.vocab_zh[piece] = i
            self.idx2word_zh[i] = piece
        
        # 英文词表
        for i in range(self.sp_en.get_piece_size()):
            piece = self.sp_en.id_to_piece(i)
            self.vocab_en[piece] = i
            self.idx2word_en[i] = piece
    
    def tokenize_zh(self, text: str, add_special_tokens=True) -> List[int]:
        """
        中文分词和编码
        
        Args:
            text: 中文文本
            add_special_tokens: 是否添加特殊标记
            
        Returns:
            token ids列表
        """
        # 先用Jieba分词
        tokens = ' '.join(jieba.cut(text.strip()))
        
        # 用SentencePiece编码
        if add_special_tokens:
            return self.sp_zh.encode(tokens, out_type=int, add_bos=True, add_eos=True)
        else:
            return self.sp_zh.encode(tokens, out_type=int, add_bos=False, add_eos=False)
    
    def tokenize_en(self, text: str, add_special_tokens=True) -> List[int]:
        """
        英文分词和编码
        
        Args:
            text: 英文文本
            add_special_tokens: 是否添加特殊标记
            
        Returns:
            token ids列表
        """
        if add_special_tokens:
            return self.sp_en.encode(text.strip(), out_type=int, add_bos=True, add_eos=True)
        else:
            return self.sp_en.encode(text.strip(), out_type=int, add_bos=False, add_eos=False)
    
    def detokenize_zh(self, token_ids: List[int]) -> str:
        """中文解码"""
        text = self.sp_zh.decode_ids(token_ids)
        return text.replace(' ', '')  # 去掉Jieba添加的空格
    
    def detokenize_en(self, token_ids: List[int]) -> str:
        """英文解码"""
        return self.sp_en.decode_ids(token_ids)
    
    def save_tokenizers(self, save_dir: str):
        """保存分词器"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型文件（SentencePiece会自动保存.model和.vocab）
        # 保存配置
        config = {
            'vocab_size': self.vocab_size,
            'model_prefix': self.model_prefix,
            'zh_vocab_size': self.sp_zh.get_piece_size(),
            'en_vocab_size': self.sp_en.get_piece_size()
        }
        
        config_path = save_dir / "tokenizer_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"分词器保存到: {save_dir}")
    
    def load_tokenizers(self, model_dir: str):
        """加载分词器"""
        model_dir = Path(model_dir)
        
        # 加载中文模型
        zh_model_path = model_dir / "zh_bpe.model"
        self.sp_zh = spm.SentencePieceProcessor()
        self.sp_zh.load(str(zh_model_path))
        
        # 加载英文模型
        en_model_path = model_dir / "en_bpe.model"
        self.sp_en = spm.SentencePieceProcessor()
        self.sp_en.load(str(en_model_path))
        
        # 重建词表映射
        self._build_vocab_mappings()
        
        print("分词器加载成功")