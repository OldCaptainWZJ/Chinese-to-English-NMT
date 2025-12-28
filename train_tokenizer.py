from pathlib import Path

from src.tokenizer import BilingualTokenizer
from src.dataloader import create_data_loaders
import json

def main():
    # 配置路径
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "dataset"
    MODEL_DIR = BASE_DIR / "checkpoints"
    PROCESSED_DIR = DATA_DIR / "processed"
    TOKENIZER_DIR = MODEL_DIR / "tokenizer"
    
    # 确保目录存在
    for dir_path in [DATA_DIR, MODEL_DIR, PROCESSED_DIR, TOKENIZER_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 数据文件在dataset目录
    train_path = DATA_DIR / "train_100k.jsonl"
    val_path = DATA_DIR / "valid.jsonl"
    
    # 检查文件是否存在
    if not train_path.exists():
        print(f"错误: 找不到数据文件 {train_path}")
        print("请先从指定链接下载数据文件")
        return
    
    # 1. 初始化分词器
    print("\n1. 初始化双语分词器...")
    tokenizer = BilingualTokenizer(
        vocab_size=20000,  # 词表大小
        model_prefix="bpe_model"
    )
    
    # 2. 准备语料
    print("\n2. 准备训练语料...")
    zh_corpus_path, en_corpus_path = tokenizer.prepare_corpus(
        data_path=train_path,
        output_dir=PROCESSED_DIR
    )
    
    # 3. 训练分词器
    print("\n3. 训练SentencePiece分词器...")
    tokenizer.train_tokenizers(
        zh_corpus_path=zh_corpus_path,
        en_corpus_path=en_corpus_path,
        model_dir=TOKENIZER_DIR
    )
    
    # 4. 保存分词器
    print("\n4. 保存分词器...")
    tokenizer.save_tokenizers(TOKENIZER_DIR)
    
    print("\n" + "=" * 50)
    print("数据准备完成!")
    print(f"分词器模型保存在: {TOKENIZER_DIR}")
    print("=" * 50)

if __name__ == "__main__":
    main()