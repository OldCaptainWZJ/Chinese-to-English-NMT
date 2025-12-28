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
    train_path = DATA_DIR / "train_10k.jsonl"  # 10k数据（测试）
    val_path = DATA_DIR / "valid.jsonl"
    
    # 检查文件是否存在
    if not train_path.exists():
        print(f"错误: 找不到数据文件 {train_path}")
        print("请先从指定链接下载数据文件")
        return
    
    tokenizer = BilingualTokenizer(
        vocab_size=20000,  # 词表大小
        model_prefix="bpe_model"
    )
    tokenizer.load_tokenizers(TOKENIZER_DIR)

    # 1. 创建数据加载器（测试用）
    print("\n1. 创建数据加载器...")
    train_loader, val_loader = create_data_loaders(
        train_path=train_path,
        val_path=val_path,
        tokenizer=tokenizer,
        batch_size=4,  # 小批量用于测试
    )
    
    # 2. 测试数据加载器
    print("\n2. 测试数据加载器...")
    test_batch = next(iter(val_loader))
    print(f"批次形状 - src: {test_batch['src'].shape}, tgt: {test_batch['tgt'].shape}")
    print(f"源语言文本示例: {test_batch['src_text'][0][:50]}...")
    print(f"目标语言文本示例: {test_batch['tgt_text'][0][:50]}...")
    
    # 3. 加载分词器
    print("\n3. 加载分词器...")
    tokenizer.load_tokenizers(TOKENIZER_DIR)

    # 4. 测试分词
    print("\n4. 测试分词...")
    for i in range(3):
        print(f"{i}:")
        embed_list_zh = tokenizer.tokenize_zh(test_batch['src_text'][i])
        embed_list_en = tokenizer.tokenize_en(test_batch['tgt_text'][i])
        print("before_zh: ", test_batch['src_text'][i])
        print("before_en: ", test_batch['tgt_text'][i])
        print(embed_list_zh)
        print(embed_list_en)
        print("after_zh: ", tokenizer.detokenize_zh(embed_list_zh))
        print("after_en: ", tokenizer.detokenize_en(embed_list_en))
    

if __name__ == "__main__":
    main()