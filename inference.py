import torch
import argparse
import sys
from pathlib import Path
import logging

# 引入项目模块
from src.tokenizer import BilingualTokenizer
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

# 模型构建函数导入
from src.models.rnn import build_model as build_rnn_model
from src.models.transformer import build_transformer_model
from src.models.pretrained import build_pretrained_model

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class HFTokenizerWrapper:
    """HuggingFace Tokenizer 适配器，使其接口与 BilingualTokenizer 兼容"""
    def __init__(self, hf_tokenizer):
        self.tokenizer = hf_tokenizer
        self.pad_id = hf_tokenizer.pad_token_id
        self.eos_id = hf_tokenizer.eos_token_id
        self.sos_id = hf_tokenizer.pad_token_id  # T5等模型通常不需要显式SOS，或者使用pad/eos
        
    def encode_zh(self, text):
        """将中文文本转换为 token ids (tensor)"""
        # return_tensors='pt' 返回 tensor
        return self.tokenizer(text, return_tensors='pt', padding=False).input_ids.squeeze(0)

    def detokenize_en(self, token_ids):
        """将 token ids 列表转换为英文文本"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

class InferenceEngine:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logger
        self.model_type = args.model_type
        
        # 1. 初始化分词器
        self._init_tokenizer()
        
        # 2. 初始化并加载模型
        self._init_model()
        
    def _init_tokenizer(self):
        self.logger.info("正在加载分词器...")
        if self.model_type == 'pretrained':
            if AutoTokenizer is None:
                raise ImportError("请安装 transformers: pip install transformers")
            raw_tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model_name)
            self.tokenizer = HFTokenizerWrapper(raw_tokenizer)
            self.src_vocab_size = raw_tokenizer.vocab_size
            self.tgt_vocab_size = raw_tokenizer.vocab_size
        else:
            # RNN 和 Scratch Transformer 使用自定义 BilingualTokenizer
            tokenizer_dir = Path(self.args.tokenizer_dir)
            if not tokenizer_dir.exists():
                # 尝试从 checkpoint 目录推断
                tokenizer_dir = Path(self.args.checkpoint).parent.parent / "tokenizer"
            
            if not tokenizer_dir.exists():
                raise FileNotFoundError(f"找不到分词器目录，请通过 --tokenizer_dir 指定。尝试路径: {tokenizer_dir}")

            self.tokenizer = BilingualTokenizer()
            self.tokenizer.load_tokenizers(tokenizer_dir)
            self.src_vocab_size = self.tokenizer.sp_zh.get_piece_size()
            self.tgt_vocab_size = self.tokenizer.sp_en.get_piece_size()
            self.logger.info(f"分词器加载成功。词表大小: Src={self.src_vocab_size}, Tgt={self.tgt_vocab_size}")

    def _init_model(self):
        self.logger.info(f"正在构建模型 ({self.model_type})...")
        
        if self.model_type == 'rnn':
            self.model = build_rnn_model(
                src_vocab_size=self.src_vocab_size,
                tgt_vocab_size=self.tgt_vocab_size,
                device=self.device,
                enc_emb_dim=self.args.enc_emb_dim,
                dec_emb_dim=self.args.dec_emb_dim,
                enc_hid_dim=self.args.enc_hid_dim,
                dec_hid_dim=self.args.dec_hid_dim,
                dropout=0.5,
                attention_type=self.args.attention_type
            )
        elif self.model_type == 'transformer':
            self.model = build_transformer_model(
                src_vocab_size=self.src_vocab_size,
                tgt_vocab_size=self.tgt_vocab_size,
                device=self.device,
                d_model=self.args.d_model,
                n_head=self.args.n_head,
                num_layers=self.args.num_layers,
                norm_type=self.args.norm_type,
                pos_emb_type=self.args.pos_emb
            )
        elif self.model_type == 'pretrained':
            self.model = build_pretrained_model(self.args.pretrained_model_name, self.device)
        
        # 加载权重
        self.logger.info(f"加载检查点: {self.args.checkpoint}")
        checkpoint = torch.load(self.args.checkpoint, map_location=self.device)
        
        # 处理不同的保存格式 (完整dict vs state_dict)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as e:
            self.logger.warning(f"加载权重时出现不匹配，请检查参数配置是否与训练时一致。\n错误信息: {e}")
            sys.exit(1)
            
        self.model.to(self.device)
        self.model.eval()
        self.logger.info("模型加载完成！")

    def preprocess(self, text: str):
        """处理输入文本：分词 -> 添加特殊标记 -> 转Tensor"""
        text = text.strip()
        
        if self.model_type == 'pretrained':
            # HF Tokenizer 自动处理
            src_tensor = self.tokenizer.encode_zh(text).to(self.device)
            # 增加 batch 维度 (1, seq_len)
            src_tensor = src_tensor.unsqueeze(0)
            src_len = torch.tensor([src_tensor.size(1)]).to(self.device)
        else:
            # BilingualTokenizer 需要手动处理 SOS/EOS
            # 获取 ID
            ids = self.tokenizer.tokenize_zh(text)
            
            # RNN/Transformer 通常逻辑: [SOS] + tokens + [EOS]
            # 注意：BilingualTokenizer.tokenize_zh 可能只返回 ids，不含特殊标记
            # 假设 ids 是纯文本 id
            sos_id = 2 # 假设
            eos_id = 3 # 假设
            # 尝试从 tokenizer 获取属性，如果没有则使用默认
            if hasattr(self.tokenizer, 'sos_id'): sos_id = self.tokenizer.sos_id
            if hasattr(self.tokenizer, 'eos_id'): eos_id = self.tokenizer.eos_id
            
            ids = [sos_id] + ids + [eos_id]
            
            src_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)
            src_len = torch.tensor([len(ids)]).to(self.device)
            
        return src_tensor, src_len

    def translate(self, text: str, method="greedy"):
        """执行翻译"""
        with torch.no_grad():
            src_tokens, src_lengths = self.preprocess(text)
            
            # 构造 dummy tgt 用于某些模型的 decode 接口 (如果需要)
            # RNN decode 需要 tgt 作为其实始化 token 或者 teacher forcing (此处不用)
            # 为了兼容性，构造一个只包含 SOS 的 tgt
            sos_id = 2
            if hasattr(self.tokenizer, 'sos_id'): sos_id = self.tokenizer.sos_id
            tgt_tokens = torch.tensor([[sos_id]], device=self.device)

            # 调用模型解码
            try:
                # 获取预测的 token IDs (list 或 tensor)
                predictions = self.model.decode(src_tokens, src_lengths, tgt_tokens, method)
                    
                # 如果返回的是 batch 列表，取第一个
                if isinstance(predictions, list): # RNN 返回 list of lists
                    # RNN 的 decode 通常返回 [batch_size, seq_len] 的 numpy array 或者 list
                    # 这里假设是 batch_size=1
                    import numpy as np
                    if isinstance(predictions, np.ndarray):
                        pred_ids = predictions[0]
                    elif isinstance(predictions[0], list):
                        pred_ids = predictions[0]
                    else:
                        pred_ids = predictions
                else:
                    pred_ids = predictions[0] # Tensor case
                        
            except TypeError:
                # 备用：如果 decode 签名不同
                predictions = self.model.decode(src_tokens, src_lengths)
                pred_ids = predictions[0]

            # 后处理：转回文本
            # 确保 pred_ids 是 list
            if isinstance(pred_ids, torch.Tensor):
                pred_ids = pred_ids.cpu().tolist()
            
            # 去除特殊标记 (SOS, EOS, PAD)
            clean_ids = []
            for tid in pred_ids:
                if tid in [0, 2]: # PAD, SOS (假设)
                    continue
                if tid == 3: # EOS (假设)
                    break
                # 针对 HF tokenizer 的特殊处理
                if hasattr(self.tokenizer, 'eos_id') and tid == self.tokenizer.eos_id:
                    break
                clean_ids.append(tid)
            
            translated_text = self.tokenizer.detokenize_en(clean_ids)
            return translated_text

def get_args():
    parser = argparse.ArgumentParser(description='中译英 NMT 一键推理脚本')
    
    # 基础配置
    parser.add_argument('--checkpoint', type=str, required=True, help='模型权重文件路径 (.pt)')
    parser.add_argument('--model_type', type=str, default='rnn', 
                       choices=['rnn', 'transformer', 'pretrained'], help='模型架构类型')
    parser.add_argument('--tokenizer_dir', type=str, default='checkpoints/tokenizer', 
                       help='分词器目录 (仅用于非 pretrained 模型)')
    parser.add_argument('--decoding', type=str, default='greedy', choices=['greedy', 'beam'],
                       help='解码策略')

    # RNN 特定参数 (需与训练一致)
    parser.add_argument('--attention_type', type=str, default='concat', choices=['concat', 'general', 'dot'])
    parser.add_argument('--enc_emb_dim', type=int, default=256)
    parser.add_argument('--dec_emb_dim', type=int, default=256)
    parser.add_argument('--enc_hid_dim', type=int, default=512)
    parser.add_argument('--dec_hid_dim', type=int, default=512)

    # Transformer (Scratch) 特定参数 (需与训练一致)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--norm_type', type=str, default='layernorm', choices=['layernorm', 'rmsnorm'])
    parser.add_argument('--pos_emb', type=str, default='absolute', choices=['absolute', 'relative'])

    # Transformer (Pretrained) 特定参数
    parser.add_argument('--pretrained_model_name', type=str, default='t5-base')

    return parser.parse_args()

def main():
    args = get_args()
    
    try:
        engine = InferenceEngine(args)
    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
        print("提示: 请检查参数是否与训练时的配置（如 attention_type, hidden_dims）一致。")
        return

    print("\n" + "="*50)
    print(f"   中译英翻译系统 ({args.model_type})")
    print("="*50)
    print("输入 'q' 或 'quit' 退出")
    print("-" * 50)

    while True:
        try:
            src_text = input("\n[中文] > ")
            if src_text.lower() in ['q', 'quit', 'exit']:
                break
            if not src_text.strip():
                continue
                
            translation = engine.translate(src_text, method=args.decoding)
            print(f"[英文] > {translation}")
            
        except KeyboardInterrupt:
            print("\n退出...")
            break
        except Exception as e:
            print(f"推理出错: {e}")

if __name__ == "__main__":
    main()