import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = x.norm(keepdim=True, dim=-1)
        x_normed = x / (norm * math.pow(x.size(-1), -0.5) + self.eps)
        return self.scale * x_normed

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, use_relative_pos=False):
        super().__init__()
        assert d_model % n_head == 0
        self.d_k = d_model // n_head
        self.n_head = n_head
        self.use_relative_pos = use_relative_pos
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))

        if use_relative_pos:
            # 简化的相对位置偏差 (Learnable bias based on distance)
            self.max_relative_pos = 16
            self.relative_embeddings = nn.Embedding(self.max_relative_pos * 2 + 1, self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # [batch, seq_len, d_model] -> [batch, seq_len, n_head, d_k] -> [batch, n_head, seq_len, d_k]
        Q = self.w_q(query).view(batch_size, -1, self.n_head, self.d_k).permute(0, 2, 1, 3)
        K = self.w_k(key).view(batch_size, -1, self.n_head, self.d_k).permute(0, 2, 1, 3)
        V = self.w_v(value).view(batch_size, -1, self.n_head, self.d_k).permute(0, 2, 1, 3)
        
        self.scale = self.scale.to(query.device)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        # Relative Position Handling
        if self.use_relative_pos:
            q_len = query.shape[1]
            k_len = key.shape[1]
            
            # 1. 创建相对位置索引矩阵
            # 直接在目标 device 上创建以避免传输开销
            range_vec_q = torch.arange(q_len, device=query.device)
            range_vec_k = torch.arange(k_len, device=query.device)
            
            # distance_mat[i][j] = j - i (Key 位置 - Query 位置)
            # shape: [q_len, k_len]
            distance_mat = range_vec_k[None, :] - range_vec_q[:, None] 
            
            # 2. 截断距离 (Clipping) 并移位正数索引
            # 距离范围被限制在 [-16, 16]，然后平移到 [0, 32] 作为 Embedding 索引
            distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_pos, self.max_relative_pos)
            final_mat = distance_mat_clipped + self.max_relative_pos
            
            # 3. 获取位置 Embedding
            # shape: [q_len, k_len, d_k]
            rel_embeddings = self.relative_embeddings(final_mat) 
            
            # 4. 计算 Content-Position Interaction (Q * R^T)
            # Q: [batch, n_head, q_len, d_k]
            # rel_embeddings: [q_len, k_len, d_k]
            # 我们需要输出: [batch, n_head, q_len, k_len]
            # 使用 einsum 处理广播机制: 
            # 'bhqd' (Query) 与 'qkd' (Relative Emb) 点积，保留 'bhqk'
            rel_bias = torch.einsum('bhqd,qkd->bhqk', Q, rel_embeddings)
            
            # 5. 将相对位置偏差加到 attention logits 上
            # 注意：标准的 Attention 公式是 (QK^T + QR^T) / sqrt(d_k)
            # 因为上面的 energy 已经除以了 scale，这里同样除以 scale 以保持量级一致
            energy = energy + (rel_bias / self.scale)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.n_head * self.d_k)
        
        return self.fc(x), attention

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout, norm_type='layernorm', use_relative_pos=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout, use_relative_pos)
        self.pff = PositionwiseFeedforward(d_model, d_ff, dropout)
        
        if norm_type == 'rmsnorm':
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        _src, _ = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(_src))
        
        _src = self.pff(src)
        src = self.norm2(src + self.dropout(_src))
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout, norm_type='layernorm', use_relative_pos=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout, use_relative_pos)
        self.enc_attn = MultiHeadAttention(d_model, n_head, dropout, False) # Cross attn usually doesn't use relative
        self.pff = PositionwiseFeedforward(d_model, d_ff, dropout)
        
        if norm_type == 'rmsnorm':
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
            self.norm3 = RMSNorm(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, enc_src, tgt_mask, src_mask):
        _tgt, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(_tgt))
        
        _tgt, _ = self.enc_attn(tgt, enc_src, enc_src, src_mask)
        tgt = self.norm2(tgt + self.dropout(_tgt))
        
        _tgt = self.pff(tgt)
        tgt = self.norm3(tgt + self.dropout(_tgt))
        return tgt

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, device, 
                 d_model=256, n_head=8, num_layers=3, d_ff=512, dropout=0.1, max_len=512,
                 norm_type='layernorm', pos_emb_type='absolute', pad_idx=0, sos_idx=2, eos_idx=3):
        super().__init__()
        self.device = device
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.max_len = max_len
        self.pos_emb_type = pos_emb_type
        
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Absolute Positional Encoding
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        self.enc_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_head, d_ff, dropout, norm_type, pos_emb_type=='relative')
            for _ in range(num_layers)
        ])
        
        self.dec_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_head, d_ff, dropout, norm_type, pos_emb_type=='relative')
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)

    def make_src_mask(self, src):
        # src: [batch, src_len]
        mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        # mask: [batch, 1, 1, src_len]
        return mask

    def make_tgt_mask(self, tgt):
        # tgt: [batch, tgt_len]
        mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        # mask: [batch, 1, 1, tgt_len]
        
        tgt_len = tgt.shape[1]
        tril = torch.tril(torch.ones((tgt_len, tgt_len), device=self.device)).bool()
        mask = mask & tril.unsqueeze(0).unsqueeze(0)
        return mask

    def forward(self, src, src_len, tgt, tgt_len, teacher_forcing_ratio=None):
        # 注意：Transformer 训练时通常一次性并行计算，不需要 teacher_forcing_ratio 循环
        # 为了兼容 RNN Trainer 接口，我们接受这些参数但可能忽略其中一些
        
        batch_size, src_seq_len = src.shape
        tgt_seq_len = tgt.shape[1]
        
        # 1. Embedding
        src_emb = self.src_embedding(src) * self.scale
        tgt_emb = self.tgt_embedding(tgt) * self.scale
        
        if self.pos_emb_type == 'absolute':
            pos_src = torch.arange(0, src_seq_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
            pos_tgt = torch.arange(0, tgt_seq_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
            src_emb = src_emb + self.pos_embedding(pos_src)
            tgt_emb = tgt_emb + self.pos_embedding(pos_tgt)
            
        src_emb = self.dropout(src_emb)
        tgt_emb = self.dropout(tgt_emb)
        
        # 2. Masks
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        # 3. Encoder
        enc_out = src_emb
        for layer in self.enc_layers:
            enc_out = layer(enc_out, src_mask)
            
        # 4. Decoder
        dec_out = tgt_emb
        for layer in self.dec_layers:
            dec_out = layer(dec_out, enc_out, tgt_mask, src_mask)
            
        output = self.fc_out(dec_out)
        return output, None # None fits the RNN return signature (output, hidden)

    def decode(self, src, src_len, tgt=None, method='greedy', max_len=512):
        """推理/评估函数"""
        self.eval()
        device = self.device
        
        # Encoder
        src_mask = self.make_src_mask(src)
        src_emb = self.src_embedding(src) * self.scale
        if self.pos_emb_type == 'absolute':
            pos = torch.arange(0, src.shape[1]).unsqueeze(0).to(device)
            src_emb = src_emb + self.pos_embedding(pos)
        src_emb = self.dropout(src_emb)
        
        enc_out = src_emb
        for layer in self.enc_layers:
            enc_out = layer(enc_out, src_mask)
            
        # Decoder initialization
        decoded_indices = [self.sos_idx]
        
        for _ in range(max_len):
            tgt_tensor = torch.LongTensor(decoded_indices).unsqueeze(0).to(device)
            tgt_mask = self.make_tgt_mask(tgt_tensor)
            
            tgt_emb = self.tgt_embedding(tgt_tensor) * self.scale
            if self.pos_emb_type == 'absolute':
                pos = torch.arange(0, tgt_tensor.shape[1]).unsqueeze(0).to(device)
                tgt_emb = tgt_emb + self.pos_embedding(pos)
            tgt_emb = self.dropout(tgt_emb)
            
            dec_out = tgt_emb
            for layer in self.dec_layers:
                dec_out = layer(dec_out, enc_out, tgt_mask, src_mask)
                
            prediction = self.fc_out(dec_out)
            # 取最后一个token的输出
            next_token_logits = prediction[:, -1, :]
            next_token = next_token_logits.argmax(1).item()
            
            if next_token == self.eos_idx:
                break
            
            decoded_indices.append(next_token)
            
        return decoded_indices[1:] # 去掉 SOS

def build_transformer_model(src_vocab_size, tgt_vocab_size, device, **kwargs):
    return Transformer(src_vocab_size, tgt_vocab_size, device, **kwargs).to(device)