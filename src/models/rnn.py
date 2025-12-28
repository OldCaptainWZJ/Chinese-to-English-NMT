import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    """单向 2层 LSTM 编码器"""
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, num_layers=2, 
                           bidirectional=False, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, src_len.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, (hidden, cell) = self.rnn(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        return outputs, hidden

class Attention(nn.Module):
    """
    Supports Dot, General (Multiplicative), and Concat (Additive) Attention.
    """
    def __init__(self, enc_hid_dim, dec_hid_dim, method='concat'):
        super().__init__()
        self.method = method
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        
        if self.method == 'concat':
            self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
            self.v = nn.Linear(dec_hid_dim, 1, bias=False)
        elif self.method == 'general':
            self.attn = nn.Linear(enc_hid_dim, dec_hid_dim)
        elif self.method == 'dot':
            # For dot, dimensions must match. If not, we might need a projection or raise error
            if enc_hid_dim != dec_hid_dim:
                # Optional: Add projection if dimensions differ
                self.attn = nn.Linear(enc_hid_dim, dec_hid_dim)
                self.method = 'general' # Fallback to general if dims differ
                print("Warning: enc_dim != dec_dim, switching 'dot' to 'general' attention.")

    def forward(self, hidden, encoder_outputs, mask):
        # hidden: [batch_size, dec_hid_dim]
        # encoder_outputs: [batch_size, src_len, enc_hid_dim]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        # Calculate alignment scores
        if self.method == 'concat':
            # hidden expanded: [batch, src_len, dec_dim]
            hidden_expanded = hidden.unsqueeze(1).repeat(1, src_len, 1)
            energy = torch.tanh(self.attn(torch.cat((hidden_expanded, encoder_outputs), dim=2)))
            attention_scores = self.v(energy).squeeze(2) # [batch, src_len]
            
        elif self.method == 'general':
            # energy: [batch, src_len, dec_dim]
            energy = self.attn(encoder_outputs) 
            # score: hidden * energy
            # hidden: [batch, dec_dim, 1]
            attention_scores = torch.bmm(energy, hidden.unsqueeze(2)).squeeze(2)
            
        elif self.method == 'dot':
            # hidden: [batch, dec_dim, 1]
            attention_scores = torch.bmm(encoder_outputs, hidden.unsqueeze(2)).squeeze(2)

        # Masking
        attention_scores = attention_scores.masked_fill(mask == 0, -1e10)
        
        return F.softmax(attention_scores, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(enc_hid_dim + emb_dim, dec_hid_dim, 
                           num_layers=2, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(enc_hid_dim + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs, mask):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        
        # Calculate Attention using the last layer of hidden state
        a = self.attention(hidden[-1], encoder_outputs, mask)
        a = a.unsqueeze(1)
        
        weighted = torch.bmm(a, encoder_outputs)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        
        return prediction, hidden, cell, a.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, pad_idx=0, sos_idx=2, eos_idx=3):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        
    def forward(self, src, src_len, tgt, tgt_len, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        max_len = tgt.shape[1] 
        vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, max_len, vocab_size).to(self.device)
        encoder_mask = (src != self.pad_idx).long()
        
        encoder_outputs, hidden = self.encoder(src, src_len)
        # cell = torch.zeros_like(hidden).to(self.device)
        cell = hidden
        input = tgt[:, 0]
        
        for t in range(1, max_len):
            output, hidden, cell, _ = self.decoder(input, hidden, cell, encoder_outputs, encoder_mask)
            outputs[:, t, :] = output
            best_guess = output.argmax(1)
            
            if self.training:
                # Teacher Forcing logic
                teacher_force = random.random() < teacher_forcing_ratio
                top1 = output.argmax(1) 
                input = tgt[:, t] if teacher_force else top1
            else:
                input = best_guess
                
        return outputs, None

    def decode(self, src, src_len, tgt=None, method='greedy', beam_width=3, max_len=None):
        # Ensure self.eval() is called
        self.eval()
        
        if max_len is None:
            max_len = 512 # Default cap

        device = self.device
        batch_size = src.shape[0]
        if batch_size != 1:
            raise ValueError("Decode supports batch_size=1 only")

        encoder_mask = (src != self.pad_idx).long()
        encoder_outputs, hidden = self.encoder(src, src_len)
        cell = torch.zeros_like(hidden).to(device)
        
        if method == 'greedy':
            decoded_indices = []
            input_token = torch.tensor([self.sos_idx], device=device)
            for _ in range(max_len):
                output, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs, encoder_mask)
                prediction = output.argmax(1)
                item = prediction.item()
                if item == self.eos_idx: break
                decoded_indices.append(item)
                input_token = prediction
            return decoded_indices

        elif method == 'beam':
            # Simple Beam implementation
            start_node = (0.0, torch.tensor([self.sos_idx], device=device), hidden, cell, [])
            beams = [start_node]
            completed_seqs = []
            
            for _ in range(max_len):
                candidates = []
                for score, input_token, h, c, seq in beams:
                    output, new_h, new_c, _ = self.decoder(input_token, h, c, encoder_outputs, encoder_mask)
                    log_probs = F.log_softmax(output, dim=1)
                    topk_log_probs, topk_indices = log_probs.topk(beam_width, dim=1)
                    
                    for k in range(beam_width):
                        idx = topk_indices[0, k]
                        prob = topk_log_probs[0, k].item()
                        new_score = score + prob
                        new_idx = idx.item()
                        
                        if new_idx == self.eos_idx:
                            final_score = new_score / (len(seq) + 1)
                            completed_seqs.append((final_score, seq))
                        else:
                            candidates.append((new_score, idx.unsqueeze(0), new_h.clone(), new_c.clone(), seq + [new_idx]))
                
                if not candidates: break
                candidates.sort(key=lambda x: x[0], reverse=True)
                beams = candidates[:beam_width]
                if len(completed_seqs) >= beam_width: break # Optimization

            if not completed_seqs:
                return beams[0][4]
            
            completed_seqs.sort(key=lambda x: x[0], reverse=True)
            return completed_seqs[0][1]
            
def build_model(src_vocab_size, tgt_vocab_size, device, 
                enc_emb_dim=256, dec_emb_dim=256, 
                enc_hid_dim=512, dec_hid_dim=512, 
                dropout=0.1, attention_type='concat'):
    
    # Initialize Attention based on type
    attn = Attention(enc_hid_dim, dec_hid_dim, method=attention_type)
    
    enc = Encoder(src_vocab_size, enc_emb_dim, enc_hid_dim, dec_hid_dim, dropout)
    dec = Decoder(tgt_vocab_size, dec_emb_dim, enc_hid_dim, dec_hid_dim, dropout, attn)
    
    model = Seq2Seq(enc, dec, device).to(device)
    
    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)
    model.apply(init_weights)
    return model