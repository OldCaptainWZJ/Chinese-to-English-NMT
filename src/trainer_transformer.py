import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from pathlib import Path

class TransformerTrainer:
    def __init__(self, model, device, checkpoint_dir='checkpoints/transformer'):
        self.model = model
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 兼容 mT5 (它内部计算loss) 和 Custom Transformer
        self.is_pretrained = hasattr(model, 'model') and 'mt5' in str(type(model.model))

        print("pretrained: ", self.is_pretrained)
        
        if not self.is_pretrained:
            self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
            # Transformer 通常使用 AdamW 和 Warmup
            self.optimizer = optim.AdamW(model.parameters(), lr=0.0005, betas=(0.9, 0.98), eps=1e-9)
        else:
            # mT5 Fine-tuning parameters
            self.optimizer = optim.AdamW(model.parameters(), lr=5e-5)
        
        self.continue_epoch = 0
        self.losses = []

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch in progress_bar:
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            # src_len/tgt_len for API compatibility
            src_len = batch['src_mask'].sum(1).to(self.device)
            tgt_len = batch['tgt_mask'].sum(1).to(self.device)
            
            # print("src_text: ", batch['src_text'][0])
            # print("tgt_text: ", batch['tgt_text'][0])
            # print("src: ", src[0])
            # print("tgt: ", tgt[0])

            self.optimizer.zero_grad()
            
            if self.is_pretrained:
                logits, loss = self.model(src, src_len, tgt, tgt_len)
    
                # 解码预测的token ids
                # 取logits中概率最大的token（贪婪解码）
                predicted_ids = torch.argmax(logits, dim=-1)
    
                # 解码为文本
                # 过滤掉padding token (通常为0)
                valid_tokens = predicted_ids[0][predicted_ids[0] != 0]
                # 解码为文本
                text = train_loader.dataset.tokenizer.decode(valid_tokens, skip_special_tokens=True)

                # print("predicted: ", predicted_ids[0])
                # print("output_text: ", text)

            else:
                # Custom Transformer inputs: src, src_len, tgt_input, tgt_len
                # Shift tgt for input and target:
                # Input: <sos> A B C
                # Target: A B C <eos>
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                # Forward
                outputs, _ = self.model(src, src_len, tgt_input, tgt_len)
                
                # Reshape for loss
                output_dim = outputs.shape[-1]
                outputs = outputs.contiguous().view(-1, output_dim)
                targets = tgt_output.contiguous().view(-1)
                
                loss = self.criterion(outputs, targets)

                # predicted_ids = torch.argmax(outputs, dim=-1)
                # valid_tokens = predicted_ids[0][predicted_ids[0] != 0]
                # text = train_loader.dataset.tokenizer.decode(valid_tokens, skip_special_tokens=True)

                # print("predicted: ", predicted_ids[0])
                # print("output_text: ", text)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                src_len = batch['src_mask'].sum(1).to(self.device)
                tgt_len = batch['tgt_mask'].sum(1).to(self.device)
                
                if self.is_pretrained:
                    _, loss = self.model(src, src_len, tgt, tgt_len)
                else:
                    tgt_input = tgt[:, :-1]
                    tgt_output = tgt[:, 1:]
                    outputs, _ = self.model(src, src_len, tgt_input, tgt_len)
                    output_dim = outputs.shape[-1]
                    loss = self.criterion(outputs.contiguous().view(-1, output_dim), 
                                          tgt_output.contiguous().view(-1))
                
                total_loss += loss.item()
                
        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader, num_epochs=10):
        best_val_loss = float('inf')
        
        for epoch in range(self.continue_epoch + 1, self.continue_epoch + num_epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)
            
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            
            self.losses.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.checkpoint_dir / "best_model.pt")
                print("Saved Best Model")

            # 定期保存检查点
            if epoch % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt', epoch, best_val_loss)
            
            # 实时更新Loss曲线
            self.loss_visualize()
            
    def save_checkpoint(self, filename, epoch, best_val_loss):
        """保存检查点"""
        checkpoint_path = self.checkpoint_dir / filename
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
            'best_val_loss': best_val_loss
        }, checkpoint_path)
    
    def load_checkpoint(self, filename):
        """加载检查点"""
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.continue_epoch = checkpoint['epoch']
        self.continue_best_val_loss = checkpoint['best_val_loss']
        self.losses = checkpoint['losses']
    
    def loss_visualize(self):
        from loss_visualize import visualize_losses_combined

        directory = self.checkpoint_dir.parts[-1]
        visualize_losses_combined(directory=directory, losses=self.losses)