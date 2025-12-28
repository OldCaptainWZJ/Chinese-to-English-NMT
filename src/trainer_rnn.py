import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from pathlib import Path

class RNNTrainer:
    """RNN Trainer supporting Policy Experiments"""
    
    def __init__(self, model, device, tokenizer, checkpoint_dir='checkpoints', 
                 tf_strategy='decay', fixed_tf_ratio=0.5):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer # Needed for decoding text
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        self.optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-6)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)

        # --- Experiment: Training Policy ---
        self.tf_strategy = tf_strategy  # 'decay', 'fixed'
        self.fixed_tf_ratio = fixed_tf_ratio
        self.current_tf_ratio = fixed_tf_ratio if tf_strategy == 'fixed' else 0.8
        self.tf_decay_rate = 0.95

        self.losses = []
        self.continue_epoch = 0
        self.continue_best_val_loss = float('inf')

    def get_tf_ratio(self, epoch):
        """Determine Teacher Forcing Ratio based on strategy"""
        if self.tf_strategy == 'fixed':
            return self.fixed_tf_ratio
        else:
            # Decay strategy
            return max(0.1, 0.8 * (self.tf_decay_rate ** (epoch - 1)))

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        total_tokens = 0
        
        # Calculate TF ratio for this epoch
        tf_ratio = self.get_tf_ratio(epoch)
        self.current_tf_ratio = tf_ratio
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch} [TF: {tf_ratio:.2f}]')
        
        for batch in progress_bar:
            src_tokens = batch['src'].to(self.device)
            tgt_tokens = batch['tgt'].to(self.device)
            src_lengths = batch['src_mask'].sum(dim=1).to(self.device)
            tgt_lengths = batch['tgt_mask'].sum(dim=1).to(self.device)

            outputs, _ = self.model(src_tokens, src_lengths, tgt_tokens, tgt_lengths, teacher_forcing_ratio=tf_ratio)
            
            outputs = outputs[:, 1:, :].contiguous().view(-1, outputs.size(-1))
            targets = tgt_tokens[:, 1:].contiguous().view(-1)
            
            loss = self.criterion(outputs, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            batch_loss = loss.item()
            batch_tokens = targets.ne(0).sum().item()
            total_loss += batch_loss * batch_tokens
            total_tokens += batch_tokens
            
            progress_bar.set_postfix({'loss': batch_loss, 'ppl': np.exp(batch_loss)})
        
        return total_loss / total_tokens if total_tokens > 0 else 0, np.exp(total_loss / total_tokens)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in val_loader:
                src_tokens = batch['src'].to(self.device)
                tgt_tokens = batch['tgt'].to(self.device)
                src_lengths = batch['src_mask'].sum(dim=1).to(self.device)
                tgt_lengths = batch['tgt_mask'].sum(dim=1).to(self.device)

                # Validation always runs without Teacher Forcing for loss calc
                outputs, _ = self.model(src_tokens, src_lengths, tgt_tokens, tgt_lengths, teacher_forcing_ratio=0.0)
                
                outputs = outputs[:, 1:, :].contiguous().view(-1, outputs.size(-1))
                targets = tgt_tokens[:, 1:].contiguous().view(-1)
                
                loss = self.criterion(outputs, targets)
                batch_tokens = targets.ne(0).sum().item()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens
        
        avg_loss = total_loss / total_tokens
        return avg_loss, np.exp(avg_loss)

    def test_decoding_strategies(self, val_loader, num_samples=3):
        """
        Experiment: Decoding Policy
        Compare Greedy vs Beam Search on a few samples from validation set.
        """
        print(f"\n--- Decoding Strategy Comparison (Epoch {self.continue_epoch}) ---")
        self.model.eval()
        
        # Get one batch
        batch = next(iter(val_loader))
        src_tokens = batch['src'].to(self.device)
        src_lengths = batch['src_mask'].sum(dim=1).to(self.device)
        tgt_tokens = batch['tgt'].to(self.device)
        
        with torch.no_grad():
            for i in range(min(num_samples, src_tokens.size(0))):
                src = src_tokens[i].unsqueeze(0) # [1, len]
                src_len = src_lengths[i].unsqueeze(0)
                
                # Ground Truth
                target_ids = tgt_tokens[i].cpu().numpy().tolist()
                target_text = self.tokenizer.decode(target_ids)
                
                # 1. Greedy Decode
                greedy_ids = self.model.decode(src, src_len, method='greedy')
                greedy_text = self.tokenizer.decode(greedy_ids)
                
                # 2. Beam Search Decode
                beam_ids = self.model.decode(src, src_len, method='beam', beam_width=3)
                beam_text = self.tokenizer.decode(beam_ids)
                
                print(f"\nSample {i+1}:")
                print(f"Target: {target_text}")
                print(f"Greedy: {greedy_text}")
                print(f"Beam 3: {beam_text}")
        print("---------------------------------------------------------")

    def train(self, train_loader, val_loader, num_epochs=10):
        best_val_loss = self.continue_best_val_loss
        
        for epoch in range(self.continue_epoch + 1, self.continue_epoch + num_epochs + 1):
            print(f"\n{'='*50}\nEpoch {epoch}/{self.continue_epoch + num_epochs}\n{'='*50}")
            
            train_loss, train_ppl = self.train_epoch(train_loader, epoch)
            print(f"Train Loss: {train_loss:.4f}, Train PPL: {train_ppl:.2f}")
            
            val_loss, val_ppl = self.validate(val_loader)
            print(f"Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")

            self.losses.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
            
            # --- Experiment: Run Decoding Comparison every epoch to see evolution ---
            # if epoch % 1 == 0: 
            #     self.test_decoding_strategies(val_loader)

            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best_model.pt', epoch, best_val_loss)
            
            if epoch % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt', epoch, best_val_loss)
            
            # 实时更新Loss曲线
            self.loss_visualize()
                
    def save_checkpoint(self, filename, epoch, best_val_loss):
        checkpoint_path = self.checkpoint_dir / filename
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
            'best_val_loss': best_val_loss
        }, checkpoint_path)
    
    def load_checkpoint(self, filename):
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