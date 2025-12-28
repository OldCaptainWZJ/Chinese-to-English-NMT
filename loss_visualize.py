import os
import torch
import matplotlib.pyplot as plt
import sys
from matplotlib.ticker import MaxNLocator
from pathlib import Path

def visualize_losses_combined(directory, filename=None, losses=None):
    """
    只显示一个组合图表，强制Epoch轴为整数
    """
    
    if losses is None and not os.path.exists(pt_file_path):
        print(f"错误：文件 '{pt_file_path}' 不存在")
        return
    elif losses is None:
        # 加载数据
        pt_file_path = Path(__file__).parent / "checkpoints" / directory / filename

        data = torch.load(pt_file_path, map_location='cpu')
        
        if 'losses' not in data:
            print("错误：文件中没有找到 'losses' 数据")
            return
        
        losses = data['losses']
    
    try:
        # 提取数据
        epochs = [item['epoch'] for item in losses]
        train_losses = [item['train_loss'] for item in losses]
        val_losses = [item['val_loss'] for item in losses]
        
        # 创建图表
        plt.figure(figsize=(10, 6))
        
        # 绘制两条曲线
        plt.plot(epochs, train_losses, 'b-', linewidth=2, marker='o', markersize=4, label='Train Loss')
        plt.plot(epochs, val_losses, 'r-', linewidth=2, marker='s', markersize=4, label='Val Loss')
        
        # 设置坐标轴标签
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'Loss Curve', fontsize=14, fontweight='bold')
        
        # 添加图例
        plt.legend(fontsize=12)
        
        # 添加网格
        plt.grid(True, alpha=0.3)
        
        # 强制Epoch轴为整数
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # 确保x轴范围从最小epoch开始
        if epochs:
            plt.xlim(min(epochs) - 0.5, max(epochs) + 0.5)
        
        # 自动调整布局
        plt.tight_layout()
        
        # 保存图像
        save_directory = Path(__file__).parent / "visualize_output"
        save_name = directory + '_loss_curve.png'
        save_path = save_directory / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存为: {save_path}")
        
        # 显示图表
        # plt.show()
        
    except Exception as e:
        print(f"错误：{e}")

# 使用示例
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python loss_visualizer.py <子目录名> <pt文件名>")
        print("示例: python loss_visualizer.py rnn-teacher model_losses.pt")
    else:
        directory = sys.argv[1]
        filename = sys.argv[2]
        visualize_losses_combined(directory, filename)