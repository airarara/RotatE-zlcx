# plot_loss.py
import re
import matplotlib.pyplot as plt
import os
import sys

def parse_log_file(log_file_path):
    """从日志文件中解析loss数据"""
    steps = []
    losses = []
    positive_losses = []
    negative_losses = []
    
    with open(log_file_path, 'r') as f:
        for line in f:
            # 匹配格式: Training average loss at step 100: 0.123456
            match = re.search(r'Training average loss at step (\d+): ([\d.]+)', line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                steps.append(step)
                losses.append(loss)
            
            # 解析positive和negative loss
            pos_match = re.search(r'Training average positive_sample_loss at step (\d+): ([\d.]+)', line)
            if pos_match:
                positive_losses.append(float(pos_match.group(2)))
            
            neg_match = re.search(r'Training average negative_sample_loss at step (\d+): ([\d.]+)', line)
            if neg_match:
                negative_losses.append(float(neg_match.group(2)))
    
    return steps, losses, positive_losses, negative_losses

def plot_loss(log_file_path, save_path=None):
    """绘制loss曲线"""
    steps, losses, positive_losses, negative_losses = parse_log_file(log_file_path)
    
    if not steps:
        print(f"未在 {log_file_path} 中找到loss数据")
        return
    
    plt.figure(figsize=(12, 6))
    
    # 绘制总loss
    plt.subplot(1, 2, 1)
    plt.plot(steps, losses, 'b-', label='Total Loss', linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制positive和negative loss
    plt.subplot(1, 2, 2)
    #if positive_losses and len(positive_losses) == len(steps):
    plt.plot(steps, positive_losses, 'g-', label='Positive Loss', linewidth=2)
    #if negative_losses and len(negative_losses) == len(steps):
    plt.plot(steps, negative_losses, 'r-', label='Negative Loss', linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Positive vs Negative Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss图已保存到: {save_path}")
    else:
        plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python plot_loss.py <log_file_path> [save_path]")
        print("示例: python plot_loss.py models/RotatE_FB15k_0/train.log loss_plot.png")
        sys.exit(1)
    
    log_file = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(log_file):
        print(f"错误: 文件 {log_file} 不存在")
        sys.exit(1)
    
    plot_loss(log_file, save_path)