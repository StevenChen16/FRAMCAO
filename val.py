import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from train import generate_event_data, EnhancedCombinedLoss
from data import combine_stock_data
from sklearn.model_selection import train_test_split
from fusion_model import FusionStockDataset, FRAMCAOCNNFusion
from model import prepare_feature_groups
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def validate_model(model, val_loader, criterion, writer, epoch, device):
    """Helper function for model validation"""
    model.eval()
    val_loss = 0
    val_metrics = {'mse': 0, 'direction': 0, 'smoothness': 0,
                  'mcao_features': 0, 'group_consistency': 0}
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            sequence = batch['sequence'].to(device)
            events = batch['events'].to(device)
            time_distances = batch['time_distances'].to(device)
            target = batch['target'].to(device)
            current_price = batch['current_price'].to(device)
            
            predictions, mcao_features, group_features = model(
                sequence, events, time_distances
            )
            
            loss, metrics = criterion(
                predictions,
                target,
                current_price,
                mcao_features,
                group_features
            )
            
            if not torch.isnan(loss):
                val_loss += loss.item()
                for k, v in metrics.items():
                    if not torch.isnan(torch.tensor(v)):
                        val_metrics[k] += v
    
    val_loss /= len(val_loader)
    writer.add_scalar('Validation/Loss', val_loss, epoch)
    
    print(f"\nValidation Loss: {val_loss:.4f}")
    for k, v in val_metrics.items():
        v_avg = v/len(val_loader)
        if not np.isnan(v_avg):
            print(f"Val {k}: {v_avg:.4f}")
            writer.add_scalar(f'Validation/{k}', v_avg, epoch)
    
    return val_loss

if __name__ == "__main__":
    symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM", 
        "^GSPC", "^NDX", "^DJI", "^IXIC",
        "UNH", "ABBV", "LLY",
        "FANG", "DLR", "PSA", "BABA", "JD", "BIDU",
        "QQQ"
    ]
    data = combine_stock_data(symbols, '2020-01-01', '2024-01-01')
    events = generate_event_data(data)
    
    # 划分训练集和验证集
    train_data, val_data = train_test_split(data, test_size=0.2, shuffle=False)
    train_events, val_events = train_test_split(events, test_size=0.2, shuffle=False)
    
    # 创建数据集
    train_dataset = FusionStockDataset(train_data, train_events)
    val_dataset = FusionStockDataset(val_data, val_events)
    
    # 创建数据加载器
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    # 参数设置
    input_dim = 21  # 输入特征维度
    hidden_dim = 128  # 隐藏层维度
    event_dim = 32  # 事件嵌入维度
    num_event_types = 10  # 事件类型数量
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载模型
    print("Loading model...")
    model = model = FRAMCAOCNNFusion(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        event_dim=event_dim,
        num_event_types=num_event_types,
        feature_groups=prepare_feature_groups()
    ).to(device)

    checkpoint = torch.load('checkpoints/fusion/stage1_best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    criterion = EnhancedCombinedLoss()
    writer = SummaryWriter('runs/fusion_model')
    epoch=1

    print("Starting validation...")
    val_loss = validate_model(model, val_loader, criterion, writer, epoch, device)

    print("Validation completed successfully.")
    print(f"Validation Loss: {val_loss:.4f}")