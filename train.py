#train the LIIF model.
import torch
import torch.nn.functional as F
import LIIF
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import torch.optim as optim
import logging
from MyDataset import *
from tqdm import tqdm
def save_checkpoint(model,optimizer, scheduler,epoch,train_loss,val_loss,checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }, checkpoint_path)
def load_checkpoint(model,optimizer, scheduler,checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
if __name__=='__main__':
    #超参数
    batch_size = 16
    epochs = 100
    learning_rate = 2e-4
    name = 'liif'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #path
    data_dir = '/path/to/your/data'
    pth_path = '/path/to/save/checkpoints'
    os.makedirs(pth_path, exist_ok=True)
    checkpoint_path = os.path.join(pth_path, 'checkpoint.pth')
    #model
    hidden_list = [256, 256, 256, 256]
    model = LIIF.LIIF(1, hidden_list).to(device)
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.L1Loss().to(device)

    #dataset
    train_data = DataLoader(SR_dataset(data_dir, start_year=2000, end_year=2015), batch_size=batch_size, shuffle=True, num_workers=4)
    val_data = DataLoader(SR_dataset(data_dir, start_year=2016, end_year=2020), batch_size=batch_size, shuffle=False, num_workers=4)
    #logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(os.path.join(pth_path, f'{name}.log')),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    best_val_loss = float('inf')
    #train
    start_point = 0
    epoch_train_loss = []
    epoch_val_loss = []
    if os.path.exists(checkpoint_path):
        model,opt,start_epoch,epoch_train_loss,epoch_val_loss = load_checkpoint(model,opt,scheduler,checkpoint_path)
        logger.info(f"Resuming training from epoch {start_epoch}")
    else:
        print("No checkpoint found, starting training from scratch.")
    for epoch in tqdm(range(epochs), desc="Epochs", position=0, disable=False):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_data, desc="Training", disable=False):
            input = batch['input'].to(device)
            hr_coord = batch['hr_coord'].to(device)
            cell = batch['cell'].to(device)
            ground_truth = batch['ground_truth'].to(device)
            output = model(input, hr_coord, cell)
            loss = criterion(output, ground_truth)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_data)
        epoch_train_loss.append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_data, desc="Validation", disable=False):
                input = batch['input'].to(device)
                hr_coord = batch['hr_coord'].to(device)
                cell = batch['cell'].to(device)
                ground_truth = batch['ground_truth'].to(device)
                output = model(input, hr_coord, cell)
                loss = criterion(output, ground_truth)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_data)
        epoch_val_loss.append(avg_val_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Val Loss: {avg_val_loss:.4f}")
        save_checkpoint(model,opt,scheduler,epoch+1,epoch_train_loss,epoch_val_loss,checkpoint_path)
        torch.save(model.state_dict(), os.path.join(pth_path,f'model_epoch{epoch+1}.pth'))
        logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)

