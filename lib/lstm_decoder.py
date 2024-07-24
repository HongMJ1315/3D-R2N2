# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader, Dataset, random_split
from lib.checkpoint import load_checkpoint, save_checkpoint
from lib.image import load_encoded_data
from lib.binvox import load_voxel_file
from lib.config import *
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

import time

# %%
LSTM_NEUROES = 13*13*13


# %%
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False


# %%
class LSTM(nn.Module):
    def __init__(self, input_size=LSTM_NEUROES, hidden_size=LSTM_NEUROES, num_layers=2, output_size=LSTM_NEUROES):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.input_fc = nn.Linear(input_size + output_size, input_size)  # 映射到 input_size

    def forward(self, input_t, prev_output, h_0, c_0):
        if prev_output is not None:
            input_t = torch.cat((input_t, prev_output), dim=1)
            input_t = self.input_fc(input_t)  # 映射到原始input_size

        input_t = input_t.unsqueeze(1)  # 调整形状为 (batch_size, seq_len, input_size)
        out, (h_0, c_0) = self.lstm(input_t, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out, h_0, c_0
        



# %%
class CTNN3DDecoder(nn.Module):
    def __init__(self):
        super(CTNN3DDecoder, self).__init__()
        self.fc = nn.Linear(LSTM_NEUROES, 8*8*8*16)
        
        self.cov1 = nn.ConvTranspose3d(in_channels = 16, out_channels = 14,
                                kernel_size = 3, stride = 2, padding = 2)
        self.cov2 = nn.ConvTranspose3d(in_channels = 14, out_channels = 12,
                                kernel_size = 3, stride = 2, padding = 2)
        self.relu1 = nn.ReLU()
        
        self.cov3 = nn.ConvTranspose3d(in_channels = 12, out_channels = 10,
                                kernel_size = 3, stride = 2, padding = 2)
        self.cov4 = nn.ConvTranspose3d(in_channels = 10, out_channels = 8,
                                kernel_size = 3, stride = 2, padding = 2)
        self.relu2 = nn.ReLU()
                
        self.cov5 = nn.ConvTranspose3d(in_channels = 8, out_channels = 6,
                                kernel_size = 3, stride = 2, padding = 2)
        self.cov6 = nn.ConvTranspose3d(in_channels = 6, out_channels = 4,
                                kernel_size = 3, stride = 2, padding = 2)
        self.relu3 = nn.ReLU()
        self.upsample = nn.Upsample(size=(32, 32, 32), mode='nearest')        
        
        # self.cov4 = nn.Conv3d(in_channels = 40, out_channel = 1, )
        
    def forward(self, x):
        out = self.fc(x)
        out = out.view(1, 16, 8, 8, 8)
        
        out = self.cov1(out)
        out = self.cov2(out)
        out = self.relu1(out)

        out = self.cov3(out)
        out = self.cov4(out)
        out = self.relu2(out)
        
        out = self.cov5(out)
        out = self.cov6(out)
        out = self.relu3(out)
        out = self.upsample(out)
        return out
        
# model = CTNN3DDecoder()
# test = torch.randn(LSTM_NEUROES)
# model(test)


# %%
class LSTMDecoder(nn.Module):
    def __init__(self):
        super(LSTMDecoder, self).__init__()
        self.lstm = LSTM()
        self.decoder = CTNN3DDecoder()
        
    def forward(self, x, prev_output, h_0, c_0):
        out, h_0, c_0 = self.lstm(x, prev_output, h_0, c_0)
        out = self.decoder(out)
        return out, h_0, c_0


# %%
class TrainDataset(Dataset):
    def __init__(self, datas, voxel):
        self.datas = datas
        self.voxel = voxel

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        return self.datas[index], self.voxel[index]

# %%
def train_sub_epoch(epoch, datas, model, criterion, optimizer, device):
    data_len = len(datas)
    train_data, val_data = random_split(datas, [int(data_len * 0.8), data_len - int(data_len * 0.8)])
    print("Epoch:{} Train Size:{} Val Size:{}".format(epoch, len(train_data), len(val_data)))

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True)
    train_logs = []
    val_logs = []
    # move data and model to device
    model.to(device)
    criterion.to(device)
    seq_len = 1
    start = time.time()
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
        batch_size = inputs.size(0)
        h_0 = torch.zeros(model.lstm.num_layers, batch_size, model.lstm.hidden_size).to(inputs.device)
        c_0 = torch.zeros(model.lstm.num_layers, batch_size, model.lstm.hidden_size).to(inputs.device)
        prev_output = None

        inputs = inputs.view(batch_size, seq_len, -1)
        for t in range(seq_len):
            input_t = inputs[:, t]
            output, h_0, c_0 = model(input_t, prev_output, h_0, c_0)
            prev_output = output

        loss = criterion(prev_output, targets)
        loss.backward()
        optimizer.step()
        train_logs.append(loss.item())

    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            batch_size = inputs.size(0)
            h_0 = torch.zeros(model.lstm.num_layers, batch_size, model.lstm.hidden_size).to(inputs.device)
            c_0 = torch.zeros(model.lstm.num_layers, batch_size, model.lstm.hidden_size).to(inputs.device)
            prev_output = None

            inputs = inputs.view(batch_size, seq_len, -1)
            for t in range(seq_len):
                input_t = input_t.view(batch_size, -1)
                output, h_0, c_0 = model(input_t, prev_output, h_0, c_0)
                prev_output = output

            loss = criterion(output, targets)
            val_logs.append(loss.item())
    end = time.time()

    print("Epoch:{} Sub Train Time:{:.2f} Train Loss:{:.4f} Val Loss:{:.4f}".format(epoch, end - start,
                                                                                    sum(train_logs) / len(train_logs),
                                                                                    sum(val_logs) / len(val_logs)))
    return sum(train_logs) / len(train_logs), sum(val_logs) / len(val_logs)


# %%
def run_training(file_path, device, checkpoint_path):
    model = LSTMDecoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    early_stopping = EarlyStopping(patience=5, min_delta=0.0001)
    epoch_losses = []
    num_epochs = 20
    
    start_epoch, epoch_losses, last_folder, train_loss, val_loss = load_checkpoint(checkpoint_path, model, optimizer, device)

    model.train()
    for epoch in range(start_epoch, num_epochs):
        start = time.time()
        cnt = 0
        train_loss = []
        val_loss = []
        resume = (last_folder is not None)
        skip_cnt = 0
        renders = []
        voxels = []
        for root, dirs, files in os.walk("dataset/"):
            folder = root.split("/")[-1]
            if(resume):
                if(folder == last_folder):
                    print("Resume from {}, Skip {} Files".format(folder, skip_cnt))
                    resume = False
                else:
                    skip_cnt += 1
                    continue
            start_io = time.time()
            if(folder == ""): continue
            print(root)
            render = load_encoded_data(root)
            voxel = load_voxel_file(root + "/voxel.txt")
            if render is None or voxel is None:
                continue             
            for i in render:
                renders.append(i)
                voxels.append(voxel)
            cnt += 1
            if(cnt >= 10):
                end_io = time.time()
                print("IO Time:{:.2f}".format(end_io-start_io))
                dataset = TrainDataset(renders, voxels)
                train, val = train_sub_epoch(epoch, dataset, model, criterion, optimizer, device)
                train_loss.append(train)
                val_loss.append(val)
                renders = []
                voxels = []
                cnt = 0
                start_io = time.time()
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch_losses': epoch_losses,
                    'last_file': folder,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, filename = checkpoint_path)
        if(cnt > 0):
            dataset = TrainDataset(renders, voxels)
            train, val = train_sub_epoch(epoch, dataset, model, criterion, optimizer, device)
            train_loss.append(train)
            val_loss.append(val)
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch_losses': epoch_losses,
                'last_file': None,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, filename = checkpoint_path)
        end = time.time()
        epoch_train_loss = sum(train_loss)/len(train_loss)
        epoch_val_loss = sum(val_loss)/len(val_loss)
        epoch_losses.append((epoch_train_loss, epoch_val_loss))
        print("Epoch:{} Time:{:.2f} Train Loss:{:.4f} Val Loss:{:.4f}".format(epoch, end-start, epoch_train_loss, epoch_val_loss))
        if early_stopping(epoch_val_loss):
            print("Early Stopping at Epoch:{}".format(epoch))
            break
    return model, epoch_losses

# %%
def train_lstmdecoder(device, dataset_path = DEFAULT_ENCODED_DATASET_FOLDER, checkpoint_path = DEFAULT_LSTMDECODER_FILE):
    model, epoch_losses = run_training(dataset_path, device, checkpoint_path)
    model.eval()
    torch.save(model.state_dict(), "model/lstmdecoder.pth")
    torch.save(model.lstm.state_dict(), "model/lstm.pth")
    torch.save(model.decoder.state_dict(), "model/ctnn3d.pth")
    return model, epoch_losses

# %%
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, epoch_losses = train_lstmdecoder(device)
    print(epoch_losses)
    