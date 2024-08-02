# %%
import asyncio
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
from lib.curve import *
from lib.config import *
import lib.autoencoder as ae
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

import time

# %%
LSTM_NEUROES = ae.ENCODED_TENSOR_SIZE


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
class ConvLSTMCell(nn.Module):
    def __init__(self, input_len, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_len = input_len
        self.fc = nn.Linear(input_len, input_dim * kernel_size[0] * kernel_size[1] * kernel_size[2])
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding_size = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
        self.bias = bias

        self.conv_f = nn.Conv3d(in_channels=self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size, 
                                padding=self.padding_size, 
                                bias=self.bias)
        self.conv_i = nn.Conv3d(in_channels=self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size, 
                                padding=self.padding_size, 
                                bias=self.bias)
        self.conv_g = nn.Conv3d(in_channels=self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size, 
                                padding=self.padding_size,
                                bias=self.bias)
        self.conv_o = nn.Conv3d(in_channels=self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size, 
                                padding=self.padding_size,
                                bias=self.bias)
        
        
        if any(k % 2 == 0 for k in kernel_size):
            raise ValueError("Only support odd kernel size")
        self.conv = nn.Conv3d(self.input_dim + self.hidden_dim, 
                              4 * self.hidden_dim,  # 4* 是因为后面输出时要切4片
                              self.kernel_size, 
                              padding=self.padding_size, 
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        # 经过全连接层
        input_tensor = input_tensor.view(-1, self.input_len)
        input_fc = self.fc(input_tensor)
        
        # 变形
        batch_size = input_tensor.size(0)
        depth, height, width = h_cur.size(2), h_cur.size(3), h_cur.size(4)
        input_fc = input_fc.view(batch_size, self.input_dim, depth, height, width)
        combined = torch.cat((input_fc, h_cur), dim=1)
        combined_conv = self.conv(combined)
        cc_f, cc_i, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        cc_f = self.conv_f(cc_f)
        cc_i = self.conv_i(cc_i)
        cc_o = self.conv_o(cc_o)
        cc_g = self.conv_g(cc_g)
        
        f = torch.sigmoid(cc_f)
        i = torch.sigmoid(cc_i)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        depth, height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, depth, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, depth, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    def __init__(self, input_len, input_dim, hidden_dim, kernel_size, num_layers, image_size, bias=False):
        super(ConvLSTM, self).__init__()

        self.input_len = input_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.image_size = image_size  # 添加 image_size
        self.bias = bias

        cell_list = []
        for i in range(0, num_layers):
            cur_input_dim = input_dim if i == 0 else self.hidden_dim[i-1]
            cell_list.append(ConvLSTMCell(input_len=self.input_len,
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size,
                                          bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state):
        cur_layer_input = input_tensor
        new_hidden_state = []

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            # print(f"Layer {layer_idx} - Hidden state h: {h.shape}, c: {c.shape}")
            h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input, cur_state=[h, c])
            # print(f"Layer {layer_idx} - Output h: {h.shape}, c: {c.shape}")
            cur_layer_input = h
            new_hidden_state.append([h, c])

        return cur_layer_input, new_hidden_state

    def init_hidden(self, batch_size):
        return [cell.init_hidden(batch_size, self.image_size) for cell in self.cell_list]
        



# %%
class CTNN3DDecoder(nn.Module):
    def __init__(self, input_dim):
        super(CTNN3DDecoder, self).__init__()
        
        self.conv1 = nn.ConvTranspose3d(in_channels=input_dim, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dropout1 = nn.Dropout3d(p=0.25)
        
        self.conv2 = nn.ConvTranspose3d(in_channels=12, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dropout2 = nn.Dropout3d(p=0.25)
        
        self.conv3 = nn.ConvTranspose3d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dropout3 = nn.Dropout3d(p=0.25)
        
        self.conv4 = nn.ConvTranspose3d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.upsample4 = nn.Upsample(size=(32, 32, 32), mode='nearest')
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.upsample1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.upsample2(out)
        out = self.dropout2(out)
        
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.upsample3(out)
        out = self.dropout3(out)
        
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.upsample4(out)
        return out

# model = CTNN3DDecoder()
# test = torch.randn(LSTM_NEUROES)
# model(test)
# %%
class ConvLSTMDecoder(nn.Module):
    def __init__(self, input_len, input_dim, hidden_dim, kernel_size, num_layers, image_size, bias=False):
        super(ConvLSTMDecoder, self).__init__()
        self.lstm = ConvLSTM(input_len, input_dim, hidden_dim, kernel_size, num_layers, image_size, bias)
        self.decoder = CTNN3DDecoder(hidden_dim[0])
        
    def forward(self, x, hidden_state):
        out, hidden_state = self.lstm(x, hidden_state)
        out = self.decoder(out)
        return out, hidden_state
    

# %%
class TrainDataset(Dataset):
    def __init__(self, datas, voxel):
        self.datas = datas
        self.voxel = voxel

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        return self.datas[index], self.voxel[index].astype(np.float32)

# %%
async def train_sub_epoch(epoch, datas, model, criterion, optimizer, device, train_loss, val_loss):
    data_len = len(datas)
    train_data, val_data = random_split(datas, [int(data_len * 0.8), data_len - int(data_len * 0.8)])
    train_data_len = len(train_data)
    seg_train_data_len = round(train_data_len / 100)
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
    timer = time.time()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
        batch_size = inputs.size(0)
        hidden_state = model.lstm.init_hidden(batch_size)

        inputs = inputs.view(batch_size, seq_len, -1)
        for t in range(seq_len):
            input_t = inputs[:, t]
            decode_output, hidden_state = model(input_t, hidden_state)
        
        test_chennal = decode_output[:, 0]
        loss = criterion(test_chennal , targets)
        loss.backward()
        optimizer.step()
        train_logs.append(loss.item())
        if(len(train_logs) % seg_train_data_len == 0):
            print("Epoch:{} Train Loss:{:.4f} Sub-epoch: {}% Time: {}".format(epoch, sum(train_logs) / len(train_logs), len(train_logs) / seg_train_data_len, time.time() - timer))
            timer = time.time()
            
    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size = inputs.size(0)
            hidden_state = model.lstm.init_hidden(batch_size)


            inputs = inputs.view(batch_size, seq_len, -1)
            for t in range(seq_len):
                input_t = inputs[:, t]
                decode_output, hidden_state = model(input_t, hidden_state)
        
            test_chennal = decode_output[:, 0]
            loss = criterion(test_chennal , targets)
            val_logs.append(loss.item())
            
    end = time.time()

    print("Epoch:{} Sub Train Time:{:.2f} Train Loss:{:.4f} Val Loss:{:.4f}".format(epoch, end - start,
                                                                                    sum(train_logs) / len(train_logs),
                                                                                    sum(val_logs) / len(val_logs)))
    train_loss.append(sum(train_logs) / len(train_logs))
    val_loss.append(sum(val_logs) / len(val_logs))
    
    await plot_losses(train_loss, val_loss)
    
    return sum(train_logs) / len(train_logs), sum(val_logs) / len(val_logs)



# %%
async def run_training(file_path, device, checkpoint_path):
    hidden_dim = [16]
    kernel_size = (15, 15, 15)
    num_layers = 1
    image_size = (15, 15, 15)  # 这里假设卷积核大小为3x3x3
    
    if(file_path[-1] != "/"):
        file_path += "/"
    
    model = ConvLSTMDecoder(input_len=LSTM_NEUROES, input_dim=16, hidden_dim=hidden_dim, kernel_size=kernel_size, num_layers=num_layers, image_size=image_size)
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
        resume = (last_folder is not None)
        skip_cnt = 0
        renders = []
        voxels = []
        for root, dirs, files in os.walk(file_path):
            folder = root.split("/")[-1]
            if(resume):
                if(folder == last_folder):
                    print("Resume from {}, Skip {} Files".format(folder, skip_cnt))
                    resume = False
                    last_folder = None
                skip_cnt += 1
                continue
            start_io = time.time()
            if(folder == ""): continue
            render = load_encoded_data(root)
            voxel = load_voxel_file(root + "/voxel.txt")
            voxel.reshape(32, 32, 32)
            if render is None or voxel is None:
                continue             
            for i in render:
                renders.append(i)
                voxels.append(voxel)
            cnt += 1
            # print("folder: {}, render: {}, renders: {}, cnt: {}".format(folder, len(render), len(renders), cnt))    

            if(cnt >= DEFAULT_LSTMDECODER_TRAINING_IMAGE_AMOUNT):
                end_io = time.time()
                print(time.strftime("%H:%M:%S", time.localtime())) 
                print("IO Time:{:.4f}".format(end_io-start_io))
                dataset = TrainDataset(renders, voxels)
                await train_sub_epoch(epoch, dataset, model, criterion, optimizer, device, train_loss, val_loss)
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
            await train_sub_epoch(epoch, dataset, model, criterion, optimizer, device, train_loss, val_loss)
            renders = []
            voxels = []
            cnt = 0
        end = time.time()
        epoch_train_loss = sum(train_loss)/len(train_loss)
        epoch_val_loss = sum(val_loss)/len(val_loss)
        print("Epoch:{} Time:{:.2f} Train Loss:{:.4f} Val Loss:{:.4f}".format(epoch, end-start, epoch_train_loss, epoch_val_loss))
        epoch_losses.append((epoch_train_loss, epoch_val_loss))
        
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch_losses': epoch_losses,
            'last_file': None,
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, filename = checkpoint_path)
        
        if early_stopping(epoch_val_loss):
            print("Early Stopping at Epoch:{}".format(epoch))
            break
    return model, epoch_losses

# %%
def train_lstmdecoder(device, dataset_path = DEFAULT_ENCODED_DATASET_FOLDER, checkpoint_path = DEFAULT_LSTMDECODER_FILE):
    print("Start Training LSTMDecoder, Device:{}".format(device))
    print()
    init_plot()
    
    result = asyncio.run(run_training(dataset_path, device, checkpoint_path))
    model, epoch_losses = result
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
    

'''
model = LSTM()
batch_size = 1

# 初始化隐藏层状态
h_0 = torch.zeros(model.num_layers, batch_size, model.hidden_size)
c_0 = torch.zeros(model.num_layers, batch_size, model.hidden_size)

# 第一次输入
input_1 = torch.randn(batch_size, 1, LSTM_NEUROES)  # (batch_size, seq_len, input_size)
print(input_1.shape)
output, h_n, c_n = model(input_1, None, h_0, c_0)
print("第一次输出:", output)

# 第二次输入，使用第一次的输出作为一部分输入
input_2 = torch.randn(batch_size, 1, LSTM_NEUROES)  # 另外的输入
new_input = torch.cat((input_2, output.unsqueeze(1)), dim=2)  # 合并输出作为输入的一部分
new_input = model.input_fc(new_input)  # 映射到原始input_size

output, h_n, c_n = model(new_input, h_n, c_n)
print("第二次输出:", output)

# 多次迭代输入进行修正
num_iterations = 5
for i in range(num_iterations):
    input_next = torch.randn(batch_size, 1, LSTM_NEUROES)  # 另外的输入
    new_input = torch.cat((input_next, output.unsqueeze(1)), dim=2)  # 合并输出作为输入的一部分
    new_input = model.input_fc(new_input)  # 映射到原始input_size

    output, h_n, c_n = model(new_input, h_n, c_n)
    print(f"第{i+3}次输出:", output)
# '''

