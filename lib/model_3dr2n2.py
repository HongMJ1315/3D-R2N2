# %% 
import asyncio
import os
import threading
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader, Dataset, random_split
from lib.checkpoint import load_checkpoint, save_checkpoint
from lib.image import load_encoded_data, read_rendering_and_voxel
from lib.binvox import load_voxel_file
from lib.curve import *
from lib.config import *
from lib.gl import *
from lib.autoencoder import Autoencoder, CNNDecoder, CNNEncoder
from lib.lstm_decoder import LSTMDecoder, LSTM, CTNN3DDecoder

class TrainDataset(Dataset):
    def __init__(self, images, voxels):
        self.images = images
        self.voxels = voxels
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.voxels[idx]

class ClippedReLU(nn.Module):
    def __init__(self, min_value=0.0, max_value=1.0):
        super(ClippedReLU, self).__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        return torch.clamp(x, min=self.min_value, max=self.max_value)

class Model3DR2N2(nn.Module):
    def __init__(self):
        super(Model3DR2N2, self).__init__()
        self.encoder = CNNEncoder()
        self.lstm = LSTM()
        self.decoder = CTNN3DDecoder()
        self.clipped_relu = ClippedReLU()  # 使用自定義的激活函數

        
    def forward(self, x, prev_output, h_0, c_0):
        out = self.encoder(x)
        prev_out, h_0, c_0 = self.lstm(out, prev_output, h_0, c_0)
        out = self.decoder(prev_out)
        out = self.clipped_relu(out)
        return out, prev_out, h_0, c_0
    
# %% 
async def train_sub_epoch(epoch, model, datas, criterion, optimizer, device, train_loss, val_loss, epoch_loss):
    data_len = len(datas)
    train_data, val_data = random_split(datas, [int(data_len*0.8), data_len-int(data_len*0.8)])
    train_data_len = len(train_data)
    seg_train_data_len = int(train_data_len/100)
    print("Epoch:{} Train Size:{} Val Size:{}".format(epoch, len(train_data), len(val_data)))

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True)
    train_logs = []
    val_logs = []
    
    model.to(device)
    criterion.to(device)
    seq_len = 1
    model.train()
    start = time.time()
    timer = time.time()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)
        batch_size = inputs.size(0)
        
        h_0 = torch.zeros(model.lstm.num_layers, batch_size, model.lstm.hidden_size).to(device)
        c_0 = torch.zeros(model.lstm.num_layers, batch_size, model.lstm.hidden_size).to(device)    
        prev_output = None
        
        inputs = inputs.view(batch_size, seq_len, 5, 137, 137)
        image = inputs[:, 0, 0:4, :, :].view(4, 137, 137)
        edge = inputs[:, 0, 4, :, :].view(1, 137, 137)
        for t in range(seq_len):
            input = inputs[:, t, :, :, :]
            decode_output , prev_output, h_0, c_0 = model(input, prev_output, h_0, c_0)

        decode_output = decode_output.view(1, 32, 32, 32)
        
        decode_voxel = decode_output.cpu().detach().numpy()
        original_voxel = targets.cpu().detach().numpy()
        image = image.cpu().detach().numpy()
        edge = edge.cpu().detach().numpy()
        update_voxel(decode_voxel[0], original_voxel[0], image, edge)
        
        loss = criterion(decode_output, targets)
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
            h_0 = torch.zeros(model.lstm.num_layers, batch_size, model.lstm.hidden_size).to(device)
            c_0 = torch.zeros(model.lstm.num_layers, batch_size, model.lstm.hidden_size).to(device)    
            prev_output = None
            
            inputs = inputs.view(batch_size, seq_len, 5, 137, 137)
            image = inputs[:, 0, 0:4, :, :].view(4, 137, 137)
            edge = inputs[:, 0, 4, :, :].view(1, 137, 137)
            for t in range(seq_len):
                input = inputs[:, t, :, :, :]
                decode_output , prev_output, h_0, c_0 = model(input, prev_output, h_0, c_0)
            
            decode_output = decode_output.view(1, 32, 32, 32)
        
            decode_voxel = decode_output.cpu().detach().numpy()
            original_voxel = targets.cpu().detach().numpy()
            image = image.cpu().detach().numpy()
            edge = edge.cpu().detach().numpy()
            update_voxel(decode_voxel[0], original_voxel[0], image, edge)
            decode_output = decode_output.view(1, 32, 32, 32)
            loss = criterion(decode_output, targets)
            val_logs.append(loss.item())
    
    end = time.time()
    
    
    print("Epoch:{} Sub Train Time:{:.2f} Train Loss:{:.4f} Val Loss:{:.4f}".format(epoch, end - start,
                                                                                    sum(train_logs) / len(train_logs),
                                                                                    sum(val_logs) / len(val_logs)))
    train_loss.append(sum(train_logs) / len(train_logs))
    val_loss.append(sum(val_logs) / len(val_logs))
    
    await plot_losses(train_loss, val_loss, epoch_loss)
    
    return sum(train_logs) / len(train_logs), sum(val_logs) / len(val_logs)

# %% 
async def run_training(voxel_dataset_path, rendering_dataset_path, device, checkpoint_path):
    model = Model3DR2N2()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    early_stopping = EarlyStopping(patience=5, min_delta=0.0001)
    epoch_losses = []
    num_epochs = 100
    
    start_epoch, epoch_losses, last_folder, train_loss, val_loss = load_checkpoint(checkpoint_path, model, optimizer, device)
    await plot_losses(train_loss, val_loss, epoch_losses)

    model.to(device)
    model.train()
    
    for epoch in range(start_epoch, num_epochs):
        start = time.time()
        cnt = 0
        resume = (last_folder is not None)
        skip_cnt = 0
        renders = []
        voxels = []
        start_io = time.time()
        for root, dirs, files in os.walk(voxel_dataset_path):
            for file in files:
                file_name = os.path.join(root, file)
                folder = file_name.split('/')
                folder = folder[-3] + '/' + folder[-2]
                if(resume):
                    if(folder == last_folder):
                        print("Resume from {}, Skip {} Files".format(folder, skip_cnt))
                        resume = False
                        last_folder = None
                    skip_cnt += 1
                    continue
                if(file_name.split('.')[-1] != 'binvox'):
                    continue
                voxel, render = read_rendering_and_voxel(file_name, rendering_dataset_path)
                if(voxel is None or render is None or len(voxel) != len(render)):
                    continue
                for v in voxel:
                    voxels.append(v.astype(np.float32))
                for r in render:
                    renders.append(r)
                cnt += 1
                
                if(cnt >= DEFAULT_3DR2N2_TRAINING_IMAGE_AMOUNT):
                    end_io = time.time()
                    print(time.strftime("%H:%M:%S", time.localtime())) 
                    print('IO Time: {:.4f}'.format(end_io-start_io)) 
                    dataset = TrainDataset(renders, voxels)
                    await train_sub_epoch(epoch, model, dataset, criterion, optimizer, device, train_loss, val_loss, epoch_losses)
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
            await train_sub_epoch(epoch, model, dataset, criterion, optimizer, device, train_loss, val_loss, epoch_losses)
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
        
        await plot_losses(train_loss, val_loss, epoch_losses)
        
        if early_stopping(epoch_val_loss):
            print("Early Stopping at Epoch:{}".format(epoch))
            break
        
    return model, epoch_losses
# %% 
def train_3dr2n2(device, voxel_dataset_path = DEFAULT_BINVOX_DATASET_FOLDER,
                 rendering_dataset_path = DEFAULT_RENDERING_DATASET_FOLDER, 
                 checkpoint_path = DEFAULT_3DR2N2_FILE):
    global gl_task_queue
    print('Start Training 3D-R2N2, Device:{}'.format(device))
    print()
    init_plot()

    gl_task_queue = queue.Queue()  # 初始化任务队列
    
    gl_thread = threading.Thread(target=gl_main)
    gl_thread.start()
    
    result = asyncio.run(run_training(voxel_dataset_path, rendering_dataset_path, device, checkpoint_path))
    gl_thread.join()
    model, epoch_losses = result
    model.eval()
    torch.save(model.state_dict(), "model/lstmdecoder.pth")
    torch.save(model.lstm.state_dict(), "model/lstm.pth")
    torch.save(model.decoder.state_dict(), "model/ctnn3d.pth")
    return model, epoch_losses

