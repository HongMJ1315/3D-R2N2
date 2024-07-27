# %%
import torch
import asyncio
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader, Dataset, random_split
from lib.checkpoint import load_checkpoint, save_checkpoint
from lib.config import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import os
import numpy as np
import json
import time


# %% 
ENCODED_TENSOR_SIZE = 3000

# 创建全局变量保存 figure 和 axes 对象
fig, ax = plt.subplots()
train_line, = ax.plot([], [], label='Train Loss')
val_line, = ax.plot([], [], label='Val Loss')
ax.set_xlabel('Sub-Epoch')
ax.set_ylabel('Loss')
ax.legend()
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
class CNNEecoder(nn.Module):
    def __init__(self):
        super(CNNEecoder, self).__init__()
        # Input Layer 1 
        self.cnn1 = nn.Conv2d(in_channels=5, out_channels=48, 
                              kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # out = 48*113*113
        self.dropout1 = nn.Dropout(p=0.25)  # Dropout layer after first max pooling

        
        # Hidden Layer 2
        self.cnn2 = nn.Conv2d(in_channels=48, out_channels=64,
                              kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # out = 64*56*56
        self.dropout2 = nn.Dropout(p=0.25)  # Dropout layer after second max pooling
        
        # Hidden Layer 3 
        self.cnn3 = nn.Conv2d(in_channels=64, out_channels=128,
                              kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        # out = 128*27*27
        self.dropout3 = nn.Dropout(p=0.25)
        
        # Output Layer 4
        # 轉為 1D
        self.fc1 = nn.Linear(32768, ENCODED_TENSOR_SIZE)

        
    def forward(self, x):
        # Input Layer 1
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.dropout1(out)
        
        # Hidden Layer 2
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = self.dropout2(out)
        
        # Output Layer 3
        out = self.cnn3(out)
        out = self.relu3(out)
        out = self.maxpool3(out)
        out = self.dropout3(out)

        # Output Layer 4
        out = out.view(1, -1)
        out = self.fc1(out)

                
        return out


# %%
class CNNDecoder(nn.Module):
    def __init__(self):
        super(CNNDecoder, self).__init__()
        # Input Layer 1
        self.fc1 = nn.Linear(ENCODED_TENSOR_SIZE, 128*28*28)
                
        # Hidden Layer 2
        self.cnn1 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                       kernel_size=3, stride=2, 
                                       padding=1, output_padding=1)
        self.relu1 = nn.ReLU()
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        # out = 64*112*112
        
        # Hidden Layer 3
        self.cnn2 = nn.ConvTranspose2d(in_channels=64, out_channels=48, 
                                       kernel_size=5, stride=2, 
                                       padding=2, output_padding=1)
        self.relu2 = nn.ReLU()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        # out = 48*224*224
        
        # Output Layer 4
        self.cnn3 = nn.ConvTranspose2d(in_channels=48, out_channels=5,
                                       kernel_size=5, stride=2,
                                       padding=2, output_padding=1)
        self.upsample3 = nn.Upsample(size=(137, 137), mode='nearest')
        
    def forward(self, x):
        # Input Layer 1
        out = self.fc1(x)
        
        # Hidden Layer 2
        out = out.view(out.size(0), 128, 28, 28)
        out = self.cnn1(out)
        out = self.relu1(out)
        out = self.upsample1(out)
        
        # Hidden Layer 3
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.upsample2(out)
        
        # Output Layer 4
        out = self.cnn3(out)
        out = self.upsample3(out)
        
        out = out.view(5, 137, 137)
        
        return out

# %%
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = CNNEecoder()
        self.decoder = CNNDecoder()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# %%
def image_preprocessing(images):
    datas = []
    for img in images:
        # 取得圖像的邊緣
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(gray, 70, 210,apertureSize=3)

        img = cv2.merge((img[:,:,0], img[:,:,1], img[:,:,2], img[:,:,3], edge))
        
        img = img.transpose(2, 0, 1)
        datas.append(img)
    datas = np.array(datas, dtype=np.float32) / 255
    datas = torch.from_numpy(datas)
    return datas

#%%
def update_plot(train_loss, val_loss):
    train_line.set_data(range(len(train_loss)), train_loss)
    val_line.set_data(range(len(val_loss)), val_loss)
    ax.set_xlim(0, len(train_loss))
    ax.set_ylim(0, max(max(train_loss), max(val_loss)) if train_loss and val_loss else 1)
    fig.canvas.draw()
    fig.canvas.flush_events()

async def plot_losses(train_loss, val_loss):
    update_plot(train_loss, val_loss)
    plt.pause(0.001)  # 暫停一小段時間以更新圖形
#%%

# 更新 train_sub_epoch 函數中的 await plot_losses
async def train_sub_epoch(epoch, datas, model, criterion, optimizer, device, train_loss, val_loss):
    train, val = train_test_split(datas, test_size=0.2)
    print("Epoch:{} Train Size:{} Val Size:{}".format(epoch, len(train), len(val)))
    train = train.to(device)
    val = val.to(device)
    criterion = criterion.to(device)
    model.to(device)
    start = time.time()
    train_logs = []
    val_logs = []
    model.train()
    for data in train:
        model.zero_grad()
        img = data
        recon = model(img)
        loss = criterion(recon, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_logs.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_loss_value = 0
        for data in val:
            img = data
            recon = model(img)
            loss = criterion(recon, img)
            val_loss_value += loss.item()
        val_logs.append(val_loss_value/len(val))
    end = time.time()

    print("Epoch:{} Sub Train Time:{:.2f} Train Loss:{:.4f} Val Loss:{:.4f}".format(epoch, end-start, sum(train_logs)/len(train_logs), sum(val_logs)/len(val_logs)))
    
    # 更新總損失列表
    train_loss.append(sum(train_logs)/len(train_logs))
    val_loss.append(sum(val_logs)/len(val_logs))
    
    # 繪製損失曲線
    await plot_losses(train_loss, val_loss)
    
    return sum(train_logs)/len(train_logs), sum(val_logs)/len(val_logs)

#%%
async def run_training(file_path, device, checkpoint_path):
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 100
    early_stopping = EarlyStopping(patience=5, min_delta=0.0001)
    epoch_losses = []

    start_epoch, epoch_losses, last_file, train_loss, val_loss = load_checkpoint(checkpoint_path, model, optimizer, device)
    
    print('Start Training')
    for epoch in range(start_epoch, num_epochs):
        start = time.time()
        cnt = 0
        datas = []
        resume = (last_file is not None)
        skip_cnt = 0
        for root, dirs, files in os.walk(file_path):
            start_io = time.time()
            for file in files:
                if resume:
                    if last_file == os.path.join(root, file):
                        print('Resuming from {}, Skip {} Files'.format(last_file, skip_cnt))
                        resume = False
                    skip_cnt += 1
                    continue

                file_name = os.path.join(root, file)
                if(file_name.split('.')[-1] != 'png'):
                    continue
                cnt += 1
                img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
                if(img is None):
                    print("Error:{}".format(file_name))
                    continue
                datas.append(img)
                if(cnt >= 100):
                    end_io = time.time()
                    print(time.strftime("%H:%M:%S", time.localtime())) 
                    print('IO Time: {:.4f}'.format(end_io-start_io))
                    datas = image_preprocessing(datas)
                    await train_sub_epoch(epoch, datas, model, criterion, optimizer, device, train_loss, val_loss)
                    datas = []
                    cnt = 0
                    start_io = time.time()
                    save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch_losses': epoch_losses,
                        'last_file': file_name,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                    }, filename=checkpoint_path)
        if(len(datas) > 0):
            end_io = time.time()
            print('IO Time: {:.4f}'.format(end_io-start_io))
            datas = image_preprocessing(datas)
            await train_sub_epoch(epoch, datas, model, criterion, optimizer, device, train_loss, val_loss)
            datas = []
            cnt = 0
        end = time.time()
        epoch_train_loss = sum(train_loss)/len(train_loss)
        epoch_val_loss = sum(val_loss)/len(val_loss)
        print('Epoch [{}/{}], Train Loss:{:.4f}, Val Loss:{:.4f}, time: {:.4f}'.format(epoch+1, num_epochs, epoch_train_loss, epoch_val_loss, end-start))
        epoch_losses.append((epoch, epoch_train_loss, epoch_val_loss))

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_log': epoch_losses,
            'last_file': None,
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, filename=checkpoint_path)

        if early_stopping(epoch_val_loss):
            print('Early Stopping')
            break
    return model, epoch_losses


# %%
def train_autoencoder(device, dataset_path = DEFAULT_RENDERING_DATASET_FOLDER, checkpoint_path = DEFAULT_AUTOENCODER_FILE):
    print('Train Autoencoder, Device:{}'.format(device))
    print()
    plt.ion()
    result = asyncio.run(run_training(dataset_path, device, checkpoint_path))
    
    model, epoch_losses = result
    model.eval()
    
    torch.save(model.state_dict(), 'model.pth')
    torch.save(model.encoder.state_dict(), 'encoder.pth')
    torch.save(model.decoder.state_dict(), 'decoder.pth')


    return model, epoch_losses
    
            

# %%
def test_autoencoder(device = DEFAULT_DEVICE, dataset_path = DEFAULT_RENDERING_DATASET_FOLDER, checkpoint_path = DEFAULT_AUTOENCODER_FILE, result_path = DEFAULT_RESAULTS_IMAGE_FOLDER, save_result = True):
    print('Test Autoencoder, Device:{}'.format(device))
    datas = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            file_name = os.path.join(root, file)
            
            if(file_name.split('.')[-1] != 'png'):
                continue
            img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)   
            # resize 
            img = cv2.resize(img, (137, 137))
            
             # 取得圖像的邊緣
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edge = cv2.Canny(blurred, 70, 210,apertureSize=3)
            
            img = cv2.merge((img[:,:,0], img[:,:,1], img[:,:,2], img[:,:,3], edge))
            
            img = img.transpose(2, 0, 1)
            datas.append(img)
    # '''
    datas = np.array(datas, dtype=np.float32)
    datas = datas / 255
    # load model
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    _, _, _, train_loss, val_loss = load_checkpoint(checkpoint_path, model, optimizer, device)
    model.to(device)

    model.eval()
    for img in datas:
        with torch.no_grad():
            timg = torch.from_numpy(img)
            timg = timg.to(device)
            output = model(timg)
            output = output.cpu()
            output = output.numpy()
            output = output.transpose(1, 2, 0)
            img = img.transpose(1, 2, 0)
            # 將超過0~1的值設為1
            image = output[:, :, 0:4]
            image = np.clip(image, 0, 1)
            alpha = output[:, :, 3]
            alpha = np.clip(alpha, 0, 1)
            edge = output[:, :, 4]
            edge = np.clip(edge, 0, 1)
            fig, axes = plt.subplots(1, 4, figsize=(10, 5))
            axes[0].set_title('input')
            axes[0].imshow(img[:, :, 0:4])
            axes[1].set_title('reconstruct')
            axes[1].imshow(image)
            axes[2].set_title('alpha')
            axes[2].imshow(alpha, cmap='gray')
            axes[3].set_title('edge')
            axes[3].imshow(edge, cmap='gray')
            if(save_result):
                plt.savefig(result_path+'/'+str(time.time())+'.png')
            plt.show()

def show_loss_curve(checkpoint_path = DEFAULT_AUTOENCODER_FILE, device = DEFAULT_DEVICE,save_result = True):
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    _, _, _, train_loss, val_loss = load_checkpoint(checkpoint_path, model, optimizer, device)
    plt.plot(train_loss, label='train loss')
    plt.plot(val_loss, label='val loss')
    plt.legend()
    plt.show()
    if(save_result):
        plt.savefig('loss_curve.png')
# %%
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:{}'.format(device))
    # model, train_log = train_autoencoder(device)
    test_autoencoder(device, dataset_path="test", checkpoint_path="autoencoder.pth.tar", result_path="results", save_result=True)
    # encode_image_dataset(device, model_file='model3/checkpoint.pth.tar', image_folder='ShapeNetRendering', encoded_image_folder='dataset')
    print('Finish')



# %%
# load_encoded_data('dataset/1a2b1863733c2ca65e26ee427f1e5a4c').shape



