import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
train_line, = ax.plot([], [], label='Train Loss')
val_line, = ax.plot([], [], label='Val Loss')
epoch_line, = ax.plot([], [], 'ro-', label='Epoch Loss')  # 修改為折線
ax.set_xlabel('Sub-Epoch')
ax.set_ylabel('Loss')
ax.legend()

epoch_size = 0

def set_epoch_size(size):
    global epoch_size
    epoch_size = size
#%%
def update_plot(train_loss, val_loss, epoch_loss):
    train_line.set_data(range(len(train_loss)), train_loss)
    val_line.set_data(range(len(val_loss)), val_loss)
    
    epoch_x = [epoch_size + i * epoch_size for i in range(len(epoch_loss))]  # 從第44開始
    epoch = []
    for i in epoch_loss:
        train, val = i
        epoch.append((train * 0.8 + val * 0.2))
    epoch_line.set_data(epoch_x, epoch)
    
    if len(train_loss) > 0 or (epoch_x and len(epoch_x) > 0):
        ax.set_xlim(0, max(len(train_loss), epoch_x[-1] if epoch_x else 0))
    else:
        ax.set_xlim(-0.5, 0.5)  # 設置一個小的範圍以避免警告
    ax.set_ylim(0, max(max(train_loss), max(val_loss), max(epoch)) if train_loss and val_loss and epoch else 1)
    
    fig.canvas.draw()
    fig.canvas.flush_events()

async def plot_losses(train_loss, val_loss, epoch_loss):
    update_plot(train_loss, val_loss, epoch_loss)
    plt.pause(0.001)  # 暫停一小段時間以更新圖形
    
def init_plot():
    plt.ion()