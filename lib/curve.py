import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
train_line, = ax.plot([], [], label='Train Loss')
val_line, = ax.plot([], [], label='Val Loss')
ax.set_xlabel('Sub-Epoch')
ax.set_ylabel('Loss')
ax.legend()

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
    
def init_plot():
    plt.ion()
