import torch

DEFAULT_DEVICE = None
DEFAULT_AUTOENCODER_FILE = "autoencoder.pth.tar"
DEFAULT_LSTMDECODER_FILE = "lstmdecoder.pth.tar"
DEFAULT_CONVLSTMDECODER_FILE = "convlstmdecoder.pth.tar"
DEFAULT_3DR2N2_FILE = "3dr2n2_LSTM_Sigmoid_DiceLoss.pth.tar"
DEFAULT_RENDERING_DATASET_FOLDER = "render"
DEFAULT_ENCODED_DATASET_FOLDER = "dataset-1"
DEFAULT_RESAULTS_IMAGE_FOLDER = "result"
DEFAULT_BINVOX_DATASET_FOLDER = "voxel"
DEFAULT_VOXEL_DATASET_FOLDER = "dataset-1"
PROCESS_IMAGE_AMOUNT = 30000
DEFAULT_AUTOENCODER_TRAINING_IMAGE_AMOUNT = 1000
DEFAULT_LSTMDECODER_TRAINING_IMAGE_AMOUNT = 1000
DEFAULT_3DR2N2_TRAINING_IMAGE_AMOUNT = 1000
DEFAULT_3DR2N2_TRAINING_MULTIVIEW_AMOUNT = 25
DEFAULT_3DR2N2_TRAINING_WITH_MULTIVIEW = True

GL_FONTS = "fonts/Consolas.ttf"
GL_FONTS_SIZE = 24
GL_VISUALIZE_THRESHOLD = 0.1

def init():
    global DEFAULT_DEVICE
    DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", DEFAULT_DEVICE)

# %%
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0001, path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, model):
        '''当验证集损失降低时，保存模型。'''
        torch.save(model.state_dict(), self.path)
        print(f'Model saved to {self.path}')

init()