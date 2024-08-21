import torch

DEFAULT_DEVICE = None
DEFAULT_AUTOENCODER_FILE = "autoencoder.pth.tar"
DEFAULT_LSTMDECODER_FILE = "lstmdecoder.pth.tar"
DEFAULT_CONVLSTMDECODER_FILE = "convlstmdecoder.pth.tar"
DEFAULT_3DR2N2_FILE = "3dr2n2_multiview.pth.tar"
DEFAULT_RENDERING_DATASET_FOLDER = "ShapeNetRendering"
DEFAULT_ENCODED_DATASET_FOLDER = "dataset-1"
DEFAULT_RESAULTS_IMAGE_FOLDER = "result"
DEFAULT_BINVOX_DATASET_FOLDER = "ShapeNetVox32"
DEFAULT_VOXEL_DATASET_FOLDER = "dataset-1"
PROCESS_IMAGE_AMOUNT = 30000
DEFAULT_AUTOENCODER_TRAINING_IMAGE_AMOUNT = 1000
DEFAULT_LSTMDECODER_TRAINING_IMAGE_AMOUNT = 1000
DEFAULT_3DR2N2_TRAINING_IMAGE_AMOUNT = 1000

def init():
    global DEFAULT_DEVICE
    DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", DEFAULT_DEVICE)

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

init()