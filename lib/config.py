import torch

DEFAULT_DEVICE = None
DEFAULT_AUTOENCODER_FILE = "autoencoder.pth.tar"
DEFAULT_LSTMDECODER_FILE = "lstmdecoder.pth.tar"
DEFAULT_RENDERING_DATASET_FOLDER = "ShapeNetRendering"
DEFAULT_ENCODED_DATASET_FOLDER = "dataset"
DEFAULT_RESAULTS_IMAGE_FOLDER = "result"

def init():
    global DEFAULT_DEVICE
    DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", DEFAULT_DEVICE)

init()