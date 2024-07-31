from lib.autoencoder import *
from lib.image import *
from lib.config import *
from lib.lstm_decoder import *
from lib.binvox import *
import torch

# print torch version
print(torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:{}'.format(device))
encode_image_dataset(device, model_file=DEFAULT_AUTOENCODER_FILE, image_folder=DEFAULT_RENDERING_DATASET_FOLDER, encoded_image_folder=DEFAULT_ENCODED_DATASET_FOLDER)
binvox_dataset(binvox_folder=DEFAULT_BINVOX_DATASET_FOLDER, output_folder=DEFAULT_VOXEL_DATASET_FOLDER)
train_lstmdecoder(device, dataset_path=DEFAULT_ENCODED_DATASET_FOLDER, checkpoint_path=DEFAULT_LSTMDECODER_FILE)
print('Finish')