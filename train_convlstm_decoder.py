from lib.autoencoder import *
from lib.image import *
from lib.config import *
from lib.convlstm_decoder import *
from lib.binvox import *
import torch


# print torch version
print(torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:{}'.format(device))
train_lstmdecoder(device, dataset_path=DEFAULT_ENCODED_DATASET_FOLDER, checkpoint_path=DEFAULT_CONVLSTMDECODER_FILE)
print('Finish')