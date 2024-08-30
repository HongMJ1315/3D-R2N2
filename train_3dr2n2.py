from lib.image import *
from lib.config import *
from lib.binvox import *
from lib.model_3dr2n2 import *
import torch

# print torch version
print(torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:{}'.format(device))
train_3dr2n2(device, model_type='Transformer')
print('Finish')