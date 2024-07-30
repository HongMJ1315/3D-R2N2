from lib.autoencoder import *
import torch

# print torch version
print(torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:{}'.format(device))
model, train_log = train_autoencoder(device)
# test_autoencoder(device)
test_autoencoder(device, dataset_path="test", checkpoint_path="autoencoder.pth.tar", result_path="result", save_result=True)
# encode_image_dataset(device, model_file='model3/checkpoint.pth.tar', image_folder='ShapeNetRendering', encoded_image_folder='dataset')
print('Finish')