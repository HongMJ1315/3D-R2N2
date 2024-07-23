import torch
import torch.nn as nn
import os
from lib.checkpoint import load_checkpoint
from lib.autoencoder import *
import time

def image_encoder(device, model, imges):
    model.eval()
    ret = []
    for img in imges:
        with torch.no_grad():
            img = img.to(device)
            output = model(img)
            output = output.cpu()
            output = output.numpy()
            ret.append(output)
    return ret

def write_encoded_data(output_directory, label, data):
    if not os.path.exists(output_directory + label):
        os.makedirs(output_directory + label)
    file_path = os.path.join(output_directory + label, 'count.txt')
    try:
        with open(file_path, 'r') as f:
            count = int(f.read())
        f.close()   
    except:
        count = 0
        
    file_path = os.path.join(output_directory + label, 'encoded.bin')
    with open(file_path, 'ab') as f:
        print('Save File:{}'.format(file_path))
        data.tofile(f)
        print('Finish Save File:{}'.format(file_path))
    f.close()
    file_path = os.path.join(output_directory + label, 'count.txt')
    
    count += 1
    with open(file_path, 'w') as f:
        f.write(str(count))
    f.close()

def encode_image(image_directory, model_file, device, output_directory):
    print('Encode Image, Device:{}'.format(device))
    model = Autoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    load_checkpoint(model_file, model, optimizer, device)
    model = model.encoder
    model.eval()
    model.to(device)
    datas = []
    labels = []
    encode_datas = []
    cnt = 0
    print('Start Encode Image Folder:{}'.format(image_directory))
    for root, dirs, files in os.walk(image_directory):
        start_io = time.time()
        for file in files:
            file_name = os.path.join(root, file)
            label = file_name.split('/')[-3]

            if(file_name.split('.')[-1] != 'png'):
                continue
            cnt += 1
            img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
            if(img is None):
                print("Error:{}".format(file_name))
                continue
            datas.append(img)
            labels.append(label)
            if(cnt > 1000):
                # SHoew Current Time hh:mm:ss
                datas = image_preprocessing(datas)
                result = image_encoder(device, model, datas)
                for i, label in enumerate(labels):
                    data = result[i].astype(np.float32).reshape(1, -1)  # 確保數據類型為 float32
                    write_encoded_data(output_directory, label, data)
                datas = []
                labels = []
                cnt = 0
    if(len(datas) > 0):
        datas = image_preprocessing(datas)
        result = image_encoder(device, model, datas)
        for i, label in enumerate(labels):
            data = result[i].astype(np.float32).reshape(1, -1)  # 確保數據類型為 float32
            write_encoded_data(output_directory, label, data)
        datas = []
        labels = []
        cnt = 0
    print('Finish Encode Image')
    return encode_datas

def encode_image_dataset(device, image_folder = DEFAULT_RENDERING_DATASET_FOLDER, model_file = DEFAULT_CHACKPOINT_FILE, encoded_image_folder = DEFAULT_ENCODED_DATASET_FOLDER):
    if(encoded_image_folder[-1] != '/'):
        encoded_image_folder += '/'
    if(not os.path.exists(encoded_image_folder)):
        os.makedirs(encoded_image_folder )
    
    encode_image(image_folder, model_file, device, encoded_image_folder)

def load_encoded_data(file_path):
    with open(file_path + '/count.txt', 'r') as f:
        count = int(f.read())
    
    data = np.fromfile(file_path + '/encoded.bin', dtype=np.float32).reshape(count, -1)
    return data    
