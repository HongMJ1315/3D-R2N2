# %%
import os
import cv2
import json
import lib.binvox_rw as binvox_rw
import numpy as np

# %%
DEFAULT_BINVOX_DATASET_FOLDER = "ShapeNetVox32"
DEFAULT_VOXEL_DATASET_FOLDER = "dataset"

# %%
def read_binvox(file_path):
    with open(file_path, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f).data
        return model
def load_model_data(floder):
    dataset = []
    print("Loading data from " + floder)
    for root, dirs, files in os.walk(floder):
        for file in files:
            file_name = os.path.join(root, file)
            if(file_name.split('.')[-1] != 'binvox'):
                continue
            label = file_name.split('/')[-2]
            model = read_binvox(file_name)
            dataset.append((label, model))
    print("Data loaded")
    return dataset


# %%
def binvox_dataset(binvox_folder=DEFAULT_BINVOX_DATASET_FOLDER, output_folder=DEFAULT_VOXEL_DATASET_FOLDER):
    result = load_model_data(binvox_folder)
    # Turn the result data form bool to int
    for i in range(len(result)):
        result[i] = (result[i][0], result[i][1].astype(int))

    if output_folder[-1] != '/':
        output_folder += '/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for label, data in result:
        if not os.path.exists(output_folder + label):
            os.makedirs(output_folder + label)
        with open(output_folder + label + '/voxel.txt', 'a') as f:
            # Write the shape of the array
            f.write(' '.join(map(str, data.shape)) + '\n')
            # Flatten the array and write the data as a string of 0s and 1s
            f.write(''.join(map(str, data.flatten())) + '\n')
            print('Save File:{}'.format(output_folder + label + '/voxel.txt'))
            print('Finish Save File:{}'.format(output_folder + label + '/voxel.txt'))
        f.close()

# %%
def load_voxel_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        shape = tuple(map(int, lines[0].strip().split()))
        data_str = lines[1].strip()
        data = np.array(list(map(int, data_str))).reshape(shape)
    return data

# %%
if __name__ == '__main__':
    binvox_dataset()

