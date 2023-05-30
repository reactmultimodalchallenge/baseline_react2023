import numpy as np
import pandas as pd
import argparse
import os


def parse_arg():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--dataset-path', default="./data", type=str, help="dataset path")
    parser.add_argument('--partition', default="val", type=str, help="dataset partition")
    args = parser.parse_args()
    return args

args = parse_arg()
root_path = args.dataset_path
data_path = pd.read_csv(os.path.join(root_path, 'data_indices.csv'), header=None, delimiter=',')
data_path = data_path.drop(0)
all_matrix = np.load(os.path.join(root_path, 'Approprirate_facial_reaction.npy'))

all_matrix_w = all_matrix.shape[0]
print("all_matrix shape:", all_matrix.shape)

val_path = pd.read_csv(os.path.join(root_path, str(args.partition)+'.csv'), header=None, delimiter=',')
val_path = val_path.drop(0)

index_list = []
_sum = 0
for index, row in val_path.iterrows():
    _sum += 1
    flag = 0
    items = row[1].split('/')
    if len(items)>4:
        path_val_item = os.path.join(items[0].upper(), items[2]+'_'+items[1], items[4])
    else:
        path_val_item = os.path.join(items[0].upper(),items[1],items[3])
    for index_2, row2 in data_path.iterrows():
        path_data_index = os.path.join(row2[0].upper(), row2[1], row2[2])
        if path_val_item == path_data_index:
            index_list.append(index_2-1)
            flag = 1

l_m = len(index_list)
new_matrix = np.zeros((l_m*2, l_m*2))
for index, item in enumerate(index_list):
    for j, item_j in enumerate(index_list):
        new_matrix[index, j] = all_matrix[item, item_j]
        new_matrix[index, j+l_m] = all_matrix[item, item_j + all_matrix_w//2]
        new_matrix[index+l_m, j] = all_matrix[item + all_matrix_w//2, item_j]
        new_matrix[index+l_m, j+l_m] = all_matrix[item + all_matrix_w//2, item_j + all_matrix_w//2]


np.save(os.path.join(root_path, 'neighbour_emotion_' + str(args.partition) + '.npy'), new_matrix)
print(new_matrix.shape)
