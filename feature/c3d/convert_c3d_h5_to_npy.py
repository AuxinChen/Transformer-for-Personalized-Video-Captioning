import os
import h5py
import json
import numpy as np

if __name__ == '__main__':
    d = h5py.File('sub_activitynet_v1-3.c3d.hdf5', 'r')
    id_list = {}
    with open('train_ids.json', 'r') as f:
        id_list['training'] = json.load(f)
    with open('val_ids.json', 'r') as f:
        id_list['validation'] = json.load(f)
    with open('test_ids.json', 'r') as f:
        id_list['testing'] = json.load(f)
    for vid in d.keys():
        feat = d[vid]['c3d_features'][:].astype('float32')
        for split, ids in id_list.items():
            if vid in ids:
                np.save(os.path.join(split, vid[2:] + '.npy'), feat)
