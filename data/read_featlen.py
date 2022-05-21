import os
import json
import torch
import numpy as np


if __name__ == '__main__':
    vid_names = set()
    dur_data = []
    with open('anet_annotations_trainval.json', 'r') as data_file:
        data_all = json.load(data_file)
    data = data_all['database']
    with open('anet_duration_frame.csv', 'r') as f:
        for line in f:
            vid_name, vid_dur, vid_frame = [l.strip() for l in line.strip().split(',')]
            vid_names.add(vid_name)
            if (vid_name in data) and os.path.isfile(os.path.join('..', 'feature', data[vid_name]['subset'], vid_name + '_bn.npy')):
                print('video: {}'.format(vid_name))
                video_prefix = os.path.join('..', 'feature', data[vid_name]['subset'], vid_name)
                resnet_feat = torch.from_numpy(np.load(video_prefix + '_resnet.npy')).float()
                bn_feat = torch.from_numpy(np.load(video_prefix + '_bn.npy')).float()
                assert resnet_feat.size(0) == bn_feat.size(0), 'number of frames does not match in feature!'
                total_frame = bn_feat.size(0)
                dur_data.append((vid_name, float(vid_dur), int(vid_frame), total_frame))
    print('num of annotated data:', len(data))
    print('num of dur data', len(dur_data))
    for vid_name in data.keys():
        if vid_name not in vid_names:
            print(vid_name, os.path.isfile(os.path.join('..', 'feature', data[vid_name]['subset'], vid_name + '_bn.npy')))
    with open('anet_duration_frame_feature.csv', 'w', encoding='utf-8') as fout:
        fout.write('\n'.join(['{:s},{:.2f},{:d},{:d}'.format(*data) for data in dur_data]))
