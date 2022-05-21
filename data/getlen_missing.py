import os
import torch
import numpy as np


if __name__ == '__main__':

    if os.path.isfile(os.path.join('..', 'feature', 'training', '_0CqozZun3U_bn.npy')):
        print('video: {}'.format('_0CqozZun3U'))
        video_prefix = os.path.join('..', 'feature', 'training', '_0CqozZun3U')
        resnet_feat = torch.from_numpy(np.load(video_prefix + '_resnet.npy')).float()
        bn_feat = torch.from_numpy(np.load(video_prefix + '_bn.npy')).float()
        assert resnet_feat.size(0) == bn_feat.size(0), 'number of frames does not match in feature!'
        total_frame = bn_feat.size(0)
        print(total_frame)
