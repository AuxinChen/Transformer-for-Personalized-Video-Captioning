import os
import torch
import numpy as np
import torchtext
import torchtext.transforms as T


class ANetTestDataset(torch.utils.data.Dataset):

    def __init__(self, feature_root, split, slide_window_size, vocab, raw_data, feature_type, learn_mask=False):
        super(ANetTestDataset, self).__init__()
        self.slide_window_size = slide_window_size
        self.learn_mask = learn_mask
        self.feature_type = feature_type
        self.split_path = os.path.join(feature_root, feature_type, split)
        self.sample_list = [] # list of list for data samples
        tokenizer = torchtext.data.utils.get_tokenizer('spacy', language='en_core_web_sm')
        keyword_transform = T.Sequential(
            T.Truncate(5),
            T.VocabTransform(vocab),
            T.ToTensor(padding_value=vocab['<pad>'])
        )
        for vid, val in raw_data.items():
            if feature_type == 'tsn':
                fn = os.path.join(self.split_path, vid + '_bn.npy')
            elif feature_type == 'c3d':
                fn = os.path.join(self.split_path, vid + '.npy')
            if val['subset'] == split and os.path.isfile(fn):
                for keyword in val['keywords']:
                    keyword_idx = keyword_transform(tokenizer(keyword.strip().lower()))
                    self.sample_list.append((vid, keyword, keyword_idx))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        vid, keyword, keyword_idx = self.sample_list[index]
        if self.feature_type == 'tsn':
            resnet_feat = torch.from_numpy(np.load(os.path.join(self.split_path, vid + '_resnet.npy'))).float()
            bn_feat = torch.from_numpy(np.load(os.path.join(self.split_path, vid + '_bn.npy'))).float()
            total_frame = bn_feat.size(0)
            if self.learn_mask:
                img_feat = torch.zeros(self.slide_window_size, resnet_feat.size(1)+bn_feat.size(1), dtype=torch.float)
                torch.cat((resnet_feat, bn_feat), dim=1, out=img_feat[:min(total_frame, self.slide_window_size)])
            else:
                img_feat = torch.cat((resnet_feat, bn_feat), dim=1)
        elif self.feature_type == 'c3d':
            c3d_feat = torch.from_numpy(np.load(os.path.join(self.split_path, vid + '.npy'))).float()
            c3d_feat = c3d_feat[::2] # downsampling
            total_frame = c3d_feat.size(0)
            img_feat = torch.zeros(self.slide_window_size, c3d_feat.size(-1), dtype=torch.float)
            img_feat[:min(c3d_feat.size(0), self.slide_window_size)] = c3d_feat[:self.slide_window_size]
        return img_feat, total_frame, vid, keyword, keyword_idx


def anet_test_collate_fn(batch_lst):
    img_feat, _, _, _, keyword_idx = batch_lst[0]
    batch_size = len(batch_lst)
    img_batch = torch.zeros(batch_size, img_feat.size(0), img_feat.size(1), dtype=torch.float)
    keyword_batch = torch.ones(batch_size, len(keyword_idx), dtype=torch.int64)
    frame_length = torch.zeros(batch_size, dtype=torch.int32)
    vids = []
    keywords = []
    for batch_idx in range(batch_size):
        img_feat, total_frame, vid, keyword, keyword_idx = batch_lst[batch_idx]
        img_batch[batch_idx, :] = img_feat
        keyword_batch[batch_idx, :] = torch.LongTensor(keyword_idx)
        frame_length[batch_idx] = total_frame
        vids.append(vid)
        keywords.append(keyword)
    return img_batch, keyword_batch, frame_length, vids, keywords
