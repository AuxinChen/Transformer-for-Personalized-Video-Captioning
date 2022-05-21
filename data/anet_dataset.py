import os
import math
import json
import pickle
import numpy as np
from collections import defaultdict, OrderedDict
# torch
import torch
import torchtext
import torchtext.transforms as T
# data iou
from .utils import segment_iou


def get_vocab_and_sentences(data_root, dataset_file, vocab_file=None, save_vocab=False, load_vocab=False, vocab_path=None):
    with open(os.path.join(data_root, dataset_file), 'r') as data_file:
        data = json.load(data_file)
    if load_vocab and os.path.exists(os.path.join('dats', vocab_path)):
        with open(os.path.join('dats', vocab_path), 'rb') as f:
            vocab = pickle.load(f)
    else:
        if vocab_file is not None:
            with open(os.path.join(data_root, vocab_file), 'r') as f:
                words = f.read().strip().split('\n')
            vocab = torchtext.vocab.vocab(OrderedDict(), specials=words)
        else:
            tokenizer = torchtext.data.utils.get_tokenizer('spacy', language='en_core_web_sm')
            def yield_tokens(data):
                for vid, val in data.items():
                    if val['subset'] in ['training', 'validation']:
                        for keyword in val['keywords']:
                            for ann in val['keywords'][keyword]:
                                yield tokenizer(ann['sentence'].strip().lower())
            vocab = torchtext.vocab.build_vocab_from_iterator(yield_tokens(data),
                                                              min_freq=5,
                                                              specials=['<unk>', '<pad>', '<init>', '<eos>'],
                                                              special_first=True)
        vocab.set_default_index(vocab['<unk>'])
        if save_vocab:
            with open(os.path.join('dats', vocab_path), 'wb') as f:
                pickle.dump(vocab, f)
    return vocab, data


class ANetDataset(torch.utils.data.Dataset):

    def __init__(self, data_root, feature_root, split, slide_window_size, dur_file, kernel_list,
                 vocab, raw_data, feature_type, pos_thresh, neg_thresh, stride_factor, logger, max_length=20,
                 save_samplelist=False, load_samplelist=False, sample_listpath=None):
        super(ANetDataset, self).__init__()
        self.slide_window_size = slide_window_size # maxlen of the frames in the video
        self.feature_type = feature_type
        self.split_path = os.path.join(feature_root, feature_type, split)
        if load_samplelist and os.path.exists(os.path.join('dats', sample_listpath)):
            logger.info('loading dataset: {}'.format(sample_listpath))
            with open(os.path.join('dats', sample_listpath), 'rb') as f:
                self.sample_list = pickle.load(f)
        else:
            # create text transformation
            tokenizer = torchtext.data.utils.get_tokenizer('spacy', language='en_core_web_sm')
            text_transform = T.Sequential(
                T.Truncate(max_length - 2),
                T.AddToken(token='<init>', begin=True),
                T.AddToken(token='<eos>', begin=False),
                T.VocabTransform(vocab),
                T.ToTensor(padding_value=vocab['<pad>'])
            )
            keyword_transform = T.Sequential(
                T.Truncate(5),
                T.VocabTransform(vocab),
                T.ToTensor(padding_value=vocab['<pad>'])
            )
            # numericalize sentences
            vid_list = [] # list of vids in this split
            train_sentences = []
            train_keywords = []
            for vid, val in raw_data.items():
                if feature_type == 'tsn':
                    fn = os.path.join(self.split_path, vid + '_bn.npy')
                elif feature_type == 'c3d':
                    fn = os.path.join(self.split_path, vid + '.npy')
                if val['subset'] == split and os.path.isfile(fn):
                    vid_list.append(vid)
                    for keyword in val['keywords']:
                        for ann in val['keywords'][keyword]:
                            train_sentences.append(tokenizer(ann['sentence'].strip().lower()))
                            train_keywords.append(tokenizer(keyword.strip().lower()))
            sentence_idx = text_transform(train_sentences)
            keyword_idx = keyword_transform(train_keywords)
            assert sentence_idx.size(0) == len(train_sentences), 'Error in numericalize sentences'
            # save sentences indexes
            idx = 0
            for vid in vid_list:
                for keyword in raw_data[vid]['keywords']:
                    for ann in raw_data[vid]['keywords'][keyword]:
                        ann['sentence_idx'] = sentence_idx[idx]
                        ann['keyword_idx'] = keyword_idx[idx]
                        idx += 1
            logger.info('size of the sentence block variable ({}): {}'.format(split, list(sentence_idx.size())))
            logger.info('size of the keyword block variable ({}): {}'.format(split, list(keyword_idx.size())))
            # construct initial anchors
            anc_len_lst = []
            anc_cen_lst = []
            for i in range(len(kernel_list)):
                kernel_len = kernel_list[i]
                anc_cen = np.arange(kernel_len/2., slide_window_size+1-(kernel_len)/2., math.ceil(kernel_len/stride_factor))
                anc_len = np.full(anc_cen.shape, kernel_len)
                anc_len_lst.append(anc_len)
                anc_cen_lst.append(anc_cen)
            anc_len_all = np.hstack(anc_len_lst)
            anc_cen_all = np.hstack(anc_cen_lst)
            # get sampling second and total frames
            frame_to_second = {} # values around 0.5
            featlens = {} # number of frames in video feature
            sampling_sec = 0.5 # hard coded, only support 0.5
            with open(os.path.join(data_root, dur_file)) as f:
                for line in f:
                    vid_name, vid_dur, vid_frame, vid_featlen = [l.strip() for l in line.split(',')]
                    vid_dur, vid_frame, vid_featlen = float(vid_dur), int(vid_frame), int(vid_featlen)
                    frame_to_second[vid_name] = vid_dur*int(vid_frame*1./int(vid_dur)*sampling_sec)*1./vid_frame
                    featlens[vid_name] = vid_featlen
            # load annotation per video and construct training set
            self.sample_list = []  # list of tuple for data samples
            keyword_stats = []
            pos_anchor_stats, neg_anchor_stats = [], []
            missing_prop = 0
            vid_num = len(vid_list)
            for idx, vid in enumerate(vid_list):
                n_pos_seg, n_neg_seg = 0, 0
                if vid not in frame_to_second: # some c3d feature file have no sampling_sec data
                    print('cannot find frame_to_second for video {}'.format(vid), end='\r')
                    frame_to_second[vid] = sampling_sec
                    featlens[vid] = slide_window_size
                for keyword, annotations in raw_data[vid]['keywords'].items():
                    pos_seg, neg_seg, is_missing = self._get_pos_neg(annotations, vid, frame_to_second[vid],
                                                                     featlens[vid], anc_len_all, anc_cen_all,
                                                                     pos_thresh, neg_thresh)
                    missing_prop += is_missing
                    for k in pos_seg: # each sentence is regarded as a single sample
                        # all neg_segs are the same, since they need to be negative
                        keyword = pos_seg[k][0][-2].tolist() # all keywords in pos_seg[k] are the same
                        sent = pos_seg[k][0][-1].tolist() # all sentences in pos_seg[k] are the same
                        pos_seg_remain = [seg[:-2] for seg in pos_seg[k]]
                        self.sample_list.append((vid, keyword, sent, pos_seg_remain, neg_seg))
                        n_pos_seg += len(pos_seg_remain)
                    n_neg_seg += len(neg_seg)
                    keyword_stats.append(len(annotations))
                pos_anchor_stats.append(n_pos_seg)
                neg_anchor_stats.append(n_neg_seg)
                print('[{}/{}] video {} has {} positive anchors and {} negative anchors'.format(
                    idx, vid_num, vid, n_pos_seg, n_neg_seg), end='\r')
            print(' ' * 100, end='\r')
            logger.info('total number of {} videos: {}'.format(split, len(vid_list)))
            logger.info('total number of {} samples (unique segments): {}'.format(split, len(self.sample_list)))
            logger.info('total number of annotations: {}'.format(len(train_sentences)))
            logger.info('total number of missing annotations: {}'.format(missing_prop))
            logger.info('avg sentence per keyword: {:.2f}'.format(np.mean(keyword_stats)))
            logger.info('avg pos anc: {:.2f}, avg neg anc: {:.2f}'.format(np.mean(pos_anchor_stats), np.mean(neg_anchor_stats)))
            # save samples if save_samplelist is True
            if save_samplelist:
                logger.info('saving dataset: {}'.format(sample_listpath))
                with open(os.path.join('dats', sample_listpath), 'wb') as f:
                    pickle.dump(self.sample_list, f)

    def _get_pos_neg(self, annotations, vid, sampling_sec, total_frame,
                     anc_len_all, anc_cen_all, pos_thresh, neg_thresh):
        window_start = 0
        window_end = self.slide_window_size
        window_start_t = window_start * sampling_sec
        window_end_t = window_end * sampling_sec
        pos_seg = defaultdict(list) # return a list if key not exists, map ann_idx to list of anchors
        max_overlap = [0] * len(anc_len_all) # the max overlap value in an anchor
        pos_collected = [False] * len(anc_len_all) # whether positive segment is collected or not
        for j in range(len(anc_len_all)):
            matched = None
            for ann_idx, ann in enumerate(annotations):
                seg = ann['segment']
                gt_start = seg[0] / sampling_sec # start frame
                gt_end = seg[1] / sampling_sec # end frame
                if gt_start > gt_end:
                    gt_start, gt_end = gt_end, gt_start
                if anc_cen_all[j] + anc_len_all[j] / 2. <= total_frame: # smaller than total frame
                    if window_start_t <= seg[0] and window_end_t + sampling_sec * 2 >= seg[1]:
                        overlap = segment_iou(
                            np.array([gt_start, gt_end]),
                            np.array([[anc_cen_all[j] - anc_len_all[j] / 2., anc_cen_all[j] + anc_len_all[j] / 2.]])
                        ).item()
                        max_overlap[j] = max(overlap, max_overlap[j])
                        if not pos_collected[j] and overlap >= pos_thresh:
                            len_offset = math.log((gt_end - gt_start) / anc_len_all[j])
                            cen_offset = ((gt_end + gt_start) / 2. - anc_cen_all[j]) / anc_len_all[j]
                            matched = (ann_idx, j, overlap, len_offset, cen_offset, ann['keyword_idx'], ann['sentence_idx'])
                            pos_collected[j] = True
            if matched is not None:
                pos_seg[matched[0]].append(matched[1:])
        missing_prop = 0 # annotations have no corresponding anchors
        if len(pos_seg.keys()) != len(annotations):
            print('some annotations in video {} have no matching proposal'.format(vid), end='\r')
            missing_prop = len(annotations) - len(pos_seg.keys())
        neg_seg = [] # the indexes of negative anchors
        for j, overlap in enumerate(max_overlap):
            if overlap < neg_thresh:
                neg_seg.append((j, overlap))
        return pos_seg, neg_seg, missing_prop

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        vid, keyword, sentence, pos_seg, neg_seg = self.sample_list[index]
        if self.feature_type == 'tsn':
            resnet_feat = torch.from_numpy(np.load(os.path.join(self.split_path, vid + '_resnet.npy'))).float()
            bn_feat = torch.from_numpy(np.load(os.path.join(self.split_path, vid + '_bn.npy'))).float()
            img_feat = torch.zeros(self.slide_window_size, resnet_feat.size(-1)+bn_feat.size(-1), dtype=torch.float)
            torch.cat((resnet_feat, bn_feat), dim=1, out=img_feat[:min(bn_feat.size(0), self.slide_window_size)])
        elif self.feature_type == 'c3d':
            c3d_feat = torch.from_numpy(np.load(os.path.join(self.split_path, vid + '.npy'))).float()
            c3d_feat = c3d_feat[::2] # downsampling
            img_feat = torch.zeros(self.slide_window_size, c3d_feat.size(-1), dtype=torch.float)
            img_feat[:min(c3d_feat.size(0), self.slide_window_size)] = c3d_feat[:self.slide_window_size]
        return img_feat, pos_seg, neg_seg, keyword, sentence


def anet_collate_fn(batch_lst):
    sample_each = 10
    img_feat, pos_seg, neg_seg, keyword, sentence = batch_lst[0]
    batch_size = len(batch_lst)
    keyword_batch = torch.ones(batch_size, len(keyword), dtype=torch.int64)
    sentence_batch = torch.ones(batch_size, len(sentence), dtype=torch.int64)
    img_batch = torch.zeros(batch_size, img_feat.size(0), img_feat.size(1), dtype=torch.float)
    tempo_seg_pos = torch.zeros(batch_size, sample_each, 4, dtype=torch.float) # anc_idx, overlap, len_offset, cen_offset
    tempo_seg_neg = torch.zeros(batch_size, sample_each, 2, dtype=torch.float) # anc_idx, overlap
    for batch_idx in range(batch_size):
        img_feat, pos_seg, neg_seg, keyword, sentence = batch_lst[batch_idx]
        img_batch[batch_idx, :] = img_feat
        keyword_batch[batch_idx, :] = torch.LongTensor(keyword)
        sentence_batch[batch_idx, :] = torch.LongTensor(sentence)
        # sample positive anchors
        pos_seg_tensor = torch.FloatTensor(pos_seg)
        perm_idx = torch.randperm(len(pos_seg))
        if len(pos_seg) >= sample_each:
            tempo_seg_pos[batch_idx, :, :] = pos_seg_tensor[perm_idx[:sample_each]]
        else:
            tempo_seg_pos[batch_idx, :len(pos_seg), :] = pos_seg_tensor
            idx = torch.randint(0, len(pos_seg), size=(sample_each-len(pos_seg),))
            tempo_seg_pos[batch_idx, len(pos_seg):, :] = pos_seg_tensor[idx]
        # sample negative anchors
        neg_seg_tensor = torch.FloatTensor(neg_seg)
        perm_idx = torch.randperm(len(neg_seg))
        if len(neg_seg) >= sample_each:
            tempo_seg_neg[batch_idx, :, :] = neg_seg_tensor[perm_idx[:sample_each]]
        else:
            tempo_seg_neg[batch_idx, :len(neg_seg), :] = neg_seg_tensor
            idx = torch.randint(0, len(neg_seg), size=(sample_each-len(neg_seg),))
            tempo_seg_neg[batch_idx, len(neg_seg):, :] = neg_seg_tensor[idx]
    return img_batch, tempo_seg_pos, tempo_seg_neg, keyword_batch, sentence_batch
