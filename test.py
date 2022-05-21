import os
import sys
import json
import time
import torch
import random
import logging
import argparse
import datetime
import numpy as np
from collections import defaultdict
# misc
from data.anet_dataset import get_vocab_and_sentences
from data.anet_test_dataset import ANetTestDataset, anet_test_collate_fn
from model.tpvc import TPVC
from tools.evaluate import ANETcaptions


class Instructor:

    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.logger.addHandler(logging.FileHandler(os.path.join('logs', args.log_name)))
        self._print_args()

    def _print_args(self):
        self.logger.info('TRAINING ARGUMENTS:')
        for arg in vars(self.args):
            self.logger.info('>>> {}: {}'.format(arg, getattr(self.args, arg)))

    def _load_data(self):
        vocab, raw_data = get_vocab_and_sentences(self.args.data_root, self.args.dataset_file, self.args.vocab_file)
        self.logger.info('id of <unk>, <pad>, <init> and <eos>: {}, {}, {}, {}'.format(vocab['<unk>'], vocab['<pad>'], vocab['<init>'], vocab['<eos>']))
        self.logger.info('number of words in the vocab: {}'.format(len(vocab)))
        test_dataset = ANetTestDataset(self.args.feature_root,
                                       self.args.val_data_folder,
                                       self.args.slide_window_size,
                                       vocab, raw_data,
                                       self.args.feature_type,
                                       self.args.learn_mask)
        self.logger.info('total number of samples (unique keywords): {}'.format(len(test_dataset)))
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=self.args.batch_size,
                                                  shuffle=False,
                                                  num_workers=self.args.num_workers,
                                                  collate_fn=anet_test_collate_fn)
        return test_loader, vocab

    def _get_model(self, vocab):
        model = ActionPropDenseCap(d_model=self.args.d_model,
                                   d_hidden=self.args.d_hidden,
                                   d_emb=self.args.d_emb,
                                   n_layers=self.args.n_layers,
                                   n_heads=self.args.n_heads,
                                   vocab=vocab,
                                   in_emb_dropout=self.args.in_emb_dropout,
                                   attn_dropout=self.args.attn_dropout,
                                   cap_dropout=self.args.cap_dropout,
                                   nsamples=0,
                                   kernel_list=self.args.kernel_list,
                                   stride_factor=self.args.stride_factor,
                                   feature_type=self.args.feature_type,
                                   learn_mask=self.args.learn_mask,
                                   use_mask=self.args.use_mask,
                                   no_keyword=self.args.no_keyword,
                                   ablation=self.args.ablation)
        # Initialize the networks and the criterion
        if self.args.checkpoint is not None:
            self.logger.info('Initializing weights from {}'.format(os.path.join('state_dict', self.args.checkpoint)))
            state_dict = model.state_dict()
            checkpoint = torch.load(os.path.join('state_dict', self.args.checkpoint), map_location='cpu')
            for key in state_dict.keys():
                if key in checkpoint and state_dict[key].shape != checkpoint[key].shape:
                    self.logger.info('Removing key {} from checkpoint'.format(key))
                    del checkpoint[key]
            model.load_state_dict(checkpoint, strict=False)
        else:
            raise Exception('checkpoint is required')
        # Ship the model to GPU, maybe
        if self.args.device.type == 'cuda':
            model.cuda()
        return model

    def _infer_results(self, pred_cap_fn, pred_prop_fn):
        self.model.eval()
        densecap_result = defaultdict(lambda: defaultdict(list))
        prop_result = defaultdict(lambda: defaultdict(list))
        avg_prop_num = 0
        frame_to_second = {}
        total = len(self.test_loader)
        with open(os.path.join(self.args.data_root, self.args.dur_file)) as f:
            for line in f:
                vid_name, vid_dur, vid_frame, _ = [l.strip() for l in line.split(',')]
                vid_dur, vid_frame = float(vid_dur), int(vid_frame)
                frame_to_second[vid_name] = vid_dur*int(vid_frame*1./int(vid_dur)*self.args.sampling_sec)*1./vid_frame
        with torch.no_grad():
            for idx, data in enumerate(self.test_loader):
                image_feat, keyword_idx, total_frame, vids, keywords = data
                image_feat = image_feat.to(self.args.device)
                keyword_idx = keyword_idx.to(self.args.device)
                vid, keyword = vids[0], keywords[0]
                if vid not in frame_to_second:
                    sampling_sec = self.args.sampling_sec
                    print('cannot find frame_to_second for video {}'.format(vid), end='\r')
                else:
                    sampling_sec = frame_to_second[vid]
                all_proposal_results = self.model.inference(image_feat,
                                                            keyword_idx,
                                                            total_frame,
                                                            sampling_sec,
                                                            self.args.min_prop_num,
                                                            self.args.max_prop_num,
                                                            self.args.min_prop_before_nms,
                                                            self.args.pos_thresh,
                                                            gated_mask=self.args.gated_mask)
                print('[{}/{}] Write results for video: {}'.format(idx, total, vid) + ' ' * 10, end='\r')
                for pred_start, pred_end, pred_s, sent in all_proposal_results:
                    densecap_result['v_'+vid][keyword].append({
                        'sentence': sent,
                        'timestamp': [pred_start * sampling_sec, pred_end * sampling_sec]
                    })
                    prop_result[vid][keyword].append({
                        'segment': [pred_start * sampling_sec, pred_end * sampling_sec],
                        'score': pred_s
                    })
                avg_prop_num += len(all_proposal_results)
        print(' ' * 100, end='\r')
        self.logger.info('average proposal number: {}'.format(avg_prop_num / len(self.test_loader.dataset)))
        # write captions to json file for evaluation (caption)
        dense_cap_all = {
            'version': 'VERSION 1.0',
            'results': densecap_result,
            'external_data': {
                'used': 'true',
                'details': 'global_pool layer from BN-Inception pretrained from ActivityNet and ImageNet (https://github.com/yjxiong/anet2016-cuhk)'
            }
        }
        with open(pred_cap_fn, 'w') as f:
            json.dump(dense_cap_all, f, indent=4)
        self.logger.info('Captions saved: {}'.format(pred_cap_fn))
        # write proposals to json file for evaluation (proposal)
        prop_all = {
            'version':'VERSION 1.0',
            'results': prop_result,
            'external_data':{
                'used': 'true',
                'details': 'global_pool layer from BN-Inception pretrained from ActivityNet and ImageNet (https://github.com/yjxiong/anet2016-cuhk)'
            }
        }
        with open(pred_prop_fn, 'w') as f:
            json.dump(prop_all, f, indent=4)
        self.logger.info('Proposals saved: {}'.format(pred_prop_fn))

    def run(self):
        fn_id = self.args.result if self.args.result is not None else self.args.timestamp
        true_cap_fn = [os.path.join(self.args.data_root, filename) for filename in self.args.densecap_references]
        pred_cap_fn = os.path.join('results', 'densecap_{}.json'.format(fn_id))
        pred_prop_fn = os.path.join('results', 'prop_{}.json'.format(fn_id))
        if self.args.result is None:
            self.logger.info('=> loading dataset')
            self.test_loader, vocab = self._load_data()
            self.logger.info('=> building model')
            self.model = self._get_model(vocab)
            if args.device.type == 'cuda':
                self.logger.info('=> cuda memory allocated: {}'.format(torch.cuda.memory_allocated(args.device.index)))
            self._infer_results(pred_cap_fn, pred_prop_fn)
        else:
            self.logger.info('=> loading result from: densecap_{}.json and prop_{}.json'.format(fn_id, fn_id))
        evaluator = ANETcaptions(ground_truth_filenames=true_cap_fn,
                                 prediction_filename=pred_cap_fn,
                                 tious=[0.3, 0.5, 0.7, 0.9],
                                 max_proposals=self.args.max_prop_num,
                                 verbose=self.args.verbose,
                                 logger=self.logger)
        evaluator.evaluate()
        # Output the results
        if self.args.verbose:
            for i, tiou in enumerate([0.3, 0.5, 0.7, 0.9]):
                self.logger.info('tIoU: ' , tiou)
                for metric in evaluator.scores:
                    score = evaluator.scores[metric][i]
                    self.logger.info('| {:s}: {:2.4f}'.format(metric, 100*score))
        # Print the averages
        self.logger.info("Average across all tIoUs")
        for metric in evaluator.scores:
            score = evaluator.scores[metric]
            self.logger.info('| {}: {:2.4f}'.format(metric, 100 * sum(score) / float(len(score))))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    ''' Dataset '''
    parser.add_argument('--data_root', default='data', type=str)
    parser.add_argument('--dataset_file', default='anet_annotations_trainval.json', type=str)
    parser.add_argument('--feature_root', default='feature', type=str, help='the feature root')
    parser.add_argument('--dur_file', default='anet_duration_frame_feature.csv', type=str)
    parser.add_argument('--val_data_folder', default='validation', type=str, help='validation data folder')
    parser.add_argument('--densecap_references', nargs='+', default=['val_1.json', 'val_2.json'], type=str)
    parser.add_argument('--vocab_file', default='vocab.txt', type=str, help='vocabulary file')
    parser.add_argument('--checkpoint', default=None, type=str, help='path to a model checkpoint to initialize model weights from')
    parser.add_argument('--feature_type', default='tsn', type=str, choices=['c3d', 'tsn'], help='Which video feature is used')
    parser.add_argument('--batch_size', default=1, type=int, help='only support 1')
    parser.add_argument('--num_workers', default=0, type=int)
    ''' Model '''
    parser.add_argument('--d_model', default=1024, type=int)
    parser.add_argument('--d_hidden', default=2048, type=int)
    parser.add_argument('--d_emb', default=256, type=int)
    parser.add_argument('--n_layers', default=2, type=int)
    parser.add_argument('--n_heads', default=8, type=int)
    parser.add_argument('--in_emb_dropout', default=0.1, type=float)
    parser.add_argument('--attn_dropout', default=0.2, type=float)
    parser.add_argument('--cap_dropout', default=0.2, type=float)
    parser.add_argument('--no_keyword', default=False, action='store_true')
    parser.add_argument('--ablation', default=None, type=str, nargs='+', help='Ablation settings')
    ''' Proposal '''
    parser.add_argument('--slide_window_size', default=480, type=int, help='the (temporal) size of the sliding window')
    parser.add_argument('--slide_window_stride', default=20, type=int, help='the step size of the sliding window')
    parser.add_argument('--sampling_sec', default=0.5, help='sample frame (RGB and optical flow) with which time interval')
    parser.add_argument('--kernel_list', default=[1, 2, 3, 4, 5, 7, 9, 11, 15, 21, 29, 41, 57, 71, 111, 161, 211, 251], type=int, nargs='+')
    parser.add_argument('--max_prop_num', default=100, type=int, help='the maximum number of proposals per video')
    parser.add_argument('--min_prop_num', default=20, type=int, help='the minimum number of proposals per video')
    parser.add_argument('--min_prop_before_nms', default=100, type=int, help='the minimum number of proposals per video')
    parser.add_argument('--pos_thresh', default=0.7, type=float)
    parser.add_argument('--stride_factor', default=50, type=int, help='the proposal temporal conv kernel stride is determined by math.ceil(kernel_len/stride_factor)')
    parser.add_argument('--use_mask', default=False, action='store_true')
    parser.add_argument('--gated_mask', default=False, action='store_true')
    parser.add_argument('--learn_mask', default=False, action='store_true')
    ''' Environment '''
    parser.add_argument('--verbose', default=False, action='store_true', help='print intermediate steps')
    parser.add_argument('--seed', default=None, type=int, help='Random number generator seed to use')
    parser.add_argument('--timestamp', default=None, type=str, help='Timestamp of running script')
    parser.add_argument('--result', default=None, type=str, help='Result file')
    parser.add_argument('--device', default=None, type=str, choices=['cpu', 'cuda'], help='Selected device')
    args = parser.parse_args()
    assert args.batch_size == 1, 'batch size has to be 1!'
    assert args.slide_window_size >= args.slide_window_stride
    args.log_name = '{}_test.log'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[2:])
    args.timestamp = args.timestamp if args.timestamp else str(int(time.time())) + format(random.randint(0, 999), '03')
    args.seed = args.seed if args.seed else random.randint(0, 2**32-1)
    args.device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ''' Set seeds '''
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    for dir_name in ['logs', 'results']:
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
    ''' Test the model '''
    ins = Instructor(args)
    ins.run()
