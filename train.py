import os
import sys
import time
import torch
import random
import logging
import argparse
import datetime
import numpy as np
from data.anet_dataset import ANetDataset, anet_collate_fn, get_vocab_and_sentences
from model.tpvc import TPVC


class Instructor:

    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.logger.addHandler(logging.FileHandler(os.path.join('logs', args.log_name)))
        self._print_args()
        self.logger.info('=> loading dataset')
        self.train_loader, self.valid_loader, vocab, self.train_sampler = self._load_data()
        self.logger.info('=> building model')
        self.model, self.module = self._get_model(vocab)
        if args.device.type == 'cuda':
            self.logger.info('=> cuda memory allocated: {}'.format(torch.cuda.memory_allocated(args.device.index)))

    def _print_args(self):
        self.logger.info('TRAINING ARGUMENTS:')
        for arg in vars(self.args):
            self.logger.info('>>> {}: {}'.format(arg, getattr(self.args, arg)))

    def _load_data(self):
        vocab, raw_data = get_vocab_and_sentences(self.args.data_root, self.args.dataset_file,
                                                  vocab_file=self.args.vocab_file,
                                                  save_vocab=self.args.save_vocab,
                                                  load_vocab=self.args.load_vocab,
                                                  vocab_path=self.args.vocab_path)
        self.logger.info('id of <unk>, <pad>, <init> and <eos>: {}, {}, {}, {}'.format(vocab['<unk>'], vocab['<pad>'], vocab['<init>'], vocab['<eos>']))
        self.logger.info('number of words in the vocab: {}'.format(len(vocab)))
        train_dataset = ANetDataset(self.args.data_root,
                                    self.args.feature_root,
                                    self.args.train_data_folder,
                                    self.args.slide_window_size,
                                    self.args.dur_file,
                                    self.args.kernel_list,
                                    vocab, raw_data,
                                    self.args.feature_type,
                                    self.args.pos_thresh,
                                    self.args.neg_thresh,
                                    self.args.stride_factor,
                                    self.logger,
                                    self.args.maxlen,
                                    self.args.save_train_samplelist,
                                    self.args.load_train_samplelist,
                                    self.args.train_samplelist_path)
        if self.args.distributed:
            torch.distributed.init_process_group(backend=self.args.dist_backend,
                                                 init_method=self.args.dist_url,
                                                 world_size=self.args.world_size)
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.args.batch_size,
                                                   shuffle=(train_sampler is None),
                                                   sampler=train_sampler,
                                                   num_workers=self.args.num_workers,
                                                   collate_fn=anet_collate_fn)
        valid_dataset = ANetDataset(self.args.data_root,
                                    self.args.feature_root,
                                    self.args.val_data_folder,
                                    self.args.slide_window_size,
                                    self.args.dur_file,
                                    self.args.kernel_list,
                                    vocab, raw_data,
                                    self.args.feature_type,
                                    self.args.pos_thresh,
                                    self.args.neg_thresh,
                                    self.args.stride_factor,
                                    self.logger,
                                    self.args.maxlen,
                                    self.args.save_valid_samplelist,
                                    self.args.load_valid_samplelist,
                                    self.args.valid_samplelist_path)
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=self.args.valid_batch_size,
                                                   shuffle=False,
                                                   num_workers=self.args.num_workers,
                                                   collate_fn=anet_collate_fn)
        return train_loader, valid_loader, vocab, train_sampler

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
                                   nsamples=self.args.train_sample,
                                   kernel_list=self.args.kernel_list,
                                   stride_factor=self.args.stride_factor,
                                   feature_type=self.args.feature_type,
                                   learn_mask=(self.args.mask_weight > 1e-6),
                                   gt_mask=self.args.gt_mask,
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
        # Ship the model to GPU, maybe
        if self.args.device.type == 'cuda':
            if self.args.distributed:
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model)
                module = model.module
            elif torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model).cuda()
                module = model.module
            else:
                model.cuda()
                module = model
        else:
            module = model
        return model, module

    def _train(self, epoch, optimizer):
        self.model.train() # training mode
        train_loss, n_train = 0, 0
        n_batch = len(self.train_loader)
        sample_prob = min(self.args.sample_prob, int(epoch/5)*0.05)
        for i_batch, data in enumerate(self.train_loader):
            img_batch, tempo_seg_pos, tempo_seg_neg, keyword_batch, sentence_batch = map(lambda x: x.to(self.args.device), data)
            outputs = self.model(img_batch, tempo_seg_pos, tempo_seg_neg, keyword_batch, sentence_batch, sample_prob, self.args.gated_mask)
            pred_score, gt_score, pred_offsets, gt_offsets, pred_sent, gt_sent, mask_loss = outputs
            cls_loss = self.module.bce_loss(pred_score, gt_score)
            reg_loss = self.module.reg_loss(pred_offsets, gt_offsets)
            sent_loss = self.module.ce_loss(pred_sent, gt_sent)
            total_loss = cls_loss * self.args.cls_weight + reg_loss * self.args.reg_weight + sent_loss * self.args.sent_weight
            if mask_loss is not None:
                total_loss += mask_loss * self.args.mask_weight
            else:
                mask_loss = cls_loss.new_empty(1).fill_(0)
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), self.args.grad_norm)
            optimizer.step()
            train_loss += total_loss.item() * img_batch.size(0)
            n_train += img_batch.size(0)
            print('iter: [{}/{}], train loss: {:.4f}, class: {:.4f}, reg: {:.4f}, sent: {:.4f}, mask: {:.4f}'.format(
                i_batch, n_batch, total_loss.item(), cls_loss.item(), reg_loss.item(), sent_loss.item(), mask_loss.item()), end='\r')
        print(' ' * 100, end='\r')
        return train_loss / n_train

    def _validate(self):
        self.model.eval()
        n_batch = len(self.valid_loader)
        valid_loss, val_cls_loss, val_reg_loss, val_sent_loss, val_mask_loss, n_valid = 0, 0, 0, 0, 0, 0
        with torch.no_grad():
            for i_batch, data in enumerate(self.valid_loader):
                img_batch, tempo_seg_pos, tempo_seg_neg, keyword_batch, sentence_batch = map(lambda x: x.to(self.args.device), data)
                outputs = self.model(img_batch, tempo_seg_pos, tempo_seg_neg, keyword_batch, sentence_batch, gated_mask=self.args.gated_mask)
                pred_score, gt_score, pred_offsets, gt_offsets, pred_sent, gt_sent, mask_loss = outputs
                cls_loss = self.module.bce_loss(pred_score, gt_score)
                reg_loss = self.module.reg_loss(pred_offsets, gt_offsets)
                sent_loss = self.module.ce_loss(pred_sent, gt_sent)
                total_loss = cls_loss * self.args.cls_weight + reg_loss * self.args.reg_weight + sent_loss * self.args.sent_weight
                if mask_loss is not None:
                    total_loss += mask_loss * self.args.mask_weight
                else:
                    mask_loss = cls_loss.new_empty(1).fill_(0)
                valid_loss += total_loss.item() * img_batch.size(0)
                val_cls_loss += cls_loss.item() * img_batch.size(0)
                val_reg_loss += reg_loss.item() * img_batch.size(0)
                val_sent_loss += sent_loss.item() * img_batch.size(0)
                val_mask_loss += mask_loss.item() * img_batch.size(0)
                n_valid += img_batch.size(0)
                print('iter: [{}/{}], valid loss: {:.4f}, class: {:.4f}, reg: {:.4f}, sent: {:.4f}, mask: {:.4f}'.format(
                    i_batch, n_batch, total_loss.item(), cls_loss.item(), reg_loss.item(), sent_loss.item(), mask_loss.item()), end='\r')
            print(' ' * 100, end='\r')
        return valid_loss / n_valid, val_cls_loss / n_valid, val_reg_loss / n_valid, val_sent_loss / n_valid, val_mask_loss / n_valid

    def run(self):
        # filter params that don't require gradient (credit: PyTorch Forum issue 679)
        # smaller learning rate for the decoder
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        if args.optim == 'adam':
            optimizer = torch.optim.Adam(_params,
                                         self.args.learning_rate,
                                         betas=(self.args.alpha, self.args.beta),
                                         eps=self.args.epsilon)
        elif args.optim == 'sgd':
            optimizer = torch.optim.SGD(_params,
                                        self.args.learning_rate,
                                        weight_decay=1e-5,
                                        momentum=self.args.alpha,
                                        nesterov=True)
        else:
            raise NotImplementedError
        # learning rate decay every 1 epoch
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               factor=self.args.reduce_factor,
                                                               patience=self.args.patience_epoch,
                                                               verbose=True)
        # scheduler = lr_scheduler.ExponentialLR(optimizer, 0.6)
        # Number of parameter blocks in the network
        self.logger.info("# of param blocks: {}".format(str(len(list(self.model.parameters())))))
        best_loss = float('inf')
        all_val_losses, all_cls_losses, all_reg_losses, all_sent_losses, all_mask_losses, all_train_losses = [], [], [], [], [], []
        for train_epoch in range(self.args.num_epoch):
            t_epoch_start = time.time()
            self.logger.info('Epoch: {}'.format(train_epoch))
            if self.args.distributed:
                self.train_sampler.set_epoch(train_epoch)
            epoch_loss = self._train(train_epoch, optimizer)
            all_train_losses.append(epoch_loss)
            valid_losses = self._validate()
            valid_loss, val_cls_loss, val_reg_loss, val_sent_loss, val_mask_loss = valid_losses
            all_val_losses.append(valid_loss)
            all_cls_losses.append(val_cls_loss)
            all_reg_losses.append(val_reg_loss)
            all_sent_losses.append(val_sent_loss)
            all_mask_losses.append(val_mask_loss)
            if valid_loss < best_loss:
                best_loss = valid_loss
                if (self.args.distributed and torch.distributed.get_rank() == 0) or not self.args.distributed:
                    torch.save(self.module.state_dict(), os.path.join('state_dict', '{}.pt'.format(self.args.timestamp)))
                self.logger.info('Better validation loss {:.4f} found, save model'.format(valid_loss))
            # save eval and train losses
            if (self.args.distributed and torch.distributed.get_rank() == 0) or not self.args.distributed:
                torch.save({
                    'train_loss': all_train_losses,
                    'val_loss': all_val_losses,
                    'val_cls_loss': all_cls_losses,
                    'val_reg_loss': all_reg_losses,
                    'val_sent_loss': all_sent_losses,
                    'val_mask_loss': all_mask_losses
                }, os.path.join('state_dict', '{}.loss'.format(self.args.timestamp)))
            # learning rate decay
            scheduler.step(valid_loss)
            self.logger.info('Train loss: {:.4f}, val loss: {:.4f}, Time: {:.4f}s'.format(
                epoch_loss, valid_loss, time.time() - t_epoch_start))
            self.logger.info('valid cls loss: {:.4f}, reg loss: {:.4f}, sent loss: {:.4f}, mask loss: {:.4f}'.format(
                val_cls_loss, val_reg_loss, val_sent_loss, val_mask_loss))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    ''' Dataset '''
    parser.add_argument('--data_root', default='data', type=str)
    parser.add_argument('--dataset_file', default='anet_annotations_trainval.json', type=str)
    parser.add_argument('--feature_root', default='feature', type=str, help='the feature root')
    parser.add_argument('--dur_file', default='anet_duration_frame_feature.csv', type=str)
    parser.add_argument('--train_data_folder', default='training', type=str, help='training data folder')
    parser.add_argument('--val_data_folder', default='validation', type=str, help='validation data folder')
    parser.add_argument('--vocab_file', default='vocab.txt', type=str, help='vocabulary file')
    parser.add_argument('--save_vocab', default=False, action='store_true')
    parser.add_argument('--load_vocab', default=False, action='store_true')
    parser.add_argument('--vocab_path', default='vocab.pkl', type=str)
    parser.add_argument('--save_train_samplelist', default=False, action='store_true')
    parser.add_argument('--load_train_samplelist', default=False, action='store_true')
    parser.add_argument('--train_samplelist_path', default='train_samplelist.pkl', type=str)
    parser.add_argument('--save_valid_samplelist', default=False, action='store_true')
    parser.add_argument('--load_valid_samplelist', default=False, action='store_true')
    parser.add_argument('--valid_samplelist_path', default='valid_samplelist.pkl', type=str)
    parser.add_argument('--checkpoint', default=None, type=str, help='Path to a model checkpoint to initialize model weights from')
    parser.add_argument('--feature_type', default='tsn', type=str, choices=['c3d', 'tsn'], help='Which video feature is used')
    parser.add_argument('--maxlen', default=20, type=int)
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
    parser.add_argument('--train_sample', default=20, type=int, help='Total number of positive+negative training samples (2*U)')
    parser.add_argument('--sample_prob', default=0, type=float, help='Probability for use model samples during training')
    parser.add_argument('--ablation', default=None, type=str, nargs='+', help='Ablation settings')
    ''' Proposal '''
    parser.add_argument('--slide_window_size', default=480, type=int, help='The (temporal) size of the sliding window')
    parser.add_argument('--slide_window_stride', default=20, type=int, help='The step size of the sliding window (not used)')
    parser.add_argument('--sampling_sec', default=0.5, help='Sample frame (RGB and optical flow) with which time interval')
    parser.add_argument('--kernel_list', default=[1, 2, 3, 4, 5, 7, 9, 11, 15, 21, 29, 41, 57, 71, 111, 161, 211, 251], type=int, nargs='+')
    parser.add_argument('--pos_thresh', default=0.7, type=float)
    parser.add_argument('--neg_thresh', default=0.3, type=float)
    parser.add_argument('--stride_factor', default=50, type=int, help='The proposal temporal conv kernel stride is determined by math.ceil(kernel_len/stride_factor)')
    ''' Loss '''
    parser.add_argument('--cls_weight', default=1.0, type=float)
    parser.add_argument('--reg_weight', default=10, type=float)
    parser.add_argument('--sent_weight', default=0.25, type=float)
    parser.add_argument('--mask_weight', default=0.0, type=float)
    parser.add_argument('--use_mask', default=False, action='store_true')
    parser.add_argument('--gated_mask', default=False, action='store_true')
    parser.add_argument('--gt_mask', default=False, action='store_true')
    ''' Optimization '''
    parser.add_argument('--num_epoch', default=20, type=int, help='Max number of epochs to run for')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size per batch in training')
    parser.add_argument('--valid_batch_size', default=64, type=int, help='Batch size per batch in validation')
    parser.add_argument('--optim', default='sgd', help='What update to use? sgd|adam')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='Learning rate')
    parser.add_argument('--alpha', default=0.95, type=float, help='Alpha for adagrad/rmsprop/momentum/adam')
    parser.add_argument('--beta', default=0.999, type=float, help='Beta used for adam')
    parser.add_argument('--epsilon', default=1e-8, help='Epsilon that goes into denominator for smoothing')
    parser.add_argument('--patience_epoch', default=1, type=int, help='Epoch to wait to determine a pateau')
    parser.add_argument('--reduce_factor', default=0.5, type=float, help='Factor of learning rate reduction')
    parser.add_argument('--grad_norm', default=1.0, type=float, help='Gradient clipping norm')
    ''' Environment '''
    parser.add_argument('--dist_url', default='env://', type=str, help='Url used to set up distributed training')
    parser.add_argument('--dist_backend', default='gloo', type=str, help='Distributed backend')
    parser.add_argument('--world_size', default=None, type=int, help='Number of distributed processes')
    parser.add_argument('--seed', default=None, type=int, help='Random number generator seed to use')
    parser.add_argument('--timestamp', default=None, type=str, help='Timestamp of running script')
    parser.add_argument('--device', default=None, type=str, choices=['cpu', 'cuda'], help='Selected device')
    args = parser.parse_args()
    assert args.slide_window_size >= args.slide_window_stride
    assert args.sampling_sec == 0.5 # attention! sampling_sec is hard coded as 0.5
    args.log_name = '{}_train.log'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[2:])
    args.timestamp = args.timestamp if args.timestamp else str(int(time.time())) + format(random.randint(0, 999), '03')
    args.seed = args.seed if args.seed else random.randint(0, 2**32-1)
    args.device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.device.type == 'cuda':
        args.world_size = args.world_size if args.world_size else torch.cuda.device_count()
        args.distributed = args.world_size > 1
    ''' Set seeds '''
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    for dir_name in ['dats', 'logs', 'state_dict']:
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
    ''' Train the model '''
    ins = Instructor(args)
    ins.run()
