import math
import torch
import torch.nn as nn
import numpy as np
from .transformer import Encoder, Transformer


def positional_encodings(x, emb_size):
    """
    Compute positional encodings for given positions.

    Parameters
    ----------
    x : 1d int array (B)
        Batch of positions.
    emb_size : int
        Dimension of the positional encodings.

    Returns
    -------
    encodings : 2d float array (B, H)
        Positional encodings in this batch.
    """
    encodings = torch.zeros(x.size(0), emb_size, device=x.device)
    for channel in range(emb_size):
        if channel % 2 == 0:
            encodings[:, channel] = torch.sin(x / (10000 ** (channel / emb_size)))
        else:
            encodings[:, channel] = torch.cos(x / (10000 ** ((channel - 1) / emb_size)))
    return encodings


class DropoutTime1D(nn.Module):
    """
    Apply 1D-dropout on input data. (Dropout entire 1D feature maps over time dimension)

    Parameters
    ----------
    p_dropout : float
        Probability of 1D feature to be zeroed.

    Inputs
    ------
    x : 3d float array (B, T, H)
        Batched sequences.

    Returns
    -------
    x : 3d float array (B, T, H)
        Batched sequences after 1D-dropout.
    """
    def __init__(self, p_drop):
        super(DropoutTime1D, self).__init__()
        self.p_drop = p_drop

    def forward(self, x):
        if self.training:
            mask = x.data.new_empty((x.data.size(0), x.data.size(1), 1)).uniform_()
            mask = (mask > self.p_drop).float()
            return x * mask
        return x * (1 - self.p_drop)

    def init_params(self):
        pass

    def __repr__(self):
        return self.__class__.__name__ + ' ({:.2f})'.format(self.p_drop)


class TPVC(nn.Module):
    """
    The caption model.

    Parameters
    ----------
    d_model : number of features in encoder/decoder inputs.
    d_hidden : number of features in position-wise feed-forward networks.
    d_emb : number of features of word embeddings.
    n_layers : number of layers in transformer encoder.
    n_heads : number of heads in multihead attention layers.
    vocab : vocabulary for decoding.
    in_emb_dropout : dropout value for input embeddings.
    attn_dropout : dropout value for attention layers.
    cap_dropout : dropout value for transformer decoder.
    nsamples : number of positive+negative training samples
    kernel_list : list of kernel size in proposal decoder.
    stride_factor : proposal conv kernel stride is determined by math.ceil(kernel_len/stride_factor)
    feature_type : type of the video feature (tsn or c3d)
    learn_mask : does caption model learn a mask
    gt_mask : does caption model use the ground-truth mask
    use_mask : does caption model use predicted mask when learn_mask is False
    window_length: temporal size of the sliding window
    no_keyword : does caption model use keyword information
    """
    def __init__(self, d_model, d_hidden, d_emb, n_layers, n_heads, vocab, in_emb_dropout, attn_dropout,
                 cap_dropout, nsamples, kernel_list, stride_factor, feature_type, learn_mask=False,
                 gt_mask=False, use_mask=False, window_length=480, no_keyword=False, ablation=None):
        super(TPVC, self).__init__()

        self.d_model = d_model
        self.kernel_list = kernel_list
        self.stride_factor = stride_factor
        self.nsamples = nsamples
        self.feature_type = feature_type
        self.learn_mask = learn_mask
        self.gt_mask = gt_mask
        self.use_mask = use_mask
        self.no_keyword = no_keyword

        d_prop = d_model + d_emb if not no_keyword else d_model

        self.mask_model = nn.Sequential(
            nn.Linear(d_model + window_length, d_model, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, window_length),
        )

        if self.feature_type == 'tsn':
            self.rgb_emb = nn.Linear(2048, d_model // 2)
            self.flow_emb = nn.Linear(1024, d_model // 2)
        elif self.feature_type == 'c3d':
            self.c3d_emb = nn.Linear(500, d_model)
        self.emb_out = nn.Sequential(
            DropoutTime1D(in_emb_dropout),
            nn.ReLU(inplace=True)
        )

        self.vis_emb = Encoder(d_model, d_hidden, n_layers, n_heads, attn_dropout, ablation)
        self.keyword_emb = nn.Embedding(len(vocab), d_emb, padding_idx=1)

        self.prop_out = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm1d(d_prop),
                nn.Conv1d(d_prop, d_prop, kernel_size=self.kernel_list[i],
                          stride=math.ceil(kernel_list[i]/stride_factor),
                          groups=d_prop, bias=False),
                nn.BatchNorm1d(d_prop),
                nn.Conv1d(d_prop, d_model, kernel_size=1, bias=False),
                nn.BatchNorm1d(d_model),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(d_model),
                nn.Conv1d(d_model, 4, kernel_size=1) # proposal score, overlapping score (DEPRECATED!), length offset, center offset
            ) for i in range(len(self.kernel_list))
        ])

        self.cap_model = Transformer(d_model, self.vis_emb, vocab, d_hidden, n_layers, n_heads, cap_dropout, ablation)

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.reg_loss = nn.SmoothL1Loss()
        self.ce_loss = nn.CrossEntropyLoss()

    @staticmethod
    def segment_iou(target_segment, candidate_segments):
        """
        Compute the temporal intersection over union between a target segment and all the test segments.

        Parameters
        ----------
        target_segment : 1d array
            Temporal target segment containing [starting, ending] times.
        candidate_segments : 2d array
            Temporal candidate segments containing N x [starting, ending] times.

        Returns
        -------
        tiou: 1d array
            Temporal intersection over union score of the N's candidate segments.
        """
        tt1 = torch.maximum(target_segment[:, 0], candidate_segments[:, 0])
        tt2 = torch.minimum(target_segment[:, 1], candidate_segments[:, 1])
        # Intersection including Non-negative overlap score.
        segments_intersection = (tt2 - tt1).clamp(min=0)
        # Segment union.
        segments_union = (candidate_segments[:, 1]-candidate_segments[:, 0])+(target_segment[:, 1]-target_segment[:, 0])-segments_intersection
        # Compute overlap as the ratio of the intersection
        # over union of two segments.
        tIoU = segments_intersection / segments_union
        return tIoU

    def _get_proposal(self, x, keyword, featlen):
        """
        Acquire the proposals.

        Parameters
        ----------
        x : 3d float array (B, T, 2048+1024) or (B, T, 500)
            Batched visual features.
        keyword : 2d array (B, K)
            Batched keyword indexes.
        featlen : int
            Length of the input feature over the time dimension.

        Returns
        -------
        x : 3d float array (B, T, H)
            Embeded visual features.
        prop_all : 3d float array (B, 6, J)
            All the decoded proposals.
        """
        B, T, _ = x.size()
        if self.feature_type == 'tsn':
            # Embed the visual feature (B, T, 2048+1024) -> (B, T, H)
            x_rgb, x_flow = x.split((2048, 1024), dim=-1)
            x_rgb = self.rgb_emb(x_rgb)
            x_flow = self.flow_emb(x_flow)
            x = torch.cat((x_rgb, x_flow), dim=-1)
        elif self.feature_type == 'c3d':
            # Embed the visual feature (B, T, 500) -> (B, T, H)
            x = self.c3d_emb(x)
        x = self.emb_out(x)
        # Feed visual feature to the transformer encoder (B, T, H) -> (B, T, H)
        vis_feat = self.vis_emb(x).transpose(1, 2) # (B, T, H) -> (B, H, T) for 1d conv
        if not self.no_keyword: # embedding keywords
            keyword_len = torch.sum(keyword!=1, dim=1, keepdim=True).float() # (B, 1)
            keyword_feat = self.keyword_emb(keyword) # (B, K, E)
            keyword_pool = torch.sum(keyword_feat, dim=1) / keyword_len # (B, E)
            keyword_pool = keyword_pool.unsqueeze(-1).expand(-1, -1, vis_feat.size(-1)) # (B, E, T)
            vis_feat = torch.cat((vis_feat, keyword_pool), dim=1) # (B, H+E, T)
        # Compute the proposals
        prop_lst = []
        for kernel_size, prop_conv in zip(self.kernel_list, self.prop_out):
            if kernel_size <= featlen:
                pred_o = prop_conv(vis_feat) # (B, 4, T)
                anchor_c = torch.arange(kernel_size/2., T+1-kernel_size/2., math.ceil(kernel_size/self.stride_factor)).type(x.type())
                assert anchor_c.size(0) == pred_o.size(-1), 'anchor & pred size mismatch'
                anchor_c = anchor_c.expand(B, 1, anchor_c.size(0)) # (B, 1, T)
                anchor_l = torch.full_like(anchor_c, kernel_size) # (B, 1, T)
                pred_final = torch.cat((pred_o, anchor_l, anchor_c), dim=1) # (B, 6, T)
                prop_lst.append(pred_final)
            else:
                print('skipping kernel sizes greater than {}'.format(kernel_size) + ' ' * 15, end='\r')
                break
        prop_all = torch.cat(prop_lst, dim=-1) # (B, 6, J)
        return x, prop_all

    def forward(self, x, s_pos, s_neg, keyword, sentence, sample_prob=0, gated_mask=False):
        """
        Train the caption model.

        Inputs
        ------
        x : 3d float array (B, T, 2048+1024) or (B, T, 500)
            Batched visual features.
        s_pos : 3d float array (B, 10, 4)
            Batched positive segments (anchor_idx, overlap, len_offset, cen_offset)
        s_neg : 3d float array (B, 10, 2)
            Batched negative segments (anchor_idx, overlap)
        keyword : 2d array (B, K)
            Batched keyword indexes.
        sentence : 2d array (B, L)
            Batched sentence indexes.
        sample_prob : int
            Probability of sampling in transformer decoder. (default=0)
        gated_mask: bool
            Use gated mask for decoding. (default=False)

        Returns
        -------
        pred_score : Predicted score of anchors.
        gt_score : Ground-truth score of anchors.
        pred_offsets : Predicted offsets (length and center) of anchors.
        gt_offsets : Ground-truth offsets (length and center) of anchors.
        pred_sent : Predicted sentence.
        gt_cent : Ground-truth sentence.
        mask_loss : Loss value of the mask.
        """
        B, T, _ = x.size()
        x, prop_all = self._get_proposal(x, keyword, T)
        sample_each = self.nsamples // 2
        pred_score = x.new_zeros((sample_each * B, 2))
        gt_score = x.new_zeros((sample_each * B, 2))
        pred_offsets = x.new_zeros((sample_each * B, 2))
        gt_offsets = x.new_zeros((sample_each * B, 2))
        pe_locs = x.new_zeros((B * 4,)) # tuple of positional encodings (S_p, E_p, S_a, E_a)
        anchor_window_mask = x.new_zeros((B, T))
        pred_bin_window_mask = x.new_zeros((B, T))
        batch_mask = x.new_zeros((B, T, 1))
        gate_scores = x.new_zeros((B, 1, 1))
        mask_loss = None
        # the maximum length offset is math.log(480) to avoid torch.nan
        pred_len = prop_all[:, 4, :] * torch.exp(prop_all[:, 2, :].clamp(max=7)) # l_a * exp(t_l)
        pred_cen = prop_all[:, 5, :] + prop_all[:, 3, :] * prop_all[:, 4, :] # c_a + t_c * l_a
        pred_index = torch.randint(0, sample_each, (B,)) # randomly choose one of the positive segments to generate caption
        # match the anchors to segments
        for b in range(B):
            pos_anchor = s_pos[b]
            neg_anchor = s_neg[b]
            # random sample anchors from different length
            for i in range(sample_each):
                # sample pos anchors
                pos_sam = pos_anchor[i]
                pos_sam_ind = int(pos_sam[0])
                pred_score[b*sample_each+i, 0] = prop_all[b, 0, pos_sam_ind] # copy the pred score
                gt_score[b*sample_each+i, 0] = 1 # copy the true score
                pred_offsets[b*sample_each+i] = prop_all[b, 2:4, pos_sam_ind] # copy the pred offsets
                gt_offsets[b*sample_each+i] = pos_sam[2:] # copy the true offsets
                # sample neg anchors
                neg_sam = neg_anchor[i]
                neg_sam_ind = int(neg_sam[0])
                pred_score[b*sample_each+i, 1] = prop_all[b, 0, neg_sam_ind] # copy the pred score
                gt_score[b*sample_each+i, 1] = 0 # copy the true score
                # caption the segment, only need once, since one sample corresponds to one sentence only
                if i == pred_index[b]:
                    anc_len = prop_all[b, 4, pos_sam_ind].item()
                    anc_cen = prop_all[b, 5, pos_sam_ind].item()
                    # compute predicted mask
                    pred_cen_i = pred_cen[b, pos_sam_ind].item()
                    pred_len_i = pred_len[b, pos_sam_ind].item()
                    pred_start_w = math.floor(max(0, min(T-1, pred_cen_i - pred_len_i / 2.))) # [0~T-1]
                    pred_end_w = math.ceil(max(1, min(T, pred_cen_i + pred_len_i / 2.))) # [1~T]
                    pred_bin_window_mask[b, pred_start_w:pred_end_w] = 1.
                    if self.gt_mask: # compute ground-truth mask
                        gt_len = anc_len * math.exp(pos_sam[2].item())
                        gt_cen = anc_cen + pos_sam[3].item() * anc_len
                        gt_start_w = max(0, math.floor(gt_cen - gt_len / 2.))
                        gt_end_w = min(T, math.ceil(gt_cen + gt_len / 2.))
                        batch_mask[b, gt_start_w:gt_end_w, :] = 1.
                    elif self.learn_mask: # compute anchor mask
                        anc_start_w = max(0, math.floor(anc_cen - anc_len / 2.))
                        anc_end_w = min(T, math.ceil(anc_cen + anc_len / 2.))
                        anchor_window_mask[b, anc_start_w:anc_end_w] = 1.
                        # generate positions
                        pe_locs[b] = x.new_tensor(pred_start_w) # predicted start
                        pe_locs[B + b] = x.new_tensor(pred_end_w) # predicted end
                        pe_locs[B*2 + b] = x.new_tensor(anc_start_w) # anchor start
                        pe_locs[B*3 + b] = x.new_tensor(anc_end_w) # anchor end
                        gate_scores[b] = pred_score[b*sample_each+i, 0] # copy the pred score
        if self.gt_mask:
            window_mask = batch_mask # (B, T, 1)
        elif self.learn_mask:
            pos_encs = positional_encodings(pe_locs, self.d_model // 4) # (B * 4, H / 4)
            in_pred_mask = torch.cat(pos_encs.split(B, dim=0) + (anchor_window_mask,), dim=-1) # (B, H + T)
            pred_mask = self.mask_model(in_pred_mask) # (B, T)
            if gated_mask:
                gate_scores = torch.sigmoid(gate_scores) # (B, 1, 1)
                window_mask = gate_scores * pred_bin_window_mask.view(B, T, 1) + (1 - gate_scores) * pred_mask.view(B, T, 1)
            else:
                window_mask = pred_mask.view(B, T, 1)
            mask_loss = self.bce_loss(pred_mask, pred_bin_window_mask)
        elif self.use_mask:
            window_mask = pred_bin_window_mask.view(B, T, 1)
        else:
            window_mask = None
        pred_sent, gt_cent = self.cap_model(x, sentence, window_mask, sample_prob)
        return pred_score, gt_score, pred_offsets, gt_offsets, pred_sent, gt_cent, mask_loss


    def inference(self, x, keyword, actual_frame_length, sampling_sec, min_prop_num,
                  max_prop_num, min_prop_num_before_nms, pos_thresh, gated_mask=False):
        """
        Acquire the output for evaluation files.

        Inputs
        ------
        x : 3d array (B, T, 2048+1024)
            Batched visual features.
        keyword : 2d array (B, K)
            Batched keyword indexes.
        actual_frame_length : 1d array (B)
            Number of frames in the videos.
        sampling_sec : float
            Sample frame (RGB and optical flow) with which time interval
        min_prop_num : int
            The maximum number of proposals per video
        max_prop_num : int
            The minimum number of proposals per video
        min_prop_num_before_nms : int
            The minimum number of proposals per video before nms
        pos_thresh : float
            Threshold of positive segments.
        gated_mask : bool
            Use gated mask for decoding. (default=False)

        Returns
        -------
        all_proposal_results : list
            All proposals.
        """
        B, T, _ = x.size()
        x, prop_all = self._get_proposal(x, keyword, actual_frame_length[0])
        # assume 1st and 2nd are action prediction and overlap, respectively
        prop_all[:, :2, :] = torch.sigmoid(prop_all[:, :2, :])
        pred_len = prop_all[0, 4, :] * torch.exp(prop_all[0, 2, :].clamp(max=7)) # l_a * exp(t_l)
        pred_cen = prop_all[0, 5, :] + prop_all[0, 3, :] * prop_all[0, 4, :] # c_a + t_c * l_a
        nms_thresh_set = np.arange(0.9, 0.95, 0.05).tolist()
        # decoded proposals
        pred_start_lst = []
        pred_end_lst = []
        anchor_start_lst = []
        anchor_end_lst = []
        anchor_window_mask = []
        pred_bin_window_mask = []
        gate_scores = []
        # count proposals
        crt_nproposal = 0
        nproposal = torch.sum(torch.gt(prop_all[0, 0, :], pos_thresh))
        nproposal = min(max(nproposal, min_prop_num_before_nms), prop_all.size(-1))
        pred_results = x.new_zeros((nproposal, 3))
        _, sel_idx = torch.topk(prop_all[0][0], nproposal)
        for nms_thresh in nms_thresh_set:
            for prop_idx in range(nproposal):
                # might be truncated at the end, hence + frame_to_second * 2
                original_frame_len = actual_frame_length[0].item() + sampling_sec * 2
                pred_start = pred_cen[sel_idx[prop_idx]] - pred_len[sel_idx[prop_idx]] / 2.
                pred_end = pred_cen[sel_idx[prop_idx]] + pred_len[sel_idx[prop_idx]] / 2.
                if pred_start >= pred_end:
                    continue
                if pred_start < 0 or pred_end >= original_frame_len:
                    continue
                # skip the anchors overlaps others
                hasoverlap = False
                if crt_nproposal > 0:
                    overlap = self.segment_iou(torch.hstack((pred_start, pred_end)).view(1, 2), pred_results[:crt_nproposal])
                    if torch.max(overlap) > nms_thresh:
                        hasoverlap = True
                if not hasoverlap:
                    pred_score = prop_all[0, 0, sel_idx[prop_idx]]
                    pred_bin_mask = x.new_zeros((1, T, 1))
                    pred_start_w = math.floor(max(min(pred_start.item(), min(original_frame_len, T)-1), 0)) # [0~min(len, T)-1]
                    pred_end_w = math.ceil(max(min(pred_end.item(), min(original_frame_len, T)), 1)) # [1~min(len, T)]
                    pred_bin_mask[:, pred_start_w:pred_end_w] = 1
                    pred_bin_window_mask.append(pred_bin_mask)
                    if self.learn_mask:
                        anc_len = prop_all[0, 4, sel_idx[prop_idx]].item()
                        anc_cen = prop_all[0, 5, sel_idx[prop_idx]].item()
                        anchor_mask = x.new_zeros((1, T))
                        anc_start_w = max(0, math.floor(anc_cen - anc_len / 2.))
                        anc_end_w = min(T, math.ceil(anc_cen + anc_len / 2.))
                        anchor_mask[0, anc_start_w:anc_end_w] = 1.
                        anchor_window_mask.append(anchor_mask)
                        pred_start_lst.append(x.new_tensor(pred_start_w))
                        pred_end_lst.append(x.new_tensor(pred_end_w))
                        anchor_start_lst.append(x.new_tensor(anc_start_w))
                        anchor_end_lst.append(x.new_tensor(anc_end_w))
                        gate_scores.append(pred_score)
                    pred_results[crt_nproposal] = torch.hstack((pred_start, pred_end, pred_score))
                    crt_nproposal += 1
                if crt_nproposal >= max_prop_num:
                    break
            if crt_nproposal >= min_prop_num:
                break
        has_proposal = True
        if len(pred_bin_window_mask) == 0: # append all-one window if no window is proposed
            pred_bin_window_mask.append(x.new_ones((1, T, 1)))
            pred_results[0] = x.new_tensor([0, min(original_frame_len, T), pos_thresh])
            crt_nproposal = 1
            has_proposal = False
        pred_bin_window_mask = torch.cat(pred_bin_window_mask, dim=0) # (P, T, 1)
        batch_x = x.expand(pred_bin_window_mask.size(0), -1, -1) # (P, T, H)
        if has_proposal and self.learn_mask:
            pe_pred_start = torch.hstack(pred_start_lst) # (P)
            pe_pred_end = torch.hstack(pred_end_lst) # (P)
            pe_anchor_start = torch.hstack(anchor_start_lst) # (P)
            pe_anchor_end = torch.hstack(anchor_end_lst) # (P)
            pe_locs = torch.cat((pe_pred_start, pe_pred_end, pe_anchor_start, pe_anchor_end), dim=0) # (P * 4)
            pos_encs = positional_encodings(pe_locs, self.d_model // 4) # (P * 4, H / 4)
            anchor_window_mask = torch.cat(anchor_window_mask, dim=0) # (P, T)
            in_pred_mask = torch.cat(pos_encs.split(crt_nproposal, dim=0) + (anchor_window_mask,), dim=-1) # (B, H + T)
            pred_mask = self.mask_model(in_pred_mask).unsqueeze(-1) # (B, T, 1)
            if gated_mask:
                gate_scores = torch.hstack(gate_scores).view(-1, 1, 1)
                window_mask = (gate_scores * pred_bin_window_mask + (1 - gate_scores) * pred_mask)
            else:
                window_mask = pred_mask
        elif self.use_mask:
            window_mask = pred_bin_window_mask
        else:
            window_mask = batch_x.new_ones((batch_x.size(0), batch_x.size(1), 1))
        # predict sentence
        pred_sentence = []
        cap_batch = math.ceil(480 * 256 / T) # use cap_batch as caption batch size
        for sent_i in range(math.ceil(batch_x.size(0) / cap_batch)):
            batch_start = sent_i * cap_batch
            batch_end = min((sent_i+1) * cap_batch, batch_x.size(0))
            pred_sentence += self.cap_model.greedy(batch_x[batch_start:batch_end], window_mask[batch_start:batch_end], 20)
        pred_results = pred_results[:crt_nproposal]
        assert len(pred_sentence) == crt_nproposal, 'number of predicted sentence and proposal does not match'
        return [(pred_results[i][0].item(), pred_results[i][1].item(), pred_results[i][2].item(), pred_sentence[i]) for i in range(crt_nproposal)]
