import json
import random
import string
import numpy as np
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.spice.spice import Spice


def random_string(string_length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(string_length))


def remove_nonascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])


class ANETcaptions:

    def __init__(self, ground_truth_filenames, prediction_filename, tious=None, max_proposals=1000,
                 prediction_fields=['results', 'version', 'external_data'], verbose=False, logger=None):
        # Check that the gt and submission files exist and load them
        if len(tious) == 0:
            raise IOError('Please input a valid tIoU.')
        if not ground_truth_filenames:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        self.logger = logger
        self.verbose = verbose
        self.tious = tious
        self.max_proposals = max_proposals
        self.pred_fields = prediction_fields
        self.ground_truths = self.import_ground_truths(ground_truth_filenames)
        self.prediction = self.import_prediction(prediction_filename)
        ground_truths_keys = [vid for gt in self.ground_truths for vid in gt]
        logger.info('available video number: {}'.format(len(set(ground_truths_keys) & set(self.prediction.keys()))))
        self.tokenizer = PTBTokenizer()
        # Set up scorers
        self.scorers = [
            (Bleu(4), ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']),
            (Meteor(), 'METEOR'),
            (Rouge(), 'ROUGE_L'),
            (Cider('corpus'), 'CIDEr'),
        ]

    def import_prediction(self, prediction_filename):
        if self.verbose:
            self.logger.info('=> Loading submission from {}'.format(prediction_filename))
        with open(prediction_filename, 'r') as f:
            submission = json.load(f)
        if not all([field in submission.keys() for field in self.pred_fields]):
            raise IOError('Please input a valid ground truth file.')
        # Ensure that every video is limited to the correct maximum number of proposals.
        results = {}
        for vid, val in submission['results'].items():
            results[vid] = {}
            for keyword, proposals in val.items():
                results[vid][keyword] = proposals[:self.max_proposals]
        return results

    def import_ground_truths(self, filenames):
        if self.verbose:
            self.logger.info('=> Loading GT from {}'.format(filenames))
        gts = []
        for filename in filenames:
            with open(filename, 'r') as f:
                gts.append(json.load(f))
        return gts

    def get_gt_vids(self):
        vids = set()
        for gt in self.ground_truths:
            vids.update(gt.keys())
        return vids

    def iou(self, interval_1, interval_2):
        start_i, end_i = interval_1[0], interval_1[1]
        start, end = interval_2[0], interval_2[1]
        intersection = max(0, min(end, end_i) - max(start, start_i))
        union = min(max(end, end_i) - min(start, start_i), end-start + end_i-start_i)
        iou = float(intersection) / (union + 1e-8)
        return iou

    def evaluate(self):
        self.scores = {}
        for tiou in self.tious:
            self.logger.info('evaluating for {} IoU...'.format(tiou))
            scores = self.evaluate_tiou(tiou)
            for metric, score in scores.items():
                if metric not in self.scores:
                    self.scores[metric] = []
                self.scores[metric].append(score)
        if self.verbose:
            self.scores['Recall'] = []
            self.scores['Precision'] = []
            for tiou in self.tious:
                precision, recall = self.evaluate_detection(tiou)
                self.scores['Recall'].append(recall)
                self.scores['Precision'].append(precision)

    def evaluate_detection(self, tiou):
        gt_vids = self.get_gt_vids()
        # Recall is the percentage of ground truth that is covered by the predictions
        # Precision is the percentage of predictions that are valid
        recall = []
        precision = []
        for vid in gt_vids:
            keywords = set()
            for gt in self.ground_truths:
                if vid in gt:
                    keywords.update(gt[vid]['keywords'].keys())
            for keyword in keywords:
                best_recall = 0
                best_precision = 0
                for gt in self.ground_truths:
                    if vid not in gt or keyword not in gt[vid]['keywords']:
                        continue
                    refs = gt[vid]['keywords'][keyword]
                    ref_set_covered = set([])
                    pred_set_covered = set([])
                    if vid in self.prediction and keyword in self.prediction[vid]:
                        for pred_i, pred in enumerate(self.prediction[vid][keyword]):
                            pred_timestamp = pred['timestamp']
                            for ref_i, ref_timestamp in enumerate(refs['timestamps']):
                                if self.iou(pred_timestamp, ref_timestamp) > tiou:
                                    ref_set_covered.add(ref_i)
                                    pred_set_covered.add(pred_i)
                        new_precision = float(len(pred_set_covered)) / (pred_i + 1) 
                        best_precision = max(best_precision, new_precision)
                    new_recall = float(len(ref_set_covered)) / len(refs['timestamps'])
                    best_recall = max(best_recall, new_recall)
                    recall.append(best_recall)
                    precision.append(best_precision)
        return sum(precision) / len(precision), sum(recall) / len(recall)

    def evaluate_tiou(self, tiou):
        # This method averages the tIoU precision from METEOR, Bleu, etc. across videos 
        gt_vids = self.get_gt_vids()
        unique_index = 0
        vid_keyword2capid = {} # video id with keyword to unique caption ids mapping
        cur_res = {}
        cur_gts = {}
        for vid in gt_vids:
            keywords = set()
            for gt in self.ground_truths:
                if vid in gt:
                    keywords.update(gt[vid]['keywords'].keys())
            for keyword in keywords:
                vid_keyword2capid[vid+keyword] = []
                if vid not in self.prediction or keyword not in self.prediction[vid]:
                    continue
                for pred in self.prediction[vid][keyword]:
                    has_added = False
                    for gt in self.ground_truths:
                        if vid not in gt or keyword not in gt[vid]['keywords']:
                            continue
                        gt_captions = gt[vid]['keywords'][keyword]
                        for caption_idx, caption_timestamp in enumerate(gt_captions['timestamps']):
                            if self.iou(pred['timestamp'], caption_timestamp) >= tiou:
                                cur_res[unique_index] = [{'caption': remove_nonascii(pred['sentence'])}]
                                cur_gts[unique_index] = [{'caption': remove_nonascii(gt_captions['sentences'][caption_idx])}]
                                vid_keyword2capid[vid+keyword].append(unique_index)
                                unique_index += 1
                                has_added = True
                # If the predicted caption does not overlap with any ground truth,
                # we should compare it with garbage
                # Note: should it be included in the inner loop?
                if not has_added:
                    cur_res[unique_index] = [{'caption': remove_nonascii(pred['sentence'])}]
                    cur_gts[unique_index] = [{'caption': random_string(random.randint(10, 20))}]
                    vid_keyword2capid[vid+keyword].append(unique_index)
                    unique_index += 1
        # call tokenizer here for all predictions and gts
        tokenize_res = self.tokenizer.tokenize(cur_res)
        tokenize_gts = self.tokenizer.tokenize(cur_gts)
        # reshape back
        res = {}
        gts = {}
        for vid_keyword in vid_keyword2capid.keys():
            res[vid_keyword] = {index: tokenize_res[index] for index in vid_keyword2capid[vid_keyword]}
            gts[vid_keyword] = {index: tokenize_gts[index] for index in vid_keyword2capid[vid_keyword]}
        # Each scorer will compute across all videos and take average score
        output = {}
        for scorer, method in self.scorers:
            self.logger.info('computing {} score...'.format(scorer.method()))
            # For each video, take all the valid pairs (based from tIoU) and compute the score
            all_scores = {}
            for vid in gt_vids:
                keywords = set()
                for gt in self.ground_truths:
                    if vid in gt:
                        keywords.update(gt[vid]['keywords'].keys())
                for keyword in keywords:
                    if len(res[vid+keyword]) == 0 or len(gts[vid+keyword]) == 0:
                        if type(method) == list:
                            score = [0] * len(method)
                        else:
                            score = 0
                    else:
                        if type(method) == list: # For BLEU Scorer
                            score, scores = scorer.compute_score(gts[vid+keyword], res[vid+keyword], verbose=0)
                        else:
                            score, scores = scorer.compute_score(gts[vid+keyword], res[vid+keyword])
                    all_scores[vid+keyword] = score
            if type(method) == list: # For BLEU Scorer
                scores = np.mean(list(all_scores.values()), axis=0)
                for m in range(len(method)):
                    output[method[m]] = scores[m]
                    if self.verbose:
                        self.logger.info("Calculated tIoU: {:%1.1f}, {:%s}: {:%0.3f}".format(tiou, method[m], output[method[m]]))
            else:
                output[method] = np.mean(list(all_scores.values()))
                if self.verbose:
                    self.logger.info("Calculated tIoU: {:%1.1f}, {:%s}: {:%0.3f}".format(tiou, method, output[method]))
        return output
