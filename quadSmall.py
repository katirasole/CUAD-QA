#!/usr/bin/env python
# coding: utf-8

# # Question Answering on CUAD

# # Setup - Install Transformer and Tensorflow

# In[ ]:


import tensorflow as tf
print(tf.__version__)

import transformers
print(transformers.__version__)


# In[4]:


import argparse
import glob
import logging
from logging import basicConfig


import logging
import requests
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger('requests').setLevel(logging.DEBUG)


import os
import random
import timeit
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from tqdm import tqdm, trange

import transformers
from transformers import (
                          MODEL_FOR_QUESTION_ANSWERING_MAPPING,
                          WEIGHTS_NAME,
                          AdamW,
                          AutoConfig,
                          AutoModelForQuestionAnswering,
                          AutoTokenizer,
                          get_linear_schedule_with_warmup,
                          squad_convert_examples_to_features,
                          )

from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from transformers.trainer_utils import is_main_process

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


# In[5]:


#From Utils.py
import collections
import json
import math
import re
import string
import json

from transformers.models.bert import BasicTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)


# # Load CUAD Dataset 

# In[ ]:


#Run the code with whole set 
#train_file = 'train_separate_questions.json'
#predict_file = 'test.json'


# In[9]:


#Run the code with small set 
train_file = 'train_separate_questions_1.json'
predict_file = 'test_1.json'

#Run the code with big set 
#train_file = 'train_separate_questions.json'
#predict_file = 'test.json'

# In[8]:


output_dir = 'train_models/roberta-large'


# # UTILS

# In[11]:


def reformat_predicted_string(remaining_contract, predicted_string):
    tokens = predicted_string.split()
    assert len(tokens) > 0
    end_idx = 0
    for i, token in enumerate(tokens):
        found = remaining_contract[end_idx:].find(token)
        assert found != -1
        end_idx += found
        if i == 0:
            start_idx = end_idx
    end_idx += len(tokens[-1])
    return remaining_contract[start_idx:end_idx]


#----------------------------------------------------------------------------------
def find_char_start_idx(contract, preceeding_tokens, predicted_string):
    contract = " ".join(contract.split())
    assert predicted_string in contract
    if contract.count(predicted_string) == 1:
        return contract.find(predicted_string)

    start_idx = 0
    for token in preceeding_tokens:
        found = contract[start_idx:].find(token)
        assert found != -1
        start_idx += found
    start_idx += len(preceeding_tokens[-1])
    remaining_str = contract[start_idx:]

    remaining_idx = remaining_str.find(predicted_string)
    assert remaining_idx != -1

    return start_idx + remaining_idx


#----------------------------------------------------------------------------------
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


#----------------------------------------------------------------------------------
def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

#----------------------------------------------------------------------------------
def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

#----------------------------------------------------------------------------------
def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

#----------------------------------------------------------------------------------
def get_raw_scores(examples, preds):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    exact_scores = {}
    f1_scores = {}

    for example in examples:
        qas_id = example.qas_id
        gold_answers = [answer["text"] for answer in example.answers if normalize_answer(answer["text"])]

        if not gold_answers:
            # For unanswerable questions, only correct answer is empty string
            gold_answers = [""]

        if qas_id not in preds:
            print("Missing prediction for %s" % qas_id)
            continue

        prediction = preds[qas_id]
        exact_scores[qas_id] = max(compute_exact(a, prediction) for a in gold_answers)
        f1_scores[qas_id] = max(compute_f1(a, prediction) for a in gold_answers)

    return exact_scores, f1_scores

#----------------------------------------------------------------------------------
def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores

#----------------------------------------------------------------------------------
def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores.values()) / total),
                ("f1", 100.0 * sum(f1_scores.values()) / total),
                ("total", total),
            ]
        )
    else:
        total = len(qid_list)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
                ("f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
                ("total", total),
            ]
        )

#----------------------------------------------------------------------------------
def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval["%s_%s" % (prefix, k)] = new_eval[k]


#----------------------------------------------------------------------------------
def find_best_thresh_v2(preds, scores, na_probs, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for i, qid in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]

    has_ans_score, has_ans_cnt = 0, 0
    for qid in qid_list:
        if not qid_to_has_ans[qid]:
            continue
        has_ans_cnt += 1

        if qid not in scores:
            continue
        has_ans_score += scores[qid]

    return 100.0 * best_score / len(scores), best_thresh, 1.0 * has_ans_score / has_ans_cnt


#----------------------------------------------------------------------------------
def find_all_best_thresh_v2(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    best_exact, exact_thresh, has_ans_exact = find_best_thresh_v2(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh, has_ans_f1 = find_best_thresh_v2(preds, f1_raw, na_probs, qid_to_has_ans)
    # NOTE: For CUAD, which is about finding needles in haystacks and for which different answers should be treated
    # differently, these metrics don't make complete sense. We ignore them, but don't remove them for simplicity.
    main_eval["best_exact"] = best_exact
    main_eval["best_exact_thresh"] = exact_thresh
    main_eval["best_f1"] = best_f1
    main_eval["best_f1_thresh"] = f1_thresh
    main_eval["has_ans_exact"] = has_ans_exact
    main_eval["has_ans_f1"] = has_ans_f1

#----------------------------------------------------------------------------------
def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for _, qid in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]
    return 100.0 * best_score / len(scores), best_thresh

#----------------------------------------------------------------------------------
def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)

    main_eval["best_exact"] = best_exact
    main_eval["best_exact_thresh"] = exact_thresh
    main_eval["best_f1"] = best_f1
    main_eval["best_f1_thresh"] = f1_thresh

#----------------------------------------------------------------------------------
def squad_evaluate(examples, preds, no_answer_probs=None, no_answer_probability_threshold=1.0):
    qas_id_to_has_answer = {example.qas_id: bool(example.answers) for example in examples}
    has_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if has_answer]
    no_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if not has_answer]

    if no_answer_probs is None:
        no_answer_probs = {k: 0.0 for k in preds}

    exact, f1 = get_raw_scores(examples, preds)

    exact_threshold = apply_no_ans_threshold(
        exact, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold
    )
    f1_threshold = apply_no_ans_threshold(f1, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold)

    evaluation = make_eval_dict(exact_threshold, f1_threshold)

    if has_answer_qids:
        has_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=has_answer_qids)
        merge_eval(evaluation, has_ans_eval, "HasAns")

    if no_answer_qids:
        no_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=no_answer_qids)
        merge_eval(evaluation, no_ans_eval, "NoAns")

    if no_answer_probs:
        find_all_best_thresh(evaluation, preds, exact, f1, no_answer_probs, qas_id_to_has_answer)

    return evaluation

#----------------------------------------------------------------------------------
def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'", orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position : (orig_end_position + 1)]
    return output_text

#----------------------------------------------------------------------------------
def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

#----------------------------------------------------------------------------------
def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

#----------------------------------------------------------------------------------
def compute_predictions_logits(
    json_input_dict,
    all_examples,
    all_features,
    all_results,
    n_best_size,
    max_answer_length,
    do_lower_case,
    output_prediction_file,
    output_nbest_file,
    output_null_log_odds_file,
    verbose_logging,
    version_2_with_negative,
    null_score_diff_threshold,
    tokenizer,):
  
    """Write final predictions to the json file and log-odds of null if needed."""
    if output_prediction_file:
      logger.info(f"Writing predictions to: {output_prediction_file}")
    if output_nbest_file:
      logger.info(f"Writing nbest to: {output_nbest_file}")
    if output_null_log_odds_file and version_2_with_negative:
      logger.info(f"Writing null_log_odds to: {output_null_log_odds_file}")

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
    )

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    contract_name_to_idx = {}
    for idx in range(len(json_input_dict["data"])):
        contract_name_to_idx[json_input_dict["data"][idx]["title"]] = idx

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        contract_name = example.title
        contract_index = contract_name_to_idx[contract_name]
        paragraphs = json_input_dict["data"][contract_index]["paragraphs"]
        assert len(paragraphs) == 1

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                        )
                    )
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                )
            )
        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"]
        )

        seen_predictions = {}
        nbest = []
        start_indexes = []
        end_indexes = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index : (pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start : (orig_doc_end + 1)]

                tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)

                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True

                start_indexes.append(orig_doc_start)
                end_indexes.append(orig_doc_end)
            else:
                final_text = ""
                seen_predictions[final_text] = True

                start_indexes.append(-1)
                end_indexes.append(-1)

            nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit))

        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(_NbestPrediction(text="", start_logit=null_start_logit, end_logit=null_end_logit))
                start_indexes.append(-1)
                end_indexes.append(-1)

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))
                start_indexes.append(-1)
                end_indexes.append(-1)

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))
            start_indexes.append(-1)
            end_indexes.append(-1)

        assert len(nbest) >= 1, "No valid predictions"
        assert len(nbest) == len(start_indexes), "nbest length: {}, start_indexes length: {}".format(len(nbest), len(start_indexes))

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            output["token_doc_start"] = start_indexes[i]
            output["token_doc_end"] = end_indexes[i]
            nbest_json.append(output)

        assert len(nbest_json) >= 1, "No valid predictions"

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json

    if output_prediction_file:
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")

    if output_nbest_file:
        with open(output_nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if output_null_log_odds_file and version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions


# # Define Parameters

# Set arguments:
# 
# **output_dir**  
# 
# /content/gdrive/My Drive/UPWORK/SQuAD-CUAD-QA/CUAD/train_models/roberta-base \
# **model_type** roberta \
# **model_name_or_path** roberta-base \
# **train_file** ./data/train_separate_questions.json \
# **predict_file** ./data/test.json \
# **do_train** \
# **do_eval** \
# **version_2_with_negative** \
# **learning_rate** 1e-4 \
# **num_train_epochs** 4 \
# **per_gpu_eval_batch_size**=40  \
# **per_gpu_train_batch_size**=40 \
# **max_seq_length** 512 \
# **max_answer_length** 512 \
# **doc_stride** 256 \
# **save_steps** 1000 \
# **n_best_size** 20 \
# **overwrite_output_dir**
# 

# In[12]:


# ------------Required parameters-------------------------
model_type = 'roberta'                             #default=None
model_name_or_path = 'roberta-large'                      #default=None
#output_dir = 'train_models/roberta-large'
  
# -------------Other parameters----------------------------
data_dir = None                              #default=None, "The input data dir. Should contain the .json files for the task."
do_train = True                                #action="store_true", "Whether to run training."
do_eval =  True                                #action="store_true", "Whether to run eval on the dev set."
version_2_with_negative = True                 #action="store_true", "If true, the SQuAD examples contain some that do not have an answer."
learning_rate = 1e-4                          #default=5e-5, "The initial learning rate for Adam."
num_train_epochs = 4                             #default=1.0, "Total number of training epochs to perform."
per_gpu_train_batch_size = 40                #default=8, "Batch size per GPU/CPU for training."
per_gpu_eval_batch_size = 40                 #default=8, "Batch size per GPU/CPU for evaluation."
max_seq_length = 128                          #default=384, "The maximum total input sequence length after WordPiece tokenization. Sequences "
max_answer_length = 128                            #default=30, "The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.",
doc_stride = 256                              #default=128, "When splitting up a long document into chunks, how much stride to take between chunks."
save_steps = 1000                             #default=500, "Save checkpoint every X updates steps."
n_best_size = 20                           #default=20, "The total number of n-best predictions to generate in the nbest_predictions.json output file.",
overwrite_output_dir = True                             #action="store_true", "Overwrite the content of the output directory"

#*****************************************************************************
config_name = ""                             #default="", "Pretrained config name or path if not the same as model_name"
tokenizer_name = ""                           #default="", "Pretrained tokenizer name or path if not the same as model_name"
cache_dir = ""                                #default="", "Where do you want to store the pre-trained models downloaded from huggingface.co"
null_score_diff_threshold = 0.0                #default=0.0 , "If null_score - best_non_null is greater than the threshold predict null."                                                        #"longer than this will be truncated, and sequences shorter than this will be padded."
max_query_length = 64                        #default=64, "The maximum number of tokens for the question. Questions longer than this will be truncated to this length.",  
evaluate_during_training = True                #action="store_true", "Run evaluation during training at each logging step."
do_lower_case = True                          #action="store_true", "Set this flag if you are using an uncased model."
gradient_accumulation_steps = 1             #default=1, "Number of updates steps to accumulate before performing a backward/update pass.",
weight_decay = 0.0                            #default=0.0, "Weight decay if we apply some."
adam_epsilon = 1e-8                            #default=1e-8, "Epsilon for Adam optimizer."
max_grad_norm = 1.0                            #default=1.0, "Max gradient norm."
max_steps = -1                            #default=-1, "If > 0: set total number of training steps to perform. Override num_train_epochs.",
warmup_steps = 0                            #default=0, "Linear warmup over warmup_steps."
verbose_logging = True                            #action="store_true", "If true, all of the warnings related to data processing will be printed. A number of warnings are expected for a normal SQuAD evaluation.",
lang_id = 0                            #default=0, language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)"
logging_steps = 500                             #default=500, "Log every X updates steps."
eval_all_checkpoints = True                             #action="store_true", "Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
no_cuda = False                             #action="store_true", "Whether not to use CUDA when available"
overwrite_cache = True                             #action="store_true", "Overwrite the cached training and evaluation sets"
seed = 42                              #default=42, "random seed for initialization"
local_rank = -1                              #default=-1, "local_rank for distributed training on gpus"
fp16 = False                              #action="store_true", "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
fp16_opt_level = 'O1'                              #default="O1", "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details at https://nvidia.github.io/apex/amp.html",
server_ip = ""                                #default="", "Can be used for distant debugging."
server_port = ""                                #default="", "Can be used for distant debugging."
threads = 1                                #default=1, "multiple threads for converting example to features"
keep_frac = 1.0                               #default=1.0, "The fraction of the balanced dataset to keep."

#***************************************************************************************************
#deifne args dictionary
args = {
        'model_type' : model_type,                             
        'model_name_or_path' : model_name_or_path,                      
        'output_dir' : output_dir,                           
        
        # Other parameters
        'data_dir' : data_dir,
        'train_file' :  train_file,                            
        'predict_file' : predict_file,                             
        'do_train' : do_train,                                
        'do_eval' : do_eval,                                 
        'version_2_with_negative' :  version_2_with_negative,                
        'learning_rate' :  learning_rate,                         
        'num_train_epochs' : num_train_epochs,                        
        'per_gpu_train_batch_size' : per_gpu_train_batch_size,                
        'per_gpu_eval_batch_size' : per_gpu_eval_batch_size,                 
        'max_seq_length' : max_seq_length,                        
        'max_answer_length' : max_answer_length,                      
        'doc_stride' : doc_stride,                             
        'save_steps' : save_steps,                            
        'n_best_size' : n_best_size,                           
        'overwrite_output_dir' : overwrite_output_dir,                

        #*****************************************************************************
        'config_name' : config_name,                           
        'tokenizer_name' : tokenizer_name,                        
        'cache_dir' : cache_dir,                            
        'null_score_diff_threshold' : null_score_diff_threshold,            
        'max_query_length' :  max_query_length,                     
        'evaluate_during_training' :  evaluate_during_training,           
        'do_lower_case' : do_lower_case,                       
        'gradient_accumulation_steps' : gradient_accumulation_steps,            
        'weight_decay' : weight_decay,                         
        'adam_epsilon' : adam_epsilon,                        
        'max_grad_norm' : max_grad_norm,                        
        'max_steps' : max_steps,                           
        'warmup_steps' : warmup_steps,                         
        'verbose_logging' : verbose_logging,                    
        'lang_id' : lang_id,                           
        'logging_steps' : logging_steps,                   
        'eval_all_checkpoints' : eval_all_checkpoints,           
        'no_cuda' : no_cuda,                       
        'overwrite_cache' : overwrite_cache,                
        'seed' : seed,                             
        'local_rank' : local_rank,
        'fp16' : fp16,                              
        'fp16_opt_level' : fp16_opt_level,                              
        'server_ip' : server_ip,                        
        'server_port' : server_port,                      
        'threads' : threads,                          
        'keep_frac' : keep_frac
  }


# In[13]:


def set_seed(args):
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    if args["n_gpu"] > 0:
        torch.cuda.manual_seed_all(args["seed"])


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def get_dataset_pos_mask(dataset):
    """
    Returns a list, pos_mask, where pos_mask[i] indicates is True if the ith example in the dataset is positive
    (i.e. it contains some text that should be highlighted) and False otherwise.
    """
    pos_mask = []
    for i in range(len(dataset)):
        ex = dataset[i]
        start_pos = ex[3]
        end_pos = ex[4]
        is_positive = end_pos > start_pos
        pos_mask.append(is_positive)
    return pos_mask


def get_random_subset(dataset, keep_frac=1):
    """
    Takes a random subset of dataset, where a keep_frac fraction is kept.
    """
    keep_indices = [i for i in range(len(dataset)) if np.random.random() < keep_frac]
    subset_dataset = torch.utils.data.Subset(dataset, keep_indices)
    return subset_dataset


def get_balanced_dataset(dataset):
    """
    returns a new dataset, where positive and negative examples are approximately balanced
    """
    pos_mask = get_dataset_pos_mask(dataset)
    neg_mask = [~mask for mask in pos_mask]
    npos, nneg = np.sum(pos_mask), np.sum(neg_mask)

    neg_keep_frac = npos / nneg  # So that in expectation there will be npos negative examples (--> balanced)
    neg_keep_mask = [mask and np.random.random() < neg_keep_frac for mask in neg_mask]

    # keep all positive examples and subset of negative examples
    keep_mask = [pos_mask[i] or neg_keep_mask[i] for i in range(len(pos_mask))]
    keep_indices = [i for i in range(len(keep_mask)) if keep_mask[i]]

    subset_dataset = torch.utils.data.Subset(dataset, keep_indices)
    return subset_dataset


# # Main

# In[14]:


#def main():

def main(args):

    if args["doc_stride"] >= args["max_seq_length"] - args["max_query_length"]:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    if (
        os.path.exists(args["output_dir"])
        and os.listdir(args["output_dir"])
        and args["do_train"]
        and not args["overwrite_output_dir"]
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args["output_dir"]
            )
        )

    # Setup distant debugging if needed
    if args["server_ip"] and args["server_port"]:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args["server_ip"], args["server_port"]), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args["local_rank"] == -1 or args["no_cuda"]:
        device = torch.device("cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")
        args["n_gpu"] = 0 if args["no_cuda"] else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args["local_rank"])
        device = torch.device("cuda", args["local_rank"])
        torch.distributed.init_process_group(backend="nccl")
        args["n_gpu"] = 1
    args["device"] = device

    # Setup logging
    '''logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args["local_rank"] in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args["local_rank"],
        device,
        args["n_gpu"],
        bool(args["local_rank"] != -1),
        args["fp16"],
    )'''
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(args["local_rank"]):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args["local_rank"] not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args["model_type"] = args["model_type"].lower()
    config = AutoConfig.from_pretrained(
        args["config_name"] if args["config_name"] else args["model_name_or_path"],
        cache_dir=args["cache_dir"] if args["cache_dir"] else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args["tokenizer_name"] if args["tokenizer_name"] else args["model_name_or_path"],
        do_lower_case=args["do_lower_case"],
        cache_dir=args["cache_dir"] if args["cache_dir"] else None,
        use_fast=False,  # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        args["model_name_or_path"],
        from_tf=bool(".ckpt" in args["model_name_or_path"]),
        config=config,
        cache_dir=args["cache_dir"] if args["cache_dir"] else None,
    )

    if args["local_rank"] == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args["device"])

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args["fp16"]:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    if args["do_train"]:
        # NOTE: balances dataset in load_and_cache_examples
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args["do_train"] and (args["local_rank"] == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", args["output_dir"])
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args["output_dir"])
        tokenizer.save_pretrained(args["output_dir"])

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args["output_dir"], "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelForQuestionAnswering.from_pretrained(args["output_dir"])  # , force_download=True)
        #print("LOADED MODEL")

        # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
        # So we use use_fast=False here for now until Fast-tokenizer-compatible-examples are out
        tokenizer = AutoTokenizer.from_pretrained(args["output_dir"], do_lower_case=args["do_lower_case"], use_fast=False)
        model.to(args["device"])

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args["do_eval"] and args["local_rank"] in [-1, 0]:
        if args["do_train"]:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [args["output_dir"]]
            if args["eval_all_checkpoints"]:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(args["output_dir"] + "/**/" + WEIGHTS_NAME, recursive=True))
                )

        else:
            logger.info("Loading checkpoint %s for evaluation", args["model_name_or_path"])
            checkpoints = [args["model_name_or_path"]]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)  # , force_download=True)
            model.to(args["device"])

            # Evaluate
            result = evaluate(args, model, tokenizer, prefix=global_step)

            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

    return results


# #TRAIN the MODEL

# In[15]:


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args["local_rank"] in [-1, 0]:
        tb_writer = SummaryWriter()

    args["train_batch_size"] = args["per_gpu_train_batch_size"] * max(1, args["n_gpu"])
    if args["keep_frac"] < 1:
        train_dataset = get_random_subset(train_dataset, keep_frac=args["keep_frac"])

    train_sampler = RandomSampler(train_dataset) if args["local_rank"] == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args["train_batch_size"])

    if args["max_steps"] > 0:
        t_total = args["max_steps"]
        args["num_train_epochs"] = args["max_steps"] // (len(train_dataloader) // args["gradient_accumulation_steps"]) + 1
    else:
        t_total = len(train_dataloader) // args["gradient_accumulation_steps"] * args["num_train_epochs"]

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args["weight_decay"],
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args["learning_rate"], eps=args["adam_epsilon"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args["warmup_steps"], num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args["model_name_or_path"], "optimizer.pt")) and os.path.isfile(
        os.path.join(args["model_name_or_path"], "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args["model_name_or_path"], "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args["model_name_or_path"], "scheduler.pt")))

    if args["fp16"]:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args["fp16_opt_level"])

    # multi-gpu training (should be after apex fp16 initialization)
    if args["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args["local_rank"] != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args["local_rank"]], output_device=args["local_rank"], find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args["num_train_epochs"])
    logger.info("  Instantaneous batch size per GPU = %d", args["per_gpu_train_batch_size"])
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args["train_batch_size"]
        * args["gradient_accumulation_steps"]
        * (torch.distributed.get_world_size() if args["local_rank"] != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args["gradient_accumulation_steps"])
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args["model_name_or_path"]):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args["model_name_or_path"].split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args["gradient_accumulation_steps"])
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args["gradient_accumulation_steps"])

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args["num_train_epochs"]), desc="Epoch", disable=args["local_rank"] not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)


    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args["local_rank"] not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args["device"]) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            if args["model_type"] in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
                del inputs["token_type_ids"]

            if args["model_type"] in ["xlnet", "xlm"]:
                raise NotImplementedError

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args["n_gpu"] > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args["gradient_accumulation_steps"] > 1:
                loss = loss / args["gradient_accumulation_steps"]

            if args["fp16"]:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args["gradient_accumulation_steps"] == 0:
                if args["fp16"]:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args["max_grad_norm"])
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args["max_grad_norm"])

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args["local_rank"] in [-1, 0] and args["logging_steps"] > 0 and global_step % args["logging_steps"] == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args["local_rank"] == -1 and args["evaluate_during_training"]:
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args["logging_steps"], global_step)
                    logging_loss = tr_loss

                # Save model checkpoint
                if args["local_rank"] in [-1, 0] and args["save_steps"] > 0 and global_step % args["save_steps"] == 0:
                    output_dir = os.path.join(args["output_dir"], "checkpoint-{}".format(global_step))
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args["max_steps"] > 0 and global_step > args["max_steps"]:
                epoch_iterator.close()
                break
        if args["max_steps"] > 0 and global_step > args["max_steps"]:
            train_iterator.close()
            break

    if args["local_rank"] in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


# # Evaluate the Model

# In[16]:


def evaluate(args, model, tokenizer, prefix=""):

    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    with open(args["predict_file"], "r") as f:
        json_test_dict = json.load(f)

    if not os.path.exists(args["output_dir"]) and args["local_rank"] in [-1, 0]:
        os.makedirs(args["output_dir"])

    args["eval_batch_size"] = args["per_gpu_eval_batch_size"] * max(1, args["n_gpu"])

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args["eval_batch_size"])

    # multi-gpu evaluate
    if args["n_gpu"] > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args["eval_batch_size"])

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args["device"]) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if args["model_type"] in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
                del inputs["token_type_ids"]

            feature_indices = batch[3]

            # XLNet and XLM use more arguments for their predictions
            if args["model_type"] in ["xlnet", "xlm"]:
                raise NotImplementedError
            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs.to_tuple()]

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(args["output_dir"], "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args["output_dir"], "nbest_predictions_{}.json".format(prefix))

    if args["version_2_with_negative"]:
        output_null_log_odds_file = os.path.join(args["output_dir"], "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    # XLNet and XLM use a more complex post-processing procedure
    if args["model_type"] in ["xlnet", "xlm"]:
        raise NotImplementedError
    else:
        predictions = compute_predictions_logits(
            json_test_dict,
            examples,
            features,
            all_results,
            args["n_best_size"],
            args["max_answer_length"],
            args["do_lower_case"],
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args["verbose_logging"],
            args["version_2_with_negative"],
            args["null_score_diff_threshold"],
            tokenizer,
        )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    print(results)
    return results


# # RUN

# In[ ]:


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    
    if args["local_rank"] not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args["data_dir"] if args["data_dir"] else "."
    cached_features_file = os.path.join(
        args["cache_dir"],
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args["model_name_or_path"].split("/"))).pop(),
            str(args["max_seq_length"]),
        ),
    )
    subset_cached_features_file = os.path.join(
        args["cache_dir"],
        "balanced_subset_cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args["model_name_or_path"].split("/"))).pop(),
            str(args["max_seq_length"]),
        ),
    )
    print(subset_cached_features_file)

    # Init features and dataset from cache if it exists
    if os.path.exists(subset_cached_features_file) and not args["overwrite_cache"]:
        logger.info("Loading features from balanced cached file %s", subset_cached_features_file)
        dataset = torch.load(subset_cached_features_file)["dataset"]
        features, examples = None, None
    elif os.path.exists(cached_features_file) and not args["overwrite_cache"]:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        if evaluate:
            features, dataset, examples = (
                features_and_dataset["features"],
                features_and_dataset["dataset"],
                features_and_dataset["examples"],
            )
        else:
            dataset = features_and_dataset["dataset"]
            dataset = get_balanced_dataset(dataset)
            if args["local_rank"] in [-1, 0]:
                logger.info("Saving balanced dataset into cached file %s", subset_cached_features_file)
                torch.save({"dataset": dataset}, subset_cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if not args["data_dir"] and ((evaluate and not args["predict_file"]) or (not evaluate and not args["train_file"])):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if args["version_2_with_negative"]:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        else:
            processor = SquadV2Processor() if args["version_2_with_negative"] else SquadV1Processor()
            if evaluate:
                examples = processor.get_dev_examples(args["data_dir"], filename=args["predict_file"])
            else:
                examples = processor.get_train_examples(args["data_dir"], filename=args["train_file"])

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args["max_seq_length"],
            doc_stride=args["doc_stride"],
            max_query_length=args["max_query_length"],
            is_training=not evaluate,
            return_dataset="pt",
            threads=args["threads"],
        )

        if evaluate:
            if args["local_rank"] in [-1, 0]:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)
        else:
            dataset = get_balanced_dataset(dataset)
            if args["local_rank"] in [-1, 0]:
                logger.info("Saving balanced dataset into cached file %s", subset_cached_features_file)
                torch.save({"dataset": dataset}, subset_cached_features_file)

    if args["local_rank"] == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset


# In[ ]:


main(args)

