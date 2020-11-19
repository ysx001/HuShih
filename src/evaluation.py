#!/usr/bin/env python3
#%%
import nlp
import os
import argparse
import torch
from datasets import load_dataset
from transformers import BertTokenizer, EncoderDecoderModel
from data_utils.lcsts import LCSTS

root = os.path.dirname(os.getcwd())

DEFAULT_MODEL_NAME = 'hfl/chinese-roberta-wwm-ext'
DEFAULT_TRAINING_PATH = os.path.join(root, 'data/LCSTS2.0/DATA/PART_I.txt')
DEFAULT_VAL_PATH = os.path.join(root, 'data/LCSTS2.0/DATA/PART_II.txt')
DEFAULT_TEST_PATH = os.path.join(root, 'data/LCSTS2.0/DATA/PART_III.txt')
DEFAULT_OUTPUT_PATH = os.path.join(root, 'data')


parser = argparse.ArgumentParser()
parser.add_argument('--training_path',
                    help='Where the training data (PART_I.txt) located',
                    type=str,
                    default=DEFAULT_TRAINING_PATH)
parser.add_argument('--val_path',
                    help='Where the validation data (PART_II.txt) located',
                    type=str,
                    default=DEFAULT_VAL_PATH)
parser.add_argument('--test_path',
                    help='Where the test data (PART_III.txt) located',
                    type=str,
                    default=DEFAULT_TEST_PATH)
parser.add_argument('--preprocess_output_path',
                    help='where to output the processed data',
                    type=str,
                    default=DEFAULT_OUTPUT_PATH)
parser.add_argument('--batch_size',
                    help='the batch size for training and validation',
                    type=int,
                    default=64)
parser.add_argument('--model_name',
                    help='the batch size for training and validation',
                    type=str,
                    default=DEFAULT_MODEL_NAME)
args = parser.parse_args()


tokenizer = BertTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
model = EncoderDecoderModel.from_pretrained("ckpt/checkpoint-2800")
# model.to("cuda")

lcsts = LCSTS(args.training_path, args.val_path, args.test_path,
                output_path=args.preprocess_output_path)
test_dataset = load_dataset('csv', data_files=[lcsts.test_merged_csv])['train']

pred_str_keys = ["greedy_pred_str", "beam_output_pred_str",
                 "beam_output_ngram_pred_str",
                 "top_k_only_ngram_pred_str",
                 "top_p_only_ngram_pred_str",
                 "top_k_top_p_ngram_pred_str"]

def generate_summary(batch):
    max_summary_length = 32
    # tokenize the short_text and the ground truth summary
    inputs = tokenizer(batch["short_text"], padding="max_length",
                       truncation=True, max_length=128)
    labels = tokenizer(batch["summary"], padding="max_length",
                       truncation=True, max_length=max_summary_length)
    # run evaluation
    # input_ids = inputs.input_ids.to("cuda")
    # attention_mask = inputs.attention_mask.to("cuda")
    input_ids = torch.tensor(inputs.input_ids)
    attention_mask = torch.tensor(inputs.attention_mask)
    # documentation
    # https://huggingface.co/blog/how-to-generate
    # greedy serach
    greedy = model.generate(input_ids,
                            attention_mask=attention_mask,
                            max_length=max_summary_length)
    # concate ids to strings
    # this is because we are calculation rouge score for chinese text
    # we needed them in enumerical ids to correctly utilized the english-based
    # rouge scores
    # number string for rouge
    greedy_pred_str = [" ".join(map(str, pred_id)) for pred_id in greedy.tolist()]
    batch["greedy_pred_str"] = greedy_pred_str
    # decode the output
    greedy_pred_decode_str = tokenizer.batch_decode(greedy, skip_special_tokens=True)
    print("greedy_pred_decode_str: ", greedy_pred_decode_str)
    # beam search + no 2-gram appears twice
    beam = model.generate(input_ids,
                          max_length=max_summary_length,
                          num_beams=3,
                          early_stopping=True)
    # number string for rouge
    beam_output_pred_str = [" ".join(map(str, pred_id)) for pred_id in beam.tolist()]
    batch["beam_output_pred_str"] = beam_output_pred_str
    # decode the output
    beam_pred_decode_str = tokenizer.batch_decode(beam, skip_special_tokens=True)
    print("beam_pred_decode_str: ", beam_pred_decode_str)
    # beam search + no 2-gram appears twice
    beam_ngram = model.generate(input_ids,
                                max_length=max_summary_length,
                                num_beams=3,
                                no_repeat_ngram_size=2,
                                early_stopping=True)
    # number string for rouge
    beam_output_ngram_pred_str = [" ".join(map(str, pred_id)) for pred_id in beam_ngram.tolist()]
    batch["beam_output_ngram_pred_str"] = beam_output_ngram_pred_str
    # decode the output
    beam_ngram_pred_decode_str = tokenizer.batch_decode(beam_ngram, skip_special_tokens=True)
    batch["beam_ngram_pred_decode_str"] = beam_ngram_pred_decode_str
    print("beam_ngram_pred_decode_str: ", beam_ngram_pred_decode_str)
    # using top-k sampling
    # top_k_only = model.generate(input_ids,
    #                             do_sample=True,
    #                             max_length=max_summary_length,
    #                             top_k=10)
    # # number string for rouge
    # top_k_only_ngram_pred_str = [" ".join(map(str, pred_id)) for pred_id in top_k_only.tolist()]
    # batch["top_k_only_ngram_pred_str"] = top_k_only_ngram_pred_str
    # # decode the output
    # top_k_only_pred_decode_str = tokenizer.batch_decode(top_k_only, skip_special_tokens=True)
    # batch["top_k_only_pred_decode_str"] = top_k_only_pred_decode_str
    # print("top_k_only_pred_decode_str: ", top_k_only_pred_decode_str)
    # # using top-p sampling
    # top_p_only = model.generate(input_ids,
    #                             do_sample=True,
    #                             max_length=max_summary_length,
    #                             top_k=0,
    #                             top_p=0.92)
    # # number string for rouge
    # top_p_only_ngram_pred_str = [" ".join(map(str, pred_id)) for pred_id in top_p_only.tolist()]
    # batch["top_p_only_ngram_pred_str"] = top_p_only_ngram_pred_str
    # # decode the output
    # top_p_only_pred_decode_str = tokenizer.batch_decode(top_p_only, skip_special_tokens=True)
    # batch["top_p_only_pred_decode_str"] = top_p_only_pred_decode_str
    # print("top_p_only_pred_decode_str: ", top_p_only_pred_decode_str)
    # # using top-p + top-k sampling
    # top_k_top_p = model.generate(input_ids,
    #                              do_sample=True,
    #                              max_length=max_summary_length,
    #                              top_k=10,
    #                              top_p=0.92)
    #                         #    num_return_sequences=3 we can sample more than 1 sentences
    # # number string for rouge
    # top_k_top_p_ngram_pred_str = [" ".join(map(str, pred_id)) for pred_id in top_k_top_p.tolist()]
    # batch["top_k_top_p_ngram_pred_str"] = top_k_top_p_ngram_pred_str
    # # decode the output
    # top_k_top_p_pred_decode_str = tokenizer.batch_decode(top_k_top_p, skip_special_tokens=True)
    # batch["top_k_top_p_pred_decode_str"] = top_k_top_p_pred_decode_str
    # print("top_k_top_p_pred_decode_str: ", top_k_top_p_pred_decode_str)
    # label str for rouge
    label_str = [" ".join(map(str, label_id)) for label_id in labels.input_ids]
    batch["label_id_str"] = label_str
    label_decode_str = tokenizer.batch_decode(labels.input_ids, skip_special_tokens=True)
    print("label_decode_str: ", label_decode_str)
    return batch

if torch.cuda.device_count() > 0:
    with torch.cuda.device(0):
        results = test_dataset.map(generate_summary, batched=True,
                                   batch_size=args.batch_size,
                                   remove_columns=["short_text"])
        # load rouge for validation
        rouge = nlp.load_metric("rouge")
        for result_key in pred_str_keys:
            pred_str = results[result_key]
            label_str = results["label_id_str"]
            rouge_output = rouge.compute(predictions=pred_str, references=label_str,
                                         rouge_types=["rouge2", "rouge1", "rougeL"])
            print("{}'s rouge score {}".format(result_key, rouge_output))
