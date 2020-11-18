#!/usr/bin/env python3
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
                    default=128)
parser.add_argument('--model_name',
                    help='the batch size for training and validation',
                    type=str,
                    default=DEFAULT_MODEL_NAME)
args = parser.parse_args()
# LOG.info("Parsed arguments %s", args)


tokenizer = BertTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
model = EncoderDecoderModel.from_pretrained("ckpt/checkpoint-1400")
# model.to("cuda")

lcsts = LCSTS(args.training_path, args.val_path, args.test_path,
                output_path=args.preprocess_output_path)
test_dataset = load_dataset('csv', data_files=[lcsts.test_merged_csv])['train']


def generate_summary(batch):
    # tokenize the short_text and the ground truth summary
    inputs = tokenizer(batch["short_text"], padding="max_length",
                       truncation=True, max_length=128)
    labels = tokenizer(batch["summary"], padding="max_length",
                       truncation=True, max_length=32)
    # run evaluation
    # input_ids = inputs.input_ids.to("cuda")
    # attention_mask = inputs.attention_mask.to("cuda")
    input_ids = torch.tensor(inputs.input_ids)
    attention_mask = torch.tensor(inputs.attention_mask)
    outputs = model.generate(input_ids, attention_mask=attention_mask)
    # concate ids to strings
    # this is because we are calculation rouge score for chinese text
    # we needed them in enumerical ids to correctly utilized the english-based
    # rouge scores
    pred_str = [" ".join(map(str, pred_id)) for pred_id in outputs.tolist()]
    label_str = [" ".join(map(str, label_id)) for label_id in labels.input_ids]
    batch["pred_id_str"] = pred_str
    batch["label_id_str"] = label_str
    # decode for logging purpose
    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    batch["pred_str"] = output_str
    print("output_str: ", output_str)
    return batch


results = test_dataset.map(generate_summary, batched=True,
                           batch_size=args.batch_size,
                           remove_columns=["short_text"])
# load rouge for validation
rouge = nlp.load_metric("rouge")
pred_str = results["pred_id_str"]
label_str = results["label_id_str"]
rouge_output = rouge.compute(predictions=pred_str, references=label_str,
                             rouge_types=["rouge2"])["rouge2"].mid
print(rouge_output)
