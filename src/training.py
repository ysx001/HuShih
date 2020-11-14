# %%
import nlp
import imp
print("nlp module", imp.find_module("nlp"))
import os
import logging
import argparse
from collections import defaultdict
from multiprocessing import Process, Array
from typing import Dict, Union, Any
import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from data_utils.lcsts import LCSTS
from transformers import (BertTokenizer, EncoderDecoderModel, Trainer,
                          TrainingArguments)
from lm_score.bert_lm import get_sentences_scores

# Get the root level dir
root = os.path.dirname(os.getcwd())
# logging settings
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
# handler = logging.StreamHandler()
# handler.setLevel(logging.INFO)
# LOG.addHandler(handler)
logging.basicConfig(level=logging.INFO)
# constants
DEFAULT_MODEL_NAME = 'hfl/chinese-roberta-wwm-ext'
DEFAULT_TRAINING_PATH = os.path.join(root, 'data/LCSTS2.0/DATA/PART_I.txt')
DEFAULT_VAL_PATH = os.path.join(root, 'data/LCSTS2.0/DATA/PART_II.txt')
DEFAULT_TEST_PATH = os.path.join(root, 'data/LCSTS2.0/DATA/PART_III.txt')
DEFAULT_OUTPUT_PATH = os.path.join(root, 'data')
REWARD_MAP = defaultdict(float)


# Freeze embedding layers, and first N layers of decoder
def freeze_decoder_weight(model, num_layers):
    for param in model.decoder.base_model.embeddings.parameters():
        param.requires_grad = False
    for i in range(num_layers):
        for param in model.decoder.base_model.encoder.layer[i].parameters():
            param.requires_grad = False


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str,
                                 rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


def compute_hybrid_rewards(inputs_labels, decode_ids):
    """TODO: input real sentence here.

    Args:
        labels ([type]): [description]
        outputs ([type]): [description]

    Returns:
        [type]: [description]
    """
    # reward = 0
    # for i in range(len(labels)):
    #     rouge_output = rouge.compute(predictions=outputs[i], references=labels[i],
    #                                 rouge_types=["rouge2"])["rouge2"].mid
    #     reward += round(rouge_output.fmeasure, 4) * get_sentence_score(outputs)
    # label_decoded = tokenizer.batch_decode(inputs['labels'],
    #                                        skip_special_tokens=True)
    curr_iter_decoded = tokenizer.batch_decode(decode_ids,
                                               skip_special_tokens=True)
    pred = [" ".join(map(str, decode_id)) for decode_id in decode_ids.tolist()]
    ref = [" ".join(map(str, input_label)) for input_label in inputs_labels.tolist()]
    rouge_outputs = rouge.compute(predictions=pred, references=ref,
                                  rouge_types=["rouge2", "rouge1", "rougeL"],
                                  use_agregator=False)
    rouge_scores = [0.0] * len(pred)
    for rouge_value in rouge_outputs.values():
        c = 0
        for score in rouge_value:
            rouge_scores[c] += score.fmeasure
            c += 1
    rouge_scores = np.asarray(rouge_scores) / 3
    print("rouge score", rouge_scores)
    ppl_values = Array('d', [0.0] * len(rouge_scores))
    p = Process(target=get_sentences_scores, args=(curr_iter_decoded, ppl_values))
    p.start()
    p.join()
    print("ppl before normalize", ppl_values[:])
    ppl = np.asarray(ppl_values[:])
    # normalize ppl score
    ppl = 2 * np.tanh(-ppl)
    print("ppl after normalize", ppl)
    return rouge_scores + ppl


class CustomizeEncoderDecoder(EncoderDecoderModel):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,  # TODO: (PVP) implement :obj:`use_cache`
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,  # TODO: (PVP) implement :obj:`use_cache`
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        id=None,
        **kwargs,
    ):
        return super().forward(input_ids=input_ids,
                               attention_mask=attention_mask,
                               decoder_input_ids=decoder_input_ids,
                               decoder_attention_mask=decoder_attention_mask,
                               encoder_outputs=encoder_outputs,
                               past_key_values=past_key_values,  # TODO: (PVP) implement :obj:`use_cache`
                               inputs_embeds=inputs_embeds,
                               decoder_inputs_embeds=decoder_inputs_embeds,
                               labels=labels,
                               use_cache=use_cache,  # TODO: (PVP) implement :obj:`use_cache`
                               output_attentions=output_attentions,
                               output_hidden_states=output_hidden_states,
                               return_dict=return_dict,
                               **kwargs,)

class CustomizeTrainer(Trainer):
    def compute_loss(self, model, inputs):
        """
        How the loss is computed by Trainer.
        By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        outputs = model(**inputs)
        print("****outputs: loss {}, shape {}".format(outputs[0], outputs[1].shape))
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        # Also return the max vocab index on (batch_size, num_tokens). This is
        # For use of decoding to Chinese words for this training step.
        return outputs[0], outputs[1].argmax(2)

    # Override training step
    def training_step(self, model: nn.Module,
                      inputs: Dict[str,
                                   Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model.
                Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for
                all accepted arguments.
        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        if hasattr(self, "_training_step"):
            warnings.warn(
                "The `_training_step` method is deprecated \
                and won't be called in a future version, \
                define `training_step` in your subclass.",
                FutureWarning,
            )
            return self._training_step(model, inputs, self.optimizer)

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.args.fp16 and _use_native_amp:
            with autocast():
                loss, decode_ids = self.compute_loss(model, inputs)
        else:
            loss, decode_ids = self.compute_loss(model, inputs)

        # mean() to average on multi-gpu parallel training
        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        # print("*****inputs: ", inputs)
        print("***input ids", inputs["id"])
        print("decode decoder input ids: ", tokenizer.batch_decode(inputs['decoder_input_ids'], skip_special_tokens=True))
        print("decode input ids: ", tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True))
        print("decode labels: ", tokenizer.batch_decode(inputs['labels'], skip_special_tokens=True))
        print("decode current iteration softmax: ", tokenizer.batch_decode(decode_ids, skip_special_tokens=True))
        # raise Exception("stop here")

        rewards = compute_hybrid_rewards(inputs['labels'], decode_ids)
        LOG.info("got reward %s", rewards)
        # compute current aggregated reward
        current_ids = inputs['id'].tolist()
        prev_reward = sum([REWARD_MAP[idx] for idx in current_ids])
        curr_reward = sum(rewards)
        loss *= (curr_reward - prev_reward)
        # store current iterations's reward in to cache
        for i in range(len(rewards)):
            REWARD_MAP[current_ids[i]] = rewards[i]

        if self.args.fp16 and _use_native_amp:
            self.scaler.scale(loss).backward()
        elif self.args.fp16 and _use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.detach()


def setup_dataset(train_data_files, val_data_files, tokenizer):
    # load train and validation data
    # TODO: using test data to see stuffs working first
    train_dataset = load_dataset('csv', data_files=[train_data_files])['train']
    val_dataset = load_dataset('csv', data_files=[val_data_files])['train']

    # map data correctly
    def map_to_encoder_decoder_inputs(batch):
        # Tokenizer will automatically set [BOS] <text> [EOS]
        # cut off at BERT max length 512
        inputs = tokenizer(batch["short_text"], padding="max_length",
                           truncation=True, max_length=128)
        # force summarization <= 128
        outputs = tokenizer(batch["summary"], padding="max_length",
                            truncation=True, max_length=32)
        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["id"] = batch["id"]

        batch["decoder_input_ids"] = outputs.input_ids
        batch["labels"] = outputs.input_ids.copy()
        # mask loss for padding
        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
        ]
        batch["decoder_attention_mask"] = outputs.attention_mask

        assert all([len(x) == 128 for x in inputs.input_ids])
        assert all([len(x) == 32 for x in outputs.input_ids])

        return batch

    # make train dataset ready
    train_dataset = train_dataset.map(
        map_to_encoder_decoder_inputs,
        batched=True,
        batch_size=args.batch_size,
        remove_columns=["short_text", "summary"],
    )
    train_dataset.set_format(
        type="torch", columns=["input_ids",
                               "attention_mask",
                               "decoder_input_ids",
                               "decoder_attention_mask",
                               "labels",
                               "id"],
    )
    # same for validation dataset
    val_dataset = val_dataset.map(
        map_to_encoder_decoder_inputs,
        batched=True,
        batch_size=args.batch_size,
        remove_columns=["short_text", "summary"],
    )
    val_dataset.set_format(
        type="torch", columns=["input_ids",
                               "attention_mask",
                               "decoder_input_ids",
                               "decoder_attention_mask",
                               "labels",
                               "id"],
    )
    return train_dataset, val_dataset


def load_tokenizer(model_name):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # CLS token will work as BOS token
    tokenizer.bos_token = tokenizer.cls_token
    # SEP token will work as EOS token
    tokenizer.eos_token = tokenizer.sep_token
    return tokenizer


def setup_model(model_name, num_freeze_decoder_layers, tokenizer):
    model = CustomizeEncoderDecoder.from_encoder_decoder_pretrained(model_name,
                                                                    model_name)

    # Freeze all layers in encoder
    for param in model.encoder.base_model.parameters():
        param.requires_grad = False

    # Try freeze first 8 layers in decoder first
    freeze_decoder_weight(model, num_freeze_decoder_layers)
    # set decoding params
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.max_length = 142
    model.config.min_length = 56
    model.config.no_repeat_ngram_size = 3
    model.early_stopping = True
    model.length_penalty = 2.0
    model.num_beams = 4
    return model

def run(args, lcsts):
    # load train and validation data
    # TODO: using test data to see stuffs working first
    train_dataset, val_dataset = setup_dataset(train_data_files=lcsts.test_merged_csv,
                                               val_data_files=lcsts.test_merged_csv,
                                               tokenizer=tokenizer)
    # setup model
    model = setup_model(args.model_name, args.num_freeze_decoder_layers, tokenizer)

    # set training arguments - these params are not really tuned,
    # feel free to change
    training_args = TrainingArguments(
        output_dir="./ckpt/",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        # predict_from_generate=True,
        evaluate_during_training=True,
        do_train=True,
        do_eval=True,
        logging_steps=1000,
        save_steps=1000,
        eval_steps=1000,
        overwrite_output_dir=True,
        warmup_steps=2000,
        save_total_limit=10,
    )

    # instantiate trainer
    trainer = CustomizeTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    # start training
    trainer.train()
    # trainer.train("checkpoint-9500")

if __name__ == '__main__':
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
                        default=16)
    parser.add_argument('--num_freeze_decoder_layers',
                        help='the number of decoder layers to freeze',
                        type=int,
                        default=8)
    parser.add_argument('--model_name',
                        help='the batch size for training and validation',
                        type=str,
                        default=DEFAULT_MODEL_NAME)
    args = parser.parse_args()
    LOG.info("Parsed arguments %s", args)

    # Step 1: preprocess the dataset and load data
    lcsts = LCSTS(args.training_path, args.val_path, args.test_path,
                  output_path=args.preprocess_output_path)

    LOG.info("Test files saved to path {}".format(lcsts.test_merged_csv))
    tokenizer = load_tokenizer(args.model_name)
    # load rouge for validation
    rouge = nlp.load_metric("rouge")

    # Load tokenizer
    if torch.cuda.device_count() > 0:
        import sys
        print('__Python VERSION:', sys.version)
        print('__pyTorch VERSION:', torch.__version__)
        print('__CUDA VERSION')
        from subprocess import call
        # call(["nvcc", "--version"]) does not work
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__Devices')
        call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
        print('Active CUDA Device: GPU', torch.cuda.current_device())

        print('Available devices ', torch.cuda.device_count())
        print('Current cuda device ', torch.cuda.current_device())
        with torch.cuda.device(0):
            run(args, lcsts)
    else:
        run(args, lcsts)
