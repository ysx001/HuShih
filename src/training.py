# %%
import nlp
import zmq
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os
import logging
import argparse
from multiprocessing import Process, Array
from typing import Dict, Union, Any
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.distributions import Categorical
from tqdm.auto import tqdm
import numpy as np
from datasets import load_dataset
from data_utils.lcsts import LCSTS
from transformers import (BertTokenizer, EncoderDecoderModel, Trainer,
                          TrainingArguments, EvalPrediction)
from transformers.trainer_utils import PredictionOutput
from lm_score.bert_lm import get_sentences_scores

USE_RL = True
HYBRID_REWARD = False
RL_WEIGHT = 0.5

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# torch.backends.cudnn.enabled = False
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

# Get the root level dir
root = os.path.dirname(os.getcwd())
# logging settings
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
# constants
MAX_PPL = 500
DEFAULT_MODEL_NAME = 'hfl/chinese-roberta-wwm-ext'
DEFAULT_TRAINING_PATH = os.path.join(root, 'data/LCSTS2.0/DATA/PART_I.txt')
DEFAULT_VAL_PATH = os.path.join(root, 'data/LCSTS2.0/DATA/PART_II.txt')
DEFAULT_TEST_PATH = os.path.join(root, 'data/LCSTS2.0/DATA/PART_III.txt')
DEFAULT_OUTPUT_PATH = os.path.join(root, 'data')

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect('tcp://13.68.135.179:5555')

# Freeze embedding layers, and first N layers of decoder
def freeze_decoder_weight(model, num_layers):
    for param in model.decoder.base_model.embeddings.parameters():
        param.requires_grad = False
    for i in range(num_layers):
        for param in model.decoder.base_model.encoder.layer[i].parameters():
            param.requires_grad = False


def compute_metrics(pred):
    label_ids = pred.label_ids
    # pred_ids = pred.predictions[0].argmax(2)
    pred_ids = pred.predictions

    # print("****labels_ids, pred_ids:", labels_ids, pred_ids)
    # print("**shape", len(pred_ids), pred_ids[0].shape, pred_ids[1].shape, pred_ids[2].shape)
    # all unnecessary tokens are removed
    #  pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    pred_str = [" ".join(map(str, pred_id)) for pred_id in pred_ids.tolist()]
    label_str = [" ".join(map(str, label_id)) for label_id in label_ids.tolist()]
    # label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    # print("*** strings:", pred_str, label_str)
    rouge_output = rouge.compute(predictions=pred_str, references=label_str,
                                 rouge_types=["rouge2"])["rouge2"].mid
    # print("*** rouge:", rouge_output)
    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }

def sample_decode_curr_iter(logits, num_batch, greedy=True):
    sum_log_probs = []
    if greedy:
        # greedy sampling
        curr_decode_ids = logits.argmax(2)
    else:
        curr_decode_ids = []
        for i in range(num_batch):
            softmax = nn.Softmax(dim=1)
            probs = softmax(logits[i, :, :])
            print("probs shape", probs.shape)
            # perform multinomial sampling
            multi_dist = Categorical(probs)
            sampled_decode_ids = multi_dist.sample()
            print("sampled_decode_ids", sampled_decode_ids)
            curr_decode_ids.append(sampled_decode_ids)
            sum_log_prob = torch.sum(multi_dist.log_prob(sampled_decode_ids))
            print("sum_log_prob", sum_log_prob)
            sum_log_probs.append(sum_log_prob)
    curr_iter_decoded = tokenizer.batch_decode(curr_decode_ids,
                                               skip_special_tokens=True)
    LOG.info("decode current iteration softmax: %s", curr_iter_decoded)
    LOG.info("sum_log_probs: %s", sum_log_probs)
    return curr_decode_ids, curr_iter_decoded, sum_log_probs


def compute_rouge_score(inputs_labels, curr_decode_ids):
    pred = [" ".join(map(str, d_id)) for d_id in curr_decode_ids]
    ref = [" ".join(map(str, i_label)) for i_label in inputs_labels.tolist()]
    rouge_outputs = rouge.compute(predictions=pred, references=ref,
                                  rouge_types=["rouge2", "rouge1", "rougeL"],
                                  use_agregator=False)
    # compute averaged score of rouge2, rouge1, and rougeL
    rouge_scores = [0.0] * len(pred)
    for rouge_value in rouge_outputs.values():
        c = 0
        for score in rouge_value:
            rouge_scores[c] += score.fmeasure
            c += 1
    rouge_scores = np.asarray(rouge_scores) / 3
    print("rouge score", rouge_scores)
    return rouge_scores


def compute_bert_ppl_score(inputs_labels, num_batch, curr_iter_decoded):
    label_decoded = tokenizer.batch_decode(inputs_labels,
                                           skip_special_tokens=True)
    ppl_values = [0.0] * num_batch
    
    to_be_sent_msg = {
        'ppl_values': ppl_values,
        'curr_iter_decoded': curr_iter_decoded
    }
    socket.send(pickle.dumps(to_be_sent_msg))
    print("sent serialized dict")
    serialized_ppl_values = socket.recv()
    ppl_values = pickle.loads(serialized_ppl_values)

    # print("ppl before normalize", ppl_values[:])
    ppl = np.asarray(ppl_values[:])
    # normalize ppl score
    ppl = np.clip(ppl, 0, MAX_PPL)
    ppl = 2 * sigmoid(-ppl)
    print("ppl after normalize", ppl)
    return ppl


def reward_function(inputs_labels, decode_ids, greedy=True, is_hybrid=False):
    """Compute the reward associated with the current iteration.


    Args:
        inputs_labels (tensor): the ground truth label ids for
                                the current iteration.
        decode_ids (tensor): the prediction matrix of shape
                             (num_batch, num_tokens, num_vocab)
        greedy (bool, optional): If we want to do random sampling
                                 or greedy argmax.
                                 Defaults to True.
        is_hybrid (bool, optional): Is the reward a hyrid of rouge
                                    and bert perplexity score.
                                    Defaults to False.

    Returns:
        np.array(num_batch), list(num_batch): the reward score
            and the sum of sampling log probilities
            (only available when greedy is False)
    """
    num_batch, num_tokens, num_vocab = decode_ids.shape
    # perform sampling and decode the sentences
    curr_decode_ids, curr_iter_decoded, sum_log_probs = \
        sample_decode_curr_iter(decode_ids, num_batch, greedy=greedy)
    # compute rouge score
    rouge_scores = compute_rouge_score(inputs_labels,
                                       curr_decode_ids)
    # compute bert language model based perplexity score
    if is_hybrid:
        ppl = compute_bert_ppl_score(inputs_labels,
                                     num_batch,
                                     curr_iter_decoded)
    else:
        ppl = np.zeros(num_batch)
    return rouge_scores + ppl, sum_log_probs

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
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
        print("****outputs: loss {}, shape 1 {} shape 2 {}, shape 3 {}".format(outputs[0], outputs[1].shape, outputs[2].shape, outputs[3].shape))
        print("outputs:", len(outputs))
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        # Also return the max vocab index on (batch_size, num_tokens). This is
        # For use of decoding to Chinese words for this training step.
        return outputs[0], outputs[1]

    def _prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.
        Works both with or without labels.
        """
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        batch_size = dataloader.batch_size
        LOG.info("***** Running %s *****", description)
        LOG.info("  Num examples = %d", self.num_examples(dataloader))
        LOG.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        model.eval()
        # if is_torch_tpu_available():
        #     dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)
        if self.args.past_index >= 0:
            past = None
        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.args.device)
            if self.args.past_index >= 0:
                inputs["mems"] = past

            with torch.no_grad():
                # if self.args.predict_from_generate:
                if True:
                    max_length = model.config.max_length
                    logits_out = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"])
                    # in case the batch is shorter then max length, the output should be padded
                    logits = model.config.eos_token_id * torch.ones(
                        (logits_out.shape[0], max_length), dtype=logits_out.dtype, device=logits_out.device
                    )
                    logits[:, : logits_out.shape[-1]] = logits_out

                    if has_labels:
                        outputs = model(**inputs)
                        step_eval_loss = outputs[0]
                        eval_losses += [step_eval_loss.mean().item()]
                else:
                    outputs = model(**inputs)

                    if has_labels:
                        step_eval_loss, logits = outputs[:2]
                        eval_losses += [step_eval_loss.mean().item()]
                    else:
                        logits = outputs[0]
                    if self.args.past_index >= 0:
                        past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]
            if not prediction_loss_only:
                if preds is None:
                    preds = logits.detach()
                else:
                    preds = torch.cat((preds, logits.detach()), dim=0)
                if inputs.get("labels") is not None:
                    if label_ids is None:
                        label_ids = inputs["labels"].detach()
                    else:
                        label_ids = torch.cat((label_ids, inputs["labels"].detach()), dim=0)
        if self.args.local_rank != -1:
            # In distributed mode, concatenate all results from all nodes:
            if preds is not None:
                preds = self.distributed_concat(preds, num_total_examples=self.num_examples(dataloader))
            if label_ids is not None:
                label_ids = self.distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))
       # elif is_torch_tpu_available():
       #     # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
       #     if preds is not None:
       #         preds = xm.mesh_reduce("eval_preds", preds, torch.cat)
       #     if label_ids is not None:
       #         label_ids = xm.mesh_reduce("eval_label_ids", label_ids, torch.cat)
        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = preds.cpu().numpy()
        if label_ids is not None:
            label_ids = label_ids.cpu().numpy()
        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics["eval_loss"] = np.mean(eval_losses)
        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)
        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

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
                mle_loss, decode_ids = self.compute_loss(model, inputs)
        else:
            mle_loss, decode_ids = self.compute_loss(model, inputs)
        # print("loss right after compute_loss:", loss)

        # mean() to average on multi-gpu parallel training
        if self.args.n_gpu > 1:
            mle_loss = mle_loss.mean()
        # print("loss right after compute_loss mean:", loss)

        if self.args.gradient_accumulation_steps > 1:
            mle_loss = mle_loss / self.args.gradient_accumulation_steps
        # print("loss right after compute_loss accumlate:", loss)
        # print("*****inputs: ", inputs)
        # print("***input ids", inputs["id"])
        LOG.info("decode decoder input ids: %s", tokenizer.batch_decode(inputs['decoder_input_ids'], skip_special_tokens=True))
        LOG.info("decode input ids: %s", tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True))
        LOG.info("decode labels: %s", tokenizer.batch_decode(inputs['labels'], skip_special_tokens=True))
        # compute rl loss
        sample_reward, sample_log_probs = reward_function(
                                                inputs['labels'],
                                                decode_ids,
                                                greedy=False,
                                                is_hybrid=HYBRID_REWARD)
        baseline_reward, _ = reward_function(
                                        inputs['labels'],
                                        decode_ids,
                                        greedy=True,
                                        is_hybrid=HYBRID_REWARD)
        mean_sample_reward = np.mean(sample_reward)
        mean_baseline_reward = np.mean(baseline_reward)
        mean_sample_log_probs = torch.mean(torch.stack(sample_log_probs))
        rl_loss = - (mean_sample_reward - mean_baseline_reward) * mean_sample_log_probs
        LOG.info("rl_loss: %s", rl_loss)
        
        
        # LOG.info("got reward %s", rewards)
        # compute current aggregated reward
        # current_ids = inputs['id'].tolist()
        # prev_reward = sum([REWARD_MAP[idx] for idx in current_ids])
        # curr_reward = sum(rewards)
        # print("prev_reward:", prev_reward, "curr_reward:", curr_reward)
        # loss *= (curr_reward - prev_reward)
        # store current iterations's reward in to cache
        # for i in range(len(rewards)):
        #     REWARD_MAP[current_ids[i]] = rewards[i]
        if USE_RL:
            loss = (1 - RL_WEIGHT) * mle_loss + RL_WEIGHT * rl_loss
        else:
            loss = mle_loss

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
        #num_train_epochs=10,
        evaluate_during_training=True,
        do_train=True,
        do_eval=True,
        logging_steps=20,
        save_steps=100,
        eval_steps=1000,
        overwrite_output_dir=True,
        warmup_steps=40,
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
                        default=4)
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

    LOG.info("Train files saved to path {}".format(lcsts.train_merged_csv))
    LOG.info("Validation files saved to path {}".format(lcsts.val_merged_csv))
    LOG.info("Test files saved to path {}".format(lcsts.test_merged_csv))
    tokenizer = load_tokenizer(args.model_name)
    # load rouge for validation
    rouge = nlp.load_metric("rouge")

    # Load tokenizer
    if torch.cuda.device_count() > 0:
        with torch.cuda.device(0):
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
            run(args, lcsts)
    else:
        run(args, lcsts)
