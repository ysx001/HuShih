# %%
import nlp
import os
import logging
from transformers import BertTokenizer, EncoderDecoderModel, Trainer, TrainingArguments
from datasets import load_dataset
from data_utils.lcsts import LCSTS

root = os.path.dirname(os.getcwd())  # Get the root level dir
#%%
training_path = os.path.join(root, 'data/LCSTS2.0/DATA/PART_I.txt')
val_path = os.path.join(root, 'data/LCSTS2.0/DATA/PART_II.txt')
test_path = os.path.join(root, 'data/LCSTS2.0/DATA/PART_III.txt')
output_path = os.path.join(root, 'data')

print(test_path)

# %%
lcsts = LCSTS(training_path, val_path, test_path, output_path=output_path)
_, _, merged_test_csv = lcsts.test_csv
print("Test files saved to path {}".format(merged_test_csv))

#%%
MODEL_NAME='hfl/chinese-roberta-wwm-ext'

logging.basicConfig(level=logging.INFO)

model = EncoderDecoderModel.from_encoder_decoder_pretrained(MODEL_NAME, MODEL_NAME)
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# # CLS token will work as BOS token
tokenizer.bos_token = tokenizer.cls_token
#
# # SEP token will work as EOS token
tokenizer.eos_token = tokenizer.sep_token

# load train and validation data
train_dataset = load_dataset('csv', data_files=['data/test_merged.csv'])['train']
val_dataset = load_dataset('csv', data_files=['data/test_merged.csv'])['train']  # placer_holder

# load rouge for validation
rouge = nlp.load_metric("rouge")

# set decoding params
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.max_length = 142
model.config.min_length = 56
model.config.no_repeat_ngram_size = 3
model.early_stopping = True
model.length_penalty = 2.0
model.num_beams = 4


# map data correctly
def map_to_encoder_decoder_inputs(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    inputs = tokenizer(batch["short_text"], padding="max_length", truncation=True, max_length=512)
    # force summarization <= 128
    outputs = tokenizer(batch["summary"], padding="max_length", truncation=True, max_length=128)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    batch["decoder_input_ids"] = outputs.input_ids
    batch["labels"] = outputs.input_ids.copy()
    # mask loss for padding
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
    ]
    batch["decoder_attention_mask"] = outputs.attention_mask

    assert all([len(x) == 512 for x in inputs.input_ids])
    assert all([len(x) == 128 for x in outputs.input_ids])

    return batch


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


# set batch size here
batch_size = 16

# make train dataset ready
train_dataset = train_dataset.map(
    map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["short_text", "summary"],
)
train_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

# same for validation dataset
val_dataset = val_dataset.map(
    map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["short_text", "summary"],
)
val_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

# set training arguments - these params are not really tuned, feel free to change
training_args = TrainingArguments(
    output_dir="./",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
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
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# start training
trainer.train()
