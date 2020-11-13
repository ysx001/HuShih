# %%
import nlp
import os
import logging
from transformers import BertTokenizer, BertForMaskedLM
import numpy as np
import torch
import torch.nn as nn
import math

MODEL_NAME='hfl/chinese-bert-wwm-ext'

#%%
def score_1(model, tokenizer, sentence):
  """
  https://stackoverflow.com/questions/63030692/how-do-i-use-bertformaskedlm-or-bertmodel-to-calculate-perplexity-of-a-sentence

  Args:
      model ([type]): [description]
      tokenizer ([type]): [description]
      sentence ([type]): [description]
  """
  tokenize_input = tokenizer.tokenize(sentence)
  tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
  sen_len = len(tokenize_input)
  sentence_loss = 0.

  for i, word in enumerate(tokenize_input):
    # add mask to i-th character of the sentence
    tokenize_input[i] = '[MASK]'
    mask_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    output = model(mask_input)
    prediction_scores = output[0]
    softmax = nn.Softmax(dim=0)
    ps = softmax(prediction_scores[0, i]).log()
    word_loss = ps[tensor_input[0, i]]
    sentence_loss += word_loss.item()
    tokenize_input[i] = word
    print("word {} loss {}".format(word, word_loss.item()))
  ppl = np.exp(-sentence_loss / sen_len)
  print("score 1, sentence_loss {}, perplexity {}".format(sentence_loss, ppl))
  return ppl

def score_2(model, tokenizer, sentence):
  """

  Args:
      model ([type]): [description]
      tokenizer ([type]): [description]
      sentence ([type]): [description]
  """
  labels = tokenizer(sentence, return_tensors="pt")["input_ids"]
  sen_len = len(sentence)
  # tokenize_input = tokenizer.tokenize(sentence)
  tokenize_input = list(sentence)
  sentence_loss = 0.
  
  for i, word in enumerate(tokenize_input):
    # add mask to i-th character of the sentence
    tokenize_input[i] = '[MASK]'
    mask_input = tokenizer("".join(tokenize_input), return_tensors="pt")
    output = model(**mask_input, labels=labels)
    word_loss = output[0].data
    sentence_loss += word_loss
    tokenize_input[i] = word
    print("word {} loss {}".format(word, word_loss))
  
  ppl = np.exp(-sentence_loss/sen_len)
  print("score 2, sentence_loss {}, perplexity {}".format(sentence_loss, ppl))
  return ppl

def score_3(model, tokenizer, sentence):
  tokenize_input = tokenizer.tokenize(sentence)
  input_ids = torch.tensor(tokenizer.encode(tokenize_input)).unsqueeze(0)
  with torch.no_grad():
      loss = model(input_ids, masked_lm_labels=input_ids)[0]
  ppl = math.exp(loss.item() / len(tokenize_input))
  print("score 3, sentence_loss {}, perplexity {}".format(loss.item(), ppl))
  return ppl

def score_4(model, tokenizer, sentence):
  """
  https://www.scribendi.ai/can-we-use-bert-as-a-language-model-to-assign-score-of-a-sentence/

  Args:
      model ([type]): [description]
      tokenizer ([type]): [description]
      sentence ([type]): [description]

  Returns:
      [type]: [description]
  """
  tokenize_input = tokenizer.tokenize(sentence)
  tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
  print(tensor_input)
  predictions = model(tensor_input)
  loss_fct = torch.nn.CrossEntropyLoss()
  loss = loss_fct(predictions, tensor_input.squeeze()).data 
  return math.exp(loss)

def read_examples(input_file):
  with open(input_file, "r", encoding="utf-8") as f:
    sentences = [x.strip('\n') for x in f.readlines()]
  return sentences

#%%
logging.basicConfig(level=logging.INFO)

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForMaskedLM.from_pretrained(MODEL_NAME)

#%%
sentences = read_examples("examples.txt")
for sentence in sentences:
  print(sentence)
  score_1(model, tokenizer, sentence)
  score_2(model, tokenizer, sentence)
  score_3(model, tokenizer, sentence)
  # score_4(model, tokenizer, sentence)

#%%
"""
可穿戴技术十大设计原则
word 可 loss -5.821658134460449
word 穿 loss -6.789828777313232
word 戴 loss -8.556593894958496
word 技 loss -2.818181037902832
word 术 loss -1.154729962348938
word 十 loss -3.8941009044647217
word 大 loss -1.6794673204421997
word 设 loss -0.3747537136077881
word 计 loss -0.043797917664051056
word 原 loss -1.5636417865753174
word 则 loss -3.6971070766448975
score 1, sentence_loss -36.39386052638292, perplexity 27.344974796862992
word 可 loss 2.3972091674804688
word 穿 loss 2.44500732421875
word 戴 loss 2.462885856628418
word 技 loss 2.428769588470459
word 术 loss 2.4409117698669434
word 十 loss 2.585999011993408
word 大 loss 2.484440326690674
word 设 loss 2.4143118858337402
word 计 loss 2.4497570991516113
word 原 loss 2.4893293380737305
word 则 loss 2.5610694885253906
score 2, sentence_loss 27.159690856933594, perplexity 0.08466417342424393
score 3, sentence_loss 2.5075254440307617, perplexity 1.2560311372567712
可穿戴技术十小设计原则
word 可 loss -5.812528133392334
word 穿 loss -7.187803745269775
word 戴 loss -8.317155838012695
word 技 loss -2.388496160507202
word 术 loss -0.9816279411315918
word 十 loss -11.150832176208496
word 小 loss -6.659573554992676
word 设 loss -0.8421952128410339
word 计 loss -0.081656813621521
word 原 loss -1.9658514261245728
word 则 loss -4.186827659606934
score 1, sentence_loss -49.57454866170883, perplexity 90.62926290900342
word 可 loss 2.5534846782684326
word 穿 loss 2.4879698753356934
word 戴 loss 2.5671751499176025
word 技 loss 2.4496703147888184
word 术 loss 2.471465587615967
word 十 loss 3.372948169708252
word 小 loss 3.304237127304077
word 设 loss 2.3967134952545166
word 计 loss 2.438890218734741
word 原 loss 2.551191568374634
word 则 loss 2.7092807292938232
score 2, sentence_loss 29.30302619934082, perplexity 0.06967514753341675
score 3, sentence_loss 2.7990007400512695, perplexity 1.289758011422196
[
  {
    "tokens": [
      {
        "token": "可",
        "prob": 0.9995731115341187
      },
      {
        "token": "穿",
        "prob": 0.999721348285675
      },
      {
        "token": "戴",
        "prob": 0.9997082948684692
      },
      {
        "token": "技",
        "prob": 0.9996678233146667
      },
      {
        "token": "术",
        "prob": 0.9998599886894226
      },
      {
        "token": "十",
        "prob": 0.115843765437603
      },
      {
        "token": "大",
        "prob": 0.9493612051010132
      },
      {
        "token": "设",
        "prob": 0.9991070628166199
      },
      {
        "token": "计",
        "prob": 0.9984894394874573
      },
      {
        "token": "原",
        "prob": 0.6245705485343933
      },
      {
        "token": "则",
        "prob": 0.41334837675094604
      }
    ],
    "ppl": 1.3828370231757596
  },
  {
    "tokens": [
      {
        "token": "可",
        "prob": 0.999401330947876
      },
      {
        "token": "穿",
        "prob": 0.997902512550354
      },
      {
        "token": "戴",
        "prob": 0.9989476799964905
      },
      {
        "token": "技",
        "prob": 0.9991917610168457
      },
      {
        "token": "术",
        "prob": 0.9993842840194702
      },
      {
        "token": "十",
        "prob": 1.747102760418784e-05
      },
      {
        "token": "小",
        "prob": 9.412464351044036e-06
      },
      {
        "token": "设",
        "prob": 0.9486653804779053
      },
      {
        "token": "计",
        "prob": 0.9938652515411377
      },
      {
        "token": "原",
        "prob": 0.5431732535362244
      },
      {
        "token": "则",
        "prob": 0.18311570584774017
      }
    ],
    "ppl": 9.618380697038356
  }
]
"""
