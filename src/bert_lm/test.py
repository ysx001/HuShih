#%%
import tokenization
import time
a = time.time()
vocab_file = "model/chinese_wwm_ext_L-12_H-768_A-12/vocab.txt"
tokenizer = tokenization.FullTokenizer(
  vocab_file=vocab_file, do_lower_case=True)

tokens = tokenizer.tokenize("全国低考卷答题模板")

print(tokens)
print(time.time() - a)

