'''
Wikipediaのデータから学習データを作成する。
一部処理はマルチプロセスで動作する。
'''
import numpy as np
import os, pickle
import train_data_gen_settings as G
from sentences_generator import Sentences
import multi_vocab_generator as V_gen
import save_prepared_data as Save

# パラメータの読み込み
total_proc = G.total_proc # 多重実行数
k = G.window_size # ウィンドウサイズ
context_size = 2*k

# 使用するファイルの設定
sentence_file = G.sentence_file
train_data = G.train_data
vocab_file = G.vocab_file
vocab_dict_file = G.vocab_dictionary_file
reverse_vocab_dict_file = G.reverse_vocab_dictionary_file
wakati_files_dir = G.wakati_files_dir

# 学習する単語を格納する
vocabulary = dict()

# 作成済みの単語辞書ファイル(Pickle)が存在する場合、それを使用する。
if os.path.exists(vocab_dict_file):
  print("Load vocabulary dictionary from pickle. {}".format(vocab_dict_file))
  G.vocab_count_per_sentence = dict() 
  with open(vocab_dict_file, 'rb') as f:
    vocabulary = pickle.load(f)

    G.vocab_size = len(vocabulary)
    G.train_words = 0
    for val in vocabulary.values():
      G.train_words += int(val)

  print("  vocab_size:{}".format(G.vocab_size))
  print("  train_words:{}".format(G.train_words))

# 作成済みの単語辞書ファイルが存在しない場合、新規に作成する。
else:  
  print("Generate vocabulary dictionary")
  vocabulary = V_gen.build_vocabulary_multi(sentence_file, G.total_proc)
  # 単語数をカウントする。
  G.vocab_size = len(vocabulary)
  G.train_words = 0
  for val in vocabulary.values():
    G.train_words += int(val)

# リバース単語辞書が存在する場合、それを使用する。
if os.path.exists(reverse_vocab_dict_file):
  print("Load reverse vocab dictionary from pickle. {}".format(reverse_vocab_dict_file))
  with open(reverse_vocab_dict_file,'rb') as f:
    reverse_vocabulary = pickle.load(f)

# リバース単語辞書が存在しない場合、新規に作成する。
else: 
  print("Generate reverse vocabulary dictionary")
  V_gen.filter_vocabulary_based_on(vocabulary, G.min_count)
  # ボキャブラリーとその出現度数のリストを作り、vocab_fileとして保存する。
  reverse_vocabulary = V_gen.generate_inverse_vocabulary_lookup(vocabulary, vocab_file)

# 訓練に使用するデータを作成し、ファイル保存する。
Save.save_data(wakati_files_dir, vocabulary, reverse_vocabulary, train_data, vocab_dict_file, reverse_vocab_dict_file)
