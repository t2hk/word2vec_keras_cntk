'''
学習に使用するデータを作成して保存する。
マルチプロセスで動作する。
'''
import os, pickle, linecache, re, sys, math, random, collections, math, re, glob
import numpy as np
import train_data_gen_settings as G
from sentences_generator import Sentences
from multiprocessing import Process

# 同時実行するプロセス数
total_proc = G.total_proc

# 全体的に出現頻度が高い単語を多く含む文章は学習対象外にするための仕組み。
def subsample_sentence(sentence, vocabulary):
   subsampled_sentence = list()
   # 単語辞書に存在しない単語をUnknownに変換して文章を組み替える。
   sentence = [word if word in vocabulary else G.UNKNOWN_WORD for word in sentence]

   # サンプリングレートが0以下の場合はそのまま文章を返す。
   if G.sample <= 0:
      return sentence
   # 文章中の単語の出現頻度を計算し、頻度が高い単語をリスト化する。
   for word in sentence:
      prob = (math.sqrt(vocabulary[word] / (G.sample * G.train_words)) + 1) * (G.sample * G.train_words) / vocabulary[word]
      rand = random.random()
      if prob < rand:
         continue
      else:
         subsampled_sentence.append(word)
   return subsampled_sentence

# cbowの学習時に使用するネガティブサンプルを組み立てる。
def get_negative_samples(current_word_index, vocab_size):
  negative_samples = random.sample(range(G.vocab_size), G.negative)
  while current_word_index in negative_samples:
    negative_samples = random.sample(range(G.vocab_size), G.negative)
  return np.array([negative_samples])

# 学習データを作成して保存する。1プロセス分の処理である。
def save_prepared_data(wakati_files, vocabulary, reverse_vocabulary, data_save_file, vocab_size, total_proc, proc_no):
  tmp_wakati_files = [] 
  processed_files = 0

  # 分かち書きされたファイルをプロセス数分取得し、自分のプロセスに対応するファイルを処理する。
  for tmp_wakati_file in wakati_files:
    tmp_wakati_files.append(tmp_wakati_file)

    if len(tmp_wakati_files) == total_proc or (len(wakati_files) - processed_files) < total_proc:
      processed_files += total_proc

      if len(tmp_wakati_files) < (proc_no + 1):
         break
      wakati_file = tmp_wakati_files[proc_no]
      tmp_wakati_files = []

      # 本プロセスで作成した学習データの保存ファイル
      data_proc_save_file = data_save_file + '_' + os.path.basename(wakati_file) + '_' + str(proc_no)

      print('proc:{} wakati file:{} prepare file:{}'.format(proc_no, wakati_file, data_proc_save_file))

      # 当該ファイルを行単位で読み込む
      sentences = Sentences(wakati_file)

      with open(data_proc_save_file, mode='w') as f:
        print("data_proc_save_file:{}".format(data_proc_save_file))

        counter = 1
        current_sentences = []

        # 行単位で処理する。
        for sentence in sentences:
          # 文章を空白で単語毎に分割する。
          sentence = sentence.split()

          # 全体的に出現頻度が高い単語が多く含まれる文章は学習対象外とする。
          sentence = subsample_sentence(sentence, vocabulary)
          if len(sentence) < G.MIN_SENTENCE_LENGTH:
            continue

          # 単語をインデックス化する。
          sent_seq = [reverse_vocabulary[word] for word in sentence]
          
          # 各単語について、現在の単語とその前後ウィンドウサイズ文の単語を取得する。
          sentence_length = len(sent_seq)
          for i in range(sentence_length):
            current_word_index = None
            context_word_indexes = list()

            # jが0の場合は中央の単語、その前後のウィンドウサイズ分だけループする。
            for j in range(-G.window_size, G.window_size + 1):
              if j == 0:
                current_word_index = np.array([sent_seq[i]])
              else:
                  if (i+j) < 0 or (i+j) >= sentence_length:
                    context_word_indexes.append(0)
                  else:
                    context_word_indexes.append(sent_seq[(i+j)])
      
            # ネガティブサンプルの単語に関する処理
            negative_samples = get_negative_samples(current_word_index, vocab_size)
            context_word_indexes = np.array([context_word_indexes])

            # 学習用のデータを組み立てる。
            # 1件の学習データは以下の構成となっている。
            # [行番号],[現在の単語インデックス],[ウィンドウサイズの単語（スペース区切り）],[ネガティブサンプルの単語(スペース区切り)]
            val =  '[' + str(counter) + '],' + str(current_word_index) + ',['
            for v in context_word_indexes[0]:
              val += str(v) + ' '
            val = val.strip()
            val += '],['

            for v in negative_samples[0]:
              val += str(v) + ' '
            val = val.strip()
            val += ']'

            counter += 1
            f.write(val + '\n')

# 学習用データを作成し、各種ファイルを保存する。
def save_data(wakati_files_dir, vocabulary, reverse_vocabulary, data_save_file, vocab_dict_file, reverse_vocab_dict_file):
  # 単語辞書と単語インデックス化辞書をPickle形式で保存する。
  with open(vocab_dict_file, 'wb') as f:
    pickle.dump(vocabulary, f)
  with open(reverse_vocab_dict_file, 'wb') as f:
    pickle.dump(reverse_vocabulary, f)

  procs = []
  # 分かち書きファイルを格納しているディレクトリ
  wakati_files = glob.glob(wakati_files_dir + '*')

  # 学習データの組み立てをマルチプロセスで実行する。
  for i in range(total_proc):
    procs.append(
      Process(target=save_prepared_data, args=(wakati_files, vocabulary, reverse_vocabulary, data_save_file, G.vocab_size, total_proc, i)))

  for proc in procs:
    proc.start()

  for proc in procs:
    proc.join()

