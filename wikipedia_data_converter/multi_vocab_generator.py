"""
word2vecのcbowで学習するためのデータを作成する。Wikipediaの文章中の単語が対象である。
negative samplingに対応。Multiprocessingで動作する。
学習する単語は、名詞（数と非自立を除く）、形容詞、自立動詞（設定で定義した文字数以上）のみとしている。
"""
import collections, math, os, random
import sys, codecs, re, pickle, glob, json
import numpy as np
import linecache
import MeCab
from multiprocessing import Process
from sentences_generator import Sentences

from keras.utils import Sequence

# 設定ファイルを読み込む
import train_data_gen_settings as G

# MeCabで分かち書きした結果ファイルを格納するディレクトリ
wakati_files_dir = G.wakati_files_dir

# 単語辞書を組み立てる処理。
def build_vocabulary_multi(sentence_file, total_proc):
  procs = []

  # 分割したWikipediaのファイル群
  sentence_files = glob.glob(G.sentence_files)

  # マルチプロセスで学習データを作成する。
  for i in range(total_proc):
     vocab_dict_file = G.vocab_dictionary_file + '_' + str(i)
     procs.append(
       Process(target=build_vocabulary_per_proc, args=(sentence_files, total_proc, vocab_dict_file, i))) 

  for proc in procs:
    proc.start()
  for proc in procs:
    proc.join()

  # 各プロセスの処理完了後、それぞれの処理結果を統合する。
  vocab_dict_merge = {} 
  vocab_count_per_sentence = {}
  G.total_words = 0
  G.total_vocab = 0
  vocab_dict_files = glob.glob(G.vocab_dictionary_file + '_*') # 各プロセスの処理結果ファイル

  for vdf in vocab_dict_files:
    print(vdf)

    with open(vdf, 'rb') as f:
      proc_vocab_dict = pickle.load(f)

      print("dict size:{}".format(len(proc_vocab_dict)))

      # 各プロセスが作成した単語情報を統合する。
      for k, v in proc_vocab_dict.items():
        if k in vocab_dict_merge:
           vocab_dict_merge[k] += v 
           G.total_words += v
        else:
           vocab_dict_merge[k] = v
           G.total_words += v
           G.total_vocab += 1

  # 作成した単語辞書をPickle保存する。
  print("merge dict type:{} size:{}".format(type(vocab_dict_merge), len(vocab_dict_merge))) 
  with open(G.vocab_dictionary_file, mode='wb') as fd:
    pickle.dump(vocab_dict_merge, fd  ) 

  return vocab_dict_merge 

# Wikipediaの文書ファイルを読み込み、各文章をMeCabで分かち書きし、単語を抽出する。
# マルチプロセスで動作するため、分割したWikipedia文書ファイルをプロセス毎に処理する。
def build_vocabulary_per_proc( sentence_files, total_proc, vocab_dict_file, proc_no):

    # NGワード（STOPワード）をファイルから読み込む。
    NG_WORD_LIST = [] 
    with open(G.NG_WORD_LIST, mode='rt') as f:
      NG_WORD_LIST = list(f)

    tmp_sentence_files = []
    dict_file_no = 0
    mecab = MeCab.Tagger('-d /usr/lib/mecab/dic/mecab-ipadic-neologd')
    mecab.parse('')

    # センテンスファイル数が総プロセス数より少ない場合、総プロセス数をセンテンスファイル数と同値とする。
    if len(sentence_files) < total_proc:
       total_proc = len(sentence_files)

    processed_files = 0

    # 文書ファイルのパスをプロセス数分だけ読み込み、自分のプロセス番号に該当するファイルを処理する。
    for tmp_sentence_file in sentence_files:
      tmp_sentence_files.append(tmp_sentence_file)
      # プロセス数分のファイルパスを読み込んだら、自分のプロセス番号のファイルを処理する。
      if len(tmp_sentence_files) == total_proc or (len(sentence_files) - processed_files) < total_proc:
        processed_files += total_proc 

        if len(tmp_sentence_files) < (proc_no)+1:
           break

        # 処理対象の文書ファイルパスを取得し、その分かち書きした結果を格納するファイルパスを作成する。
        sentence_file = tmp_sentence_files[proc_no]
        sentence_wakati_file = wakati_files_dir + os.path.basename(sentence_file) + '_wakati'

        tmp_sentence_files = []
        vocabulary = dict()

        print("proc_no:{} file:{}".format(proc_no, sentence_file)) 

        with open(sentence_wakati_file, mode='w') as swf:
          sentences = Sentences(sentence_file)

          print("Generating Vocabulary from the sentences")

          train_words = 0
          sentence_procs = []
          counter = 0
          for sentence in sentences:
             wakati_line = []
             counter += 1
             if counter % 100000 == 0:
               print("proc:{}  counter:{}".format(proc_no, counter))

             node = mecab.parseToNode(sentence)

             while node:
                word = node.surface
                wakati_line.append(word)

                # NGワードはスキップする。
                if word in NG_WORD_LIST:
                    node = node.next
                    continue

                # min_charで指定された文字数未満の場合スキップする。
                if len(word) < G.min_char:
                  node = node.next
                  continue

                # 一文字の数字・アルファベットはスキップする。
                if len(word) == 1 and re.match('[a-xA-Z0-9]', word):
                    node = node.next
                    continue

                pos1 = node.feature.split(',')[0]
                pos2 = node.feature.split(',')[1]

                #数以外の名詞、最小文字数を超える動詞形容詞は学習する。
                if (pos1 == '名詞' and pos2 != '数' and pos2 != '非自立') or \
                   (pos1 == '動詞' and pos2 == '自立' and len(word) > G.min_char) or \
                   (pos1 == '形容詞' and len(word) > G.min_char):
                      vocabulary.setdefault(word, 0)
                      vocabulary[word] += 1
                      train_words += 1

                node = node.next
             swf.write(' '.join(wakati_line) + '\n') 
  
          print("Vocabulary size = %d" % len(vocabulary))
          print("Total words to be trained = %d" % train_words)

          with open(vocab_dict_file + '_' + str(dict_file_no), 'wb') as f:
            pickle.dump(vocabulary, f)

          dict_file_no += 1

# 使用頻度の少ない単語を削除する処理
def filter_vocabulary_based_on(vocabulary, min_count):
        # 頻度の少ない単語を削除する。削除する単語はUNKNOWNとする。
        print( "Deleting the words which occur less than %d times" % min_count)
        delete_word_list = [word for word, count in vocabulary.items() if count < min_count]
        unk_count = 0
        for word in delete_word_list:
          unk_count += vocabulary.pop(word, 0)
        vocabulary[G.UNKNOWN_WORD] = unk_count
        G.vocab_size = len(vocabulary)
        print("Vocabulary size after filtering words = %d" % G.vocab_size)

# 単語とそのインデックス値を出現順に並び替える。
def generate_inverse_vocabulary_lookup(vocabulary, save_filepath):
        # 単語を出現順に並び替える
        sorted_words = reversed(sorted(vocabulary, key=lambda word: vocabulary[word]))
        
        reverse_vocabulary = dict()
        # UNKNOWN WORDのインデックスを1にする。
        reverse_vocabulary[G.UNKNOWN_WORD] = 1
        index = 2
        with codecs.open(save_filepath, "w", "utf-8") as wf:
             # 最初にUNKNOWN WORDを記録する。
             wf.write(G.UNKNOWN_WORD + "\t" + str(vocabulary[G.UNKNOWN_WORD]) + "\n")
             for word in sorted_words:
               if word == G.UNKNOWN_WORD:
                  continue
               reverse_vocabulary[word] = index
               wf.write(word + "\t" + str(vocabulary[word]) + "\n")
               index += 1
        return reverse_vocabulary

'''
学習時のバッチ処理用のデータを組み立てるためのクラス。
Kerasのfit_generatorで使用する。
'''
class data_generator(Sequence):
  def __init__(self, data_file, batch_size):
    self.batch_size = batch_size
    self.data_file = data_file
    self.data_length = len(open(self.data_file).readlines())
   
  def __getitem__(self, idx):
      # バッチサイズ分のデータの開始と終了位置を算出する。
      start_idx = idx * self.batch_size + 1
      end_idx = (idx + 1) * self.batch_size + 1

      if end_idx > (self.data_length  + 1):
        end_idx = self.data_length + 1

      # x1は現在の単語、x2はウィドウサイズ分の単語群、x3はネガティブサンプルの単語群
      # y1は正解を表す1、y2はネガティブサンプルで不正解を表す0
      x1 = []
      x2 = []
      x3 = []
      y1 = []
      y2 = []

      # バッチサイズ分のデータについて処理する。
      for i in range(start_idx, end_idx):
        # テストデータから処理対象の行を読み込み、単語単位に分割する。
        current_line = i
        target_line = linecache.getline(self.data_file, current_line)
        target_line = target_line.split(',')

        # 現在の単語を取得する。
        current_word_index = target_line[1].replace('[','').replace(']','').strip()
        current_word_index = int(current_word_index)

        # ウィンドウサイズ分の単語を取得する。
        context_word_indexes = target_line[2].replace('[','').replace(']','').replace('\n','').split(' ')
        context_word_indexes = np.array([int(s) for s in context_word_indexes])

        # ネガティブサンプル分の単語を取得する。
        negative_samples = target_line[3].replace('[','').replace(']','').replace('\n','').split(' ')
        negative_samples = np.array([int(s) for s in negative_samples])

        # 学習データと正解ラベルとしてリストに追加する。
        x1.append( current_word_index)
        x2.append( context_word_indexes) 
        x3.append( negative_samples)
        y1.append( np.array((1.0)))
        y2.append( np.zeros((G.negative,)) )

      return [np.array(x1), np.array(x2), np.array(x3)], [np.array(y1), np.array(y2)]

  def __len__(self):
     return math.ceil(self.data_length / self.batch_size)
