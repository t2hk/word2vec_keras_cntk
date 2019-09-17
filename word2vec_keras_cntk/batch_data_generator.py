"""
Kerasのfit_generatorで使用する、バッチ処理用データのジェネレータクラス
"""
import os
import sys
import math
import numpy as np
import linecache
import re
from keras.utils import Sequence

# 設定ファイル
import training_settings as G

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
