'''
wordwvecの動作確認用のプログラム。
'''
from keras import backend as K
import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Lambda, Dense, merge, dot
from keras.layers.embeddings import Embedding
import training_settings as G
from sentences_generator import Sentences
import save_embeddings as S
import cntk as C
import util as U

# 設定ファイルの読み込み
k = G.window_size
context_size = 2*k
train_data = G.train_data
vocab_file = G.vocab_file
model_file = G.model_file
model_onnx_file = G.model_onnx_file

# モデルのロード
model = load_model(model_file)
print(model.summary())

# モデルから単語ベクトルを取得
shared_embedding_layer = model.layers[2]
embedding = shared_embedding_layer.get_weights()[0]

#### ボキャブラリのディクショナリ作成
i_to_v = dict()
v_to_i = dict()

vocab_size = 0
with open(vocab_file) as f:
   while True:
     line = f.readline()

     if not line:
       break

     vocab_size += 1
     vocab = line.split('\t')[0].strip()
     i_to_v[vocab_size] = vocab
     v_to_i[vocab] = vocab_size 

### コサイン類似度

word1 = "猫"
word2 = "ライオン"
word3 = "犬"

emb1 = embedding[v_to_i[word1]]
emb2 = embedding[v_to_i[word2]]
emb3 = embedding[v_to_i[word3]]

cos_sim_1_2 = U.cos_similarity(emb1, emb2)
cos_sim_1_3 = U.cos_similarity(emb1, emb3)
cos_sim_2_3 = U.cos_similarity(emb2, emb3)

print("===== cosine similarity =====")
print("{} : {} = {}".format(word1, word2, cos_sim_1_2))
print("{} : {} = {}".format(word1, word3, cos_sim_1_3))
print("{} : {} = {}".format(word2, word3, cos_sim_2_3))

### 類似単語トップ5

print("\n===== most similar =====")
query=[]
query.append('猫')
print(U.most_similar(query[0], v_to_i, i_to_v, embedding, top=5))

### 類推

print("\n===== analogy =====")
U.analogy('日本', '東京', 'アメリカ', v_to_i, i_to_v, embedding)
U.analogy('王', '男', '女王', v_to_i, i_to_v, embedding)
