'''
word2vecのCBOWの学習を行う。
CNTKバックエンドのKerasを利用する。
'''
import os, pickle
import numpy as np
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Lambda, Dense, dot
from keras.layers.embeddings import Embedding
import cntk as C
import training_settings as G
import batch_data_generator as V_gen
import save_embeddings as S

# 設定を読み込む
k = G.window_size
batch_size = G.batch_size
epoch = G.epoch
context_size = 2*k

# 各種ファイルの設定
train_data = G.train_data
vocab_file = G.vocab_file
vocab_dict_file = G.vocab_dictionary_file
reverse_vocab_dict_file = G.reverse_vocab_dictionary_file
embedding_file = G.embedding_file
model_file = G.model_file
model_onnx_file = G.model_onnx_file

# 単語辞書のPickleファイルをロードする。
print("===== Start data preparation.")
print("      Load sentences")
with open(vocab_dict_file, 'rb') as f:
    vocabulary = pickle.load(f)
with open(reverse_vocab_dict_file, 'rb') as f:
    reverse_vocabulary = pickle.load(f)
G.vocab_size = len(vocabulary)
 
# CBOWのネットワークを構築する。
print("===== Start build network.")

# 埋め込みレイヤーのWeightsの初期値
embedding = np.random.uniform(-1.0/2.0/G.embedding_dimension, 1.0/2.0/G.embedding_dimension, (G.vocab_size+3, G.embedding_dimension))

# CBOWモデルを組み立てる。
# 入力は、中心単語、そのウィンドウサイズ分の周辺単語、ネガティブサンプルとする単語の3つである。
word_index = Input(shape=(1,), name="word_index_input", dtype="float32")
context = Input(shape=(context_size,), name="context_input", dtype="float32")
negative_samples = Input(shape=(G.negative,), name="negative_samples_input", dtype="float32")

# 埋め込みレイヤーを作成する。各入力値共通で学習する。
shared_embedding_layer = Embedding(input_dim=(G.vocab_size+3), 
                                   output_dim=G.embedding_dimension, 
                                   weights=[embedding])
word_embedding = shared_embedding_layer(word_index)
context_embeddings = shared_embedding_layer(context)
negative_words_embedding = shared_embedding_layer(negative_samples)

# 周辺単語を平均化し、埋め込みレイヤーのベクトルを取得する。
cbow = Lambda(lambda x: K.mean(x, axis=1), 
              output_shape=(G.embedding_dimension,))(context_embeddings)

# 周辺単語のベクトルと、中心単語・ネガティブサンプルの単語の内積を求める。
word_context_product = dot([word_embedding, cbow],axes=-1)
negative_context_product = dot([negative_words_embedding, cbow], axes=-1)

# 上記で算出した内積がモデルの出力となる。
model = Model(input=[word_index, context, negative_samples],
              output=[word_context_product, negative_context_product])

# モデルのオプティマイザとしてRMSprop、損失関数としてバイナリ交差エントロピーを使用する。
model.compile(optimizer='rmsprop', loss='binary_crossentropy')
print(model.summary())

# 訓練を開始する。
print("===== Start training.")
model.fit_generator(V_gen.data_generator(train_data, batch_size), epochs=epoch, shuffle=True)

# 訓練結果である埋め込みレイヤーのweightsをテキスト保存する。
S.save_embeddings(embedding_file, shared_embedding_layer.get_weights()[0], vocabulary)
embedding = shared_embedding_layer.get_weights()[0]

### 学習済みモデルをKerasモデルとして保存する。
model.save(model_file)

### 学習済みモデルをONNXモデルとして保存する。
C.combine(model.outputs).save(model_onnx_file, format=C.ModelFormat.ONNX)
