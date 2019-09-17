'''
設定ファイル
'''
# Unknownワードを表す文字列
UNKNOWN_WORD = "<unk>"

#埋め込みレイヤーの次元数
embedding_dimension = 300
#学習する単語のウィンドウサイズ
window_size = 5
#ネガティブサンプルの単語数
negative = 5
# 単語の総数
vocab_size = None
train_words = None
# 学習のエポック数
epoch = 5
# 学習のバッチサイズ
batch_size = 10000

train_data = "/data/train_data/wiki_prepared_data.txt"
vocab_file = "/data/train_data/wiki_vocab.txt"
vocab_dictionary_file = '/data/train_data/wiki_vocab_dict.pickle'
reverse_vocab_dictionary_file = '/data/train_data/wiki_reverse_vocab_dict.pickle'
embedding_file = "/data/train_data/wiki_embedding.txt"
model_file = "/data/train_data/wiki_w2v_model.h2"
model_onnx_file = "/data/train_data/wiki_w2v_model.onnx"
