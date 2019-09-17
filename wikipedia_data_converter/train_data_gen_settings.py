'''
設定ファイル
'''
# Unknownワードを表す文字列
UNKNOWN_WORD = "<unk>" 

# 同時実行するプロセス数
total_proc = 4
# 学習対象とする単語の出現頻度(この件数以上の単語を対象とする)
#min_count = 55 
min_count = 60 
# 学習対象とする単語の最低文字数(動詞と形容詞)
min_char = 1
# ウィンドウサイズ
window_size =5
# 頻出単語の学習率を下げるためのパラメータ。0以下の場合は全単語を学習対象とする。
sample = 1e-3
# ネガティブサンプリングする単語数
negative = 5
# 処理中にカウントする総単語数
vocab_size = None
# 処理中にカウントする学習対象の単語数
train_words = 0
# NGワードのリストファイル
NG_WORD_LIST = "./ng_word_list.txt"
# Wikipediaのファイル
sentence_file = "/data/wikipedia/jawiki.txt"
# 上記のWikipediaファイルを一定行数ずつに分割したファイル群
sentence_files = "/data/wikipedia/jawiki_split_*"
# 処理中にMeCabで分かち書きしたファイル群を格納するディレクトリ
wakati_files_dir = "/data/wikipedia/wakati/"
# 各プロセスが作成する学習データファイルのプレフィックス
train_data = "/data/train_data/wiki_prepared_data"
# 抽出した学習対象の単語のリストファイル
vocab_file = "/data/train_data/wiki_vocab.txt"
# 単語とその出現数を保持するPickle
vocab_dictionary_file = '/data/train_data/wiki_vocab_dict.pickle'
# 単語の出現数純に並び替え、インデックスと対応づけているPickle
reverse_vocab_dictionary_file = '/data/train_data/wiki_reverse_vocab_dict.pickle'
# 学習した結果として生成される埋め込みレイヤーのWeightのテキスト化データ
embedding_file = "/data/train_data/wiki_embedding.txt"
# 学習結果のモデルファイル（Keras形式）
model_file = "/data/train_data/wiki_w2v_model.h5"
# 学習結果のモデルファイル（ONNX形式）
model_onnx_file = "/data/train_data/wiki_w2v_model.onnx"

# 学習対象とする文章の最低単語数（頻出単語を除いた件数）
MIN_SENTENCE_LENGTH = 5
