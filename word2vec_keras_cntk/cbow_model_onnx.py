from __future__ import absolute_import

from keras import backend as K
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Lambda, Dense, merge, dot
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from keras.objectives import mse

import training_settings as G
from sentences_generator import Sentences
import vocab_generator as V_gen
import save_embeddings as S
#from winmltools import convert_coreml
#import winmltools

import cntk as C

def cos_similarity(x, y, eps=1e-8):
    '''コサイン類似度の算出

    :param x: ベクトル
    :param y: ベクトル
    :param eps: ”0割り”防止のための微小値
    :return:
    '''
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)


k = G.window_size # context windows size
batch_size = G.batch_size
epoch = G.epoch
context_size = 2*k
sentence_file = G.sentence_file
train_data = G.train_data
vocab_file = G.vocab_file
model_file = G.model_file
model_onnx_file = G.model_onnx_file

# Creating a sentence generator from demo file
#sentences = Sentences("test_file.txt")
sentences = Sentences(sentence_file)
vocabulary = dict()
V_gen.build_vocabulary(vocabulary, sentences)
V_gen.filter_vocabulary_based_on(vocabulary, G.min_count)
reverse_vocabulary = V_gen.generate_inverse_vocabulary_lookup(vocabulary, vocab_file)

# generate embedding matrix with all values between -1/2d, 1/2d
embedding = np.random.uniform(-1.0/2.0/G.embedding_dimension, 1.0/2.0/G.embedding_dimension, (G.vocab_size+3, G.embedding_dimension))

# Creating CBOW model
# Model has 3 inputs
# Current word index, context words indexes and negative sampled word indexes
word_index = Input(shape=(1,), name="word_index_input", dtype="float32")
context = Input(shape=(context_size,), name="context_input", dtype="float32")
negative_samples = Input(shape=(G.negative,), name="negative_samples_input", dtype="float32")

# All the inputs are processed through a common embedding layer
shared_embedding_layer = Embedding(input_dim=(G.vocab_size+3), 
                                   output_dim=G.embedding_dimension, 
                                   weights=[embedding])
word_embedding = shared_embedding_layer(word_index)
context_embeddings = shared_embedding_layer(context)
negative_words_embedding = shared_embedding_layer(negative_samples)

# Now the context words are averaged to get the CBOW vector
cbow = Lambda(lambda x: K.mean(x, axis=1), 
              output_shape=(G.embedding_dimension,))(context_embeddings)

# The context is multiplied (dot product) with current word and negative sampled words
#word_context_product = merge([word_embedding, cbow], mode='dot')
word_context_product = dot([word_embedding, cbow],axes=-1)

#negative_context_product = merge([negative_words_embedding, cbow], mode='dot', concat_axis=-1)
negative_context_product = dot([negative_words_embedding, cbow], axes=-1)

# The dot products are outputted
model = Model(input=[word_index, context, negative_samples],
              output=[word_context_product, negative_context_product])

# binary crossentropy is applied on the output
model.compile(optimizer='rmsprop', loss='binary_crossentropy')
print(model.summary())

model.fit_generator(V_gen.data_generator(train_data, batch_size), epochs=epoch, shuffle=True)

# Save the trained embedding
S.save_embeddings("embedding.txt", shared_embedding_layer.get_weights()[0], vocabulary)
embedding = shared_embedding_layer.get_weights()[0]

### export Keras model ###
model.save(model_file)

### export ONNX model ###
C.combine(model.outputs).save(model_onnx_file, format=C.ModelFormat.ONNX)

load_model = C.Function.load(model_onnx_file, device=C.device.gpu(0), format=C.ModelFormat.ONNX)

### following lines are commented

vocab_key_val = {v: k for k, v in reverse_vocabulary.items()}

input_context = np.random.randint(10, size=(1, context_size))
#input_word = np.random.randint(10, size=(1,))
#input_word = np.array(reverse_vocabulary['映画']).reshape(1,)
input_word = np.array(reverse_vocabulary['東京']).reshape(1,)
input_context[0] = reverse_vocabulary['ワシントン']
#input_negative = np.random.randint(10, size=(1, G.negative))
input_negative = np.zeros(G.negative).reshape(1, G.negative)
input_negative[0] = reverse_vocabulary['日本']

emb1 = embedding[reverse_vocabulary['日本']]
emb2 = embedding[reverse_vocabulary['東京']]
emb3 = embedding[reverse_vocabulary['ワシントン']]

sim1_2 = cos_similarity(emb1, emb2)
sim1_3 = cos_similarity(emb1, emb3)
sim2_3 = cos_similarity(emb2, emb3)

print("emb1 : emb2 = {}".format(sim1_2))
print("emb1 : emb3 = {}".format(sim1_3))
print("emb2 : emb3 = {}".format(sim2_3))

cont_val = []
for key in input_context[0]:
  if key in vocab_key_val: 
    cont_val.append(vocab_key_val[key])

word_val = []
for key in input_word:
  if key in vocab_key_val: 
    word_val.append(vocab_key_val[key])

negative_val = []
for key in input_negative[0]:
  if key in vocab_key_val: 
    negative_val.append(vocab_key_val[key])

print( "word, context, negative samples, val")
print( input_word.shape, input_word, word_val)
print( input_context.shape, input_context, cont_val)
print( input_negative.shape, input_negative, negative_val)

output_dot_product, output_negative_product = model.predict([input_word, input_context, input_negative])
print( "word cbow dot product")
print( output_dot_product.shape, output_dot_product)
# print "cbow negative dot product"
# print output_negative_product.shape, output_negative_product
