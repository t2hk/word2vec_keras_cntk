# word2vec-cbow-keras-cntk
The implementation of word2vec cbow in Keras using CNTK backend.

## Overview
This is the implementation of word2vec cbow and data converter for training data.

* The wikipedia data converter

  To convert Japanese wikipedia text file to training data that is input to the cbow model.
  The Japanese Wikipedia text data is https://dumps.wikimedia.org/jawiki/latest/

* The word2vec cbow model

  Training and saving the model with Keras using CNTK backend.
  The model format is ONNX.

* Test  

  Try to analogize using the learned model.
  
  - cosine similarity
  - most similar
  - analogy

## Requirements
I developed in the following environment.

| Software | version | 
|---|---|
| Ubuntu | 16.04.6 LTS |
| Python | 3.6.8 |
| Keras | 2.2.4 |
| MeCab | 0.996 |
| mecab-python3 | 0.996.2 | 
| CNTK | 2.7 |
| CUDA | 10.1 |

# How to use
### Data Converter
  1. download the japanese wikipedia full data and convert to text from xml by wp2txt and so on.
  2. edit setting file "train_data_gen_settings.py"
  3. execute "gen_prepared_data_multi.py"

     ```
     $ python gen_prepared_data_multi.py
     ```
### Training and Saving
  1. edit setting file "training_settings.py"
  2. execute "cbow_train_onnx.py"

     ```
     $ python cbow_train_onnx.py
     ```

### Test
  1. edit setting file "training_settings.py"
  2. execute "cbow_eval.py"

     ```
     $ python cbow_eval.py
     ```

## Thanks
I refered the following project. Thank you.

  - https://github.com/abaheti95/Deep-Learning
  - https://github.com/oreilly-japan/deep-learning-from-scratch-2
