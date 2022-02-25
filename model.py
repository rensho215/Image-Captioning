#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
from os import listdir
from pickle import dump
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import Model
import string
from pickle import load
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from numpy import array
from tensorflow.keras.layers import Input,Dense,LSTM,Embedding,Dropout
from keras.layers.merge import add
from tensorflow.keras.callbacks import ModelCheckpoint

image_directory = 'photo_data'#画像データのパスを指定すること
caption_data = 'caption.txt'#キャプションデータのパスを指定すること
train_data = 'train.txt'#トレーニングデータのパスを指定すること
val_data = 'val.txt'#バリデーションデータのパスを指定すること

# 指定したディレクトリ内の各写真から特徴を抽出する関数
def extract_features(directory):
    model = VGG16()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    features = dict() #特徴を格納するためのディクショナリ
    for name in listdir(directory):
        filename = directory + '/' + name #ファイルから画像を読み込む
        image = load_img(filename, target_size=(224, 224)) #VGG16用に224×224に成形
        image = img_to_array(image) #numpy配列に変換
        image = image.reshape((1, image.shape[0],
        image.shape[1], image.shape[2])) #モデルに読み込ませるために成形
        image = preprocess_input(image) #VGGモデルに画像を読み込ませる
        feature = model.predict(image, verbose=0) #特徴抽出
        image_id = name.split('.')[0] #画像の名前を取得
        features[image_id] = feature #画像の名前と特徴を紐付け
    return features
 
#特徴抽出
features = extract_features(image_directory)
 
#特徴をpklファイルとして保存
dump(features, open('features.pkl', 'wb'))

#ファイルを読み込む関数
def load_doc(filename):
    file = open(filename, 'r', encoding='utf-8')
    text = file.read()
    file.close()
    return text
 
#キャプションデータの読み込み
doc = load_doc(caption_data)

#キャプションと画像名を紐づけする関数
def load_descriptions(doc):
    mapping = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        image_id, image_desc = tokens[0], tokens[1:] #最初の単語を画像名、残り全てをキャプションとして読み込む
        image_id = image_id.split('.')[0] #ピリオドより手前を画像名とする
        image_desc = ' '.join(image_desc) #キャプションの単語を文字列に戻す
        if image_id not in mapping: #その画像名が一つ目ならリストを作成
            mapping[image_id] = list()
        mapping[image_id].append(image_desc) #画像名にキャプションを紐づけてディクショナリに格納
    return mapping
 
#キャプションと画像の紐づけ 
descriptions = load_descriptions(doc)

#余計な記号を除去する関数
def clean_descriptions(descriptions):
    table = str.maketrans('', '', string.punctuation)#記号をリストアップ
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split() #キャプションを単語に区切る
            desc = [w.translate(table) for w in desc] #記号を消去
            desc_list[i] = ' '.join(desc) #キャプションの単語を文字列に戻す
 
#余計な記号を除去する 
clean_descriptions(descriptions)

#語彙が縮小されたキャプションを保存する関数
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w', encoding='utf-8')
    file.write(data)
    file.close()
 
#語彙が縮小されたキャプションをtxtファイルとして保存
save_descriptions(descriptions, 'descriptions.txt')
 
#画像数のチェック
print('Loaded: %d ' % len(descriptions))

#データセットの画像名のリストを作成する関数
def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    for line in doc.split('\n'):
        if len(line) < 1:
            continue
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)
 
#トレーニングデータの画像名のリスト作成
train = load_set(train_data)
#バリデーションデータの画像名のリスト作成
val = load_set(val_data)

#画像名とキャプションを紐付けたディクショナリを作成する関数
def load_clean_descriptions(filename, dataset):#引数datasetはtrainとかvalとか
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):#一行ずつ読み込む
        tokens = line.split() #空白で区切る
        image_id, image_desc = tokens[0], tokens[1:] #最初の単語を画像名、残り全てをキャプションとして読み込む
        if image_id in dataset: #画像名がデータセット中に指定されていれば以下を実行
            if image_id not in descriptions: #その画像名が一つ目ならリストを作成
                descriptions[image_id] = list()
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq' #キャプションを開始語と終了語で囲む
            descriptions[image_id].append(desc) #ディクショナリに格納
    return descriptions
 
#トレーニングデータのキャプションと画像名を紐付ける
train_descriptions = load_clean_descriptions('descriptions.txt', train)
#バリデーションデータのキャプションと画像名を紐付ける
val_descriptions = load_clean_descriptions('descriptions.txt', val)

#画像の特徴量を読み込む関数
def load_photo_features(filename, dataset):
    all_features = load(open(filename, 'rb'))
    features = {k: all_features[k] for k in dataset}#画像名と特徴量を紐づけてディクショナリに格納
    return features
 
#トレーニングデータの特徴量と画像名を紐付ける
train_features = load_photo_features('features.pkl', train)
#バリデーションデータの特徴量と画像名を紐付ける
val_features = load_photo_features('features.pkl', val)
 
#キャプションのディクショナリをリストにする関数
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc
 
#キャプションをKerasのTokenizerで扱うために変換する
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer
 
#tokenizerを準備する
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1

#最も多くの単語を含むキャプションの長さを計算する関数
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)
 
#最大シーケンス長を計算する
max_length = max_length(train_descriptions)

#画像と出力単語を紐づける関数
def create_sequences(tokenizer, max_length, descriptions, photos):
    X1, X2, y = list(), list(), list()#X1が入力画像、X2が入力語、yがX1とX2に対応する出力語
    #各画像名でループ
    for key, desc_list in descriptions.items():
        #各画像のキャプションでループ
        for desc in desc_list:
            #シーケンスをエンコードする
            seq = tokenizer.texts_to_sequences([desc])[0]
            #1つのシーケンスを複数のX、Yペアに分割する
            for i in range(1, len(seq)):
                #入力と出力のペアに分割する
                in_seq, out_seq = seq[:i], seq[i]
                #行列のサイズを最大の単語数に合わせる
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                #出力シーケンス
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                #全てをarrayに格納
                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)
    return array(X1), array(X2), array(y)
 
#トレーニングデータの入力画像、入力語、出力語を紐付ける
X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features)
#バリデーションデータの入力画像、入力語、出力語を紐付ける
X1val, X2val, yval = create_sequences(tokenizer, max_length, val_descriptions, val_features)

#モデルを定義する関数
def define_model(vocab_size, max_length):
    #画像の特徴を入力するレイヤ
    inputs1 = Input(shape=(1000,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    #文章を入力するレイヤ
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    #上の二つの出力を統合する部分
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    #モデルの定義．二つを入力にとって一つを出力する形になる
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

#モデルの定義
model = define_model(vocab_size, max_length)
#コールバックを定義する
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
#学習
model.fit([X1train, X2train], ytrain, epochs=10, verbose=2, callbacks=[checkpoint], validation_data=([X1val, X2val], yval))


# In[2]:



from numpy import argmax
from keras.models import load_model

#ディレクトリ内の写真から特徴を抽出する
def extract_features_from_image(filename):
    model = VGG16()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature
 
photo = extract_features_from_image('gazou/sample3.jpg')#テスト画像のパスを指定すること

#整数を単語に変換する関数
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
 
#画像からキャプションを生成する関数
def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

model = load_model('model-ep005-loss2.644-val_loss3.245.h5')#最後に出力されたモデルを指定する
description = generate_desc(model, tokenizer, photo, max_length)
print(description)

