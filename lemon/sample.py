# ライブラリのインポート
import numpy as np
import pandas as pd
# import efficientnet.keras as efn 
import zipfile
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold
from keras.applications.vgg16 import VGG16
import efficientnet.keras as efn 

from keras.layers import Input,GlobalAveragePooling2D,Flatten,Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint

#zipの解凍
def unzip_dataset(INPATH,OUTPATH):
    with zipfile.ZipFile(INPATH) as zf:
        zf.extractall(OUTPATH)

#train_zip
unzip_dataset(INPATH='../input/quest_images_noleak.zip',OUTPATH='./input/')

#test_zip
unzip_dataset(INPATH='../input/quest_test_images_noleak.zip',OUTPATH='./input/')

#train&test
unzip_dataset(INPATH='../input/quest_images_all_noleak.zip',OUTPATH='./input/')

# 学習用マッピングcsv
quest_train_images = pd.read_csv('../input/quest_train_images.csv')
quest_train_images['class_num']=quest_train_images['class_num'].astype(str)

# テスト用マッピングcsv
quest_test_images = pd.read_csv('../input/quest_test_images.csv')
quest_test_images['class_num']=quest_test_images['class_num'].astype(str)

BS = 16
EPOCH = 50

# 学習用
train_gen = ImageDataGenerator(
        rotation_range=360,     #rotation_range:画像をランダムに回転する回転範囲
        width_shift_range=0.2,  #width_shift_range:浮動小数点数（横幅に対する割合）．ランダムに水平シフトする範囲．
        height_shift_range=0.2, #height_shift_range:浮動小数点数（縦幅に対する割合）．ランダムに垂直シフトする範囲．
        horizontal_flip=True,   #horizontal_flip:真理値．水平方向に入力をランダムに反転します．
        vertical_flip = True,   #vertical_flip:真理値．垂直方向に入力をランダムに反転します
        zoom_range = [0.5, 1.0],#zoom_range:浮動小数点数または[lower，upper]．ランダムにズームする範囲．
        shear_range = 0.1,      #shear_range:浮動小数点数．シアー強度（反時計回りのシアー角度）
        validation_split=0.3    #validation_split:浮動小数点数．検証のために予約しておく画像の割合
    )

# テスト用
test_gen = ImageDataGenerator(
        rotation_range=360,
    )

# ネットワークの定義(vgg16)
def create_vgg16_model():
    # 入力の定義
    inputs = Input(shape=(64, 64, 3))
    # VGGネットワークの作成
    vgg = VGG16(include_top=False, weights=None)
    vgg.load_weights('../input/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    # VGGネットワークに画像を入力
    x = vgg(inputs)
    # Poolingレイヤーの追加と入力
    x = GlobalAveragePooling2D()(x)
    # 全結合層の追加と入力
    x = Dense(4,activation='softmax')(x)
    # 完成形のモデルとして再定義
    model = Model(inputs=inputs, outputs=x)
    # 損失関数、最適化手法の指定
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    return model

# ネットワークの定義(enet)
def create_enet_model():
    # 入力の定義
    inputs = Input(shape=(64, 64, 3))
    # ENETネットワークの作成
    enet = efn.EfficientNetB0(include_top=False, weights=None)
    enet.load_weights('../input/efficientnet-b0_imagenet_1000_notop.h5')
    # ENTネットワークに画像を入力
    x = enet(inputs)
    # Poolingレイヤーの追加と入力
    x = GlobalAveragePooling2D()(x)
    # 全結合層の追加と入力
    x = Dense(4,activation='softmax')(x)
    # 完成形のモデルとして再定義
    model = Model(inputs=inputs, outputs=x)
    # 損失関数、最適化手法の指定
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    return model

kf = StratifiedKFold(n_splits=3,shuffle=True,random_state=0)
for fold,(train_index, test_index) in enumerate(kf.split(quest_train_images, quest_train_images['class_num'])):
    print('FOLD{}'.format(fold))
    
    trainData = quest_train_images.iloc[train_index]
    testData = quest_train_images.iloc[test_index]
    
    #学習データの読み込み
    train_generator = train_gen.flow_from_dataframe(
            dataframe = trainData,       #dataframe:ディレクトリから画像への相対パス
            directory = './input/quest_images_noleak/', #directory:画像を読み込むディレクトリのパス．
            x_col='id',                           #x_col:ファイル名を含むdataframeの列名
            y_col='class_num',                    #y_col:ターゲットとなるデータを含むdataframeの列名
            subset='training',                    #subset:データのサブセット（"training"か"validation")．
            batch_size=BS,                         #batch_size:データのバッチのサイズ（デフォルト: 32）
            shuffle=True,                         #shuffle:データをシャッフルするかどうか（デフォルト: True）
            class_mode='categorical',             #class_mode:"categorical"、"binary"、"sparse"、"input"、"raw"、"None"のいずれか1つ．　　　　　　　
            target_size=(64,64)                 #target_size:画像の（height， width）．デフォルトは（256， 256）
        )
    
    #検証データの読み込み
    valid_generator = train_gen.flow_from_dataframe(
            dataframe = testData,                 #dataframe:ディレクトリから画像への相対パス
            directory = './input/quest_images_noleak/', #directory:画像を読み込むディレクトリのパス．
            x_col='id',                           #x_col:ファイル名を含むdataframeの列名
            y_col='class_num',                    #y_col:ターゲットとなるデータを含むdataframeの列名
            subset='validation',                  #subset:データのサブセット（"training"か"validation")．
            batch_size=BS,                        #batch_size:データのバッチのサイズ（デフォルト: 32）
            shuffle=True,                         #shuffle:データをシャッフルするかどうか（デフォルト: True）
            class_mode='categorical',             #class_mode:"categorical"、"binary"、"sparse"、"input"、"raw"、"None"のいずれか1つ．　　　　　　　
            target_size=(64,64)                 #target_size:画像の（height， width）．デフォルトは（256， 256）
        )
    
    # チェックポイント設定
    ckp = ModelCheckpoint(filepath = f'./input/model_{fold}.hdf5',
                          monitor='val_loss',
                          save_best_only=True,
                          save_weights_only=True)
                          

    # modelを作成してfitting
    model = create_vgg16_model()
    model.fit(
        train_generator,
        validation_data = valid_generator,
        epochs = EPOCH,
        callbacks=[ckp]
    )

    # 1FOLDのみにしたいときはbreak
    break

! ls ./input/

# テスト用のGenerator定義
test_gen = ImageDataGenerator(
        rotation_range=360,
    )

test_generator = test_gen.flow_from_dataframe(
        dataframe = quest_test_images,             #dataframe:ディレクトリから画像への相対パス
        directory = './input/quest_test_images_noleak/', #directory:画像を読み込むディレクトリのパス．
        x_col='id',                                #x_col:ファイル名を含むdataframeの列名
        y_col='class_num',                         #y_col:ターゲットとなるデータを含むdataframeの列名
        batch_size=1,                              #batch_size:データのバッチのサイズ（デフォルト: 32）
        shuffle=False,                             #shuffle:データをシャッフルするかどうか（デフォルト: True）　　　　　　
        target_size=(64,64)                        #target_size:画像の（height， width）．デフォルトは（256， 256）
    )

# 空のモデル作成
trained_model = create_vgg16_model()

# weightの読み込み
#trained_model.load_weights('./input/efficientnet-b0_imagenet_1000_notop.h5')

# 予測値の算出
preds = np.argmax(trained_model.predict(test_generator),axis=1)
quest_test_images['class_num'] = preds

quest_test_images.to_csv('submit.csv',index=False,header=False