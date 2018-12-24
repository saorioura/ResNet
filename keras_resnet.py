import tensorflow as tf
from keras.layers import Conv2D, Activation, BatchNormalization, Add, Input, GlobalAveragePooling2D, Dense, MaxPool2D
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.regularizers import l2
import time
import pickle
import os
import numpy as np
from tensorflow.python.keras import backend as K
import struct
from datetime import datetime
import argparse

IMAGE_H = 100
IMAGE_W = 100
NUM_CLASS = 2


# 経過時間用のコールバック
class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_start_time)


class TensorBoardWrapper(TensorBoard):
    '''Sets the self.validation_data property for use with TensorBoard callback.'''

    def __init__(self, batch_gen, nb_steps, **kwargs):
        super().__init__(**kwargs)
        self.batch_gen = batch_gen  # The generator.
        self.nb_steps = nb_steps  # Number of times to call next() on the generator.

    def on_epoch_end(self, epoch, logs):
        # Fill in the `validation_data` property. Obviously this is specific to how your generator works.
        # Below is an example that yields images and classification tags.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        imgs, tags = None, None
        for s in range(self.nb_steps):
            ib, tb = next(self.batch_gen)
            if imgs is None and tags is None:
                imgs = np.zeros((self.nb_steps * ib.shape[0], *ib.shape[1:]), dtype=np.float32)
                tags = np.zeros((self.nb_steps * tb.shape[0], *tb.shape[1:]), dtype=np.uint8)
            imgs[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
            tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb
        self.validation_data = [imgs, tags, np.ones(imgs.shape[0]), 0.0]
        return super().on_epoch_end(epoch, logs)


class ResNet:
    def __init__(self, n, framework, channels_first=False, initial_lr=0.01, nb_epochs=100):
        self.n = n
        self.framework = framework
        # 論文通りの初期学習率=0.1だと発散するので0.01にする
        self.initial_lr = initial_lr
        self.nb_epochs = nb_epochs
        self.weight_decay = 0.0005
        # MX-Netではchannels_firstなのでその対応をする
        self.channels_first = channels_first
        self.data_format = "channels_first" if channels_first else "channels_last"
        self.bn_axis = 1 if channels_first else -1
        # Make model
        #self.model = self.make_model
        self.savepath = './keras_log/{0}'.format(datetime.now().strftime("%Y%m%d%H%M%S"))

    # resnet1 conv -> bn -> relu -> conv -> bn -> add -> relu
    # resnet2 bn -> relu -> conv -> bn -> relu -> conv -> add こっち。上の方を使用する際には最初にbn, relu追加必要。
    def block(self, channles, input_tensor):
        with tf.name_scope('block'):
            # ショートカット元
            shortcut = input_tensor
            # メイン側
            x = BatchNormalization(axis=self.bn_axis)(input_tensor)
            x = Activation("relu")(x)
            x = Conv2D(channles, kernel_size=3, padding="same", data_format=self.data_format,
                       kernel_regularizer=l2(self.weight_decay))(x)
            x = BatchNormalization(axis=self.bn_axis)(x)
            x = Activation("relu")(x)
            x = Conv2D(channles, kernel_size=3, padding="same", data_format=self.data_format,
                       kernel_regularizer=l2(self.weight_decay))(x)
            # 結合
            return Add()([x, shortcut])

    @property
    def make_model(self):
        input = Input(shape=(3, IMAGE_H, IMAGE_H)) if self.channels_first else Input(shape=(IMAGE_H, IMAGE_W, 3))
        # conv
        x = Conv2D(64, kernel_size=3, padding="same", data_format=self.data_format,
                   kernel_regularizer=l2(self.weight_decay))(input)
        # maxpool
        x = MaxPool2D(pool_size=3, strides=2, padding='SAME')(x)

        # resiual block(bottle neckなし)
        for i in range(self.n):
            x = self.block(64, x)

        # bn, relu (resnet1の方を使うなら不要)
        x = BatchNormalization(axis=self.bn_axis)(x)
        x = Activation("relu")(x)

        # Global Average Pooling
        x = GlobalAveragePooling2D(data_format=self.data_format)(x)
        x = Dense(NUM_CLASS, activation="softmax")(x)
        # model
        model = Model(input, x)
        return model

    def lr_schduler(self, epoch):
        x = self.initial_lr
        if epoch >= self.nb_epochs * 0.5: x /= 10.0
        if epoch >= self.nb_epochs * 0.75: x /= 10.0
        return x

    def train(self, X_train, y_train, X_val, y_val, weightpath):
        # コンパイル
        if weightpath is None:
            print("Make New Model")
            self.model = self.model = self.make_model
            self.model.compile(optimizer=SGD(lr=self.initial_lr, momentum=0.9), loss="categorical_crossentropy",
                           metrics=["acc"])
        else:
            # weight restore
            print("Load Model")
            self.model = load_model(weightpath)
        # Data Augmentation
        traingen = ImageDataGenerator(
            rescale=1. / 255,
            width_shift_range=4. / IMAGE_W,
            height_shift_range=4. / IMAGE_H,
            horizontal_flip=True)
        valgen = ImageDataGenerator(
            rescale=1. / 255)
        # Callback
        time_cb = TimeHistory()
        lr_cb = LearningRateScheduler(self.lr_schduler)
        # tb追加
        # tb_cb = TensorBoard(log_dir='./keras_log',
        #                     histogram_freq=0, write_graph=True, write_images=True)
        tb_cb = TensorBoardWrapper(valgen.flow(X_val, y_val), nb_steps=5, log_dir=self.savepath + '/tb',
                                   histogram_freq=1,
                                   batch_size=32, write_graph=False, write_grads=True)
        mc_cb = ModelCheckpoint(filepath=self.savepath + '/weights.{epoch:04d}-{val_loss:.2f}.hdf5', monitor='val_loss',
                                verbose=1, save_best_only=True, mode='auto')
        # Train
        print("Start Training")
        history = self.model.fit_generator(traingen.flow(X_train, y_train, batch_size=128), epochs=self.nb_epochs,
                                           steps_per_epoch=len(X_train) / 128,
                                           validation_data=valgen.flow(X_val, y_val),
                                           callbacks=[time_cb, lr_cb, tb_cb, mc_cb]).history
        history["time"] = time_cb.times
        # Save history
        file_name = f"{self.framework}_n{self.n}.dat"
        with open(file_name, "wb") as fp:
            pickle.dump(history, fp)

    def test(self, X_val, y_val, weightpath):
        # restore
        print("Load Model")
        self.model = load_model(weightpath)
        print("Start Evaluation...")
        score = self.model.predict(X_val, verbose=0)
        print('Prediction : Ans')
        print(np.hstack((score, y_val)))
        score = self.model.evaluate(X_val, y_val, verbose=0)
        print('loss and accuracy', score)


def load_batch(fpath, label_key='labels'):
    labels = np.array([], dtype=np.uint8)
    data = np.array([], dtype=np.uint8)
    with open(fpath, 'rb') as f:
        while True:
            label = f.read(1)
            if len(label) == 0:
                break
            labels = np.append(labels, ord(label))
            img = np.array(struct.unpack('30000B', f.read(3 * 100 * 100)), dtype=np.uint8)
            data = np.append(data, img)
    data = data.reshape(labels.shape[0], 3, 100, 100)
    return data, labels


def load_data():
    """Loads binary dataset.
    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = 'C:/Users/saori/PycharmProjects/data/original-data-bin/'

    fpath = os.path.join(dirname, 'train_batch_1.bin')
    (x_train, y_train) = load_batch(fpath)

    fpath = os.path.join(dirname, 'test_batch.bin')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)


# Main function
def main(framework, n=3, istrain=True, epochs=100, weightpath=None):
    # layers = 6n+2
    net = ResNet(n, framework, nb_epochs=epochs)
    # CIFAR
    (X_train, y_train), (X_test, y_test) = load_data()
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)
    if istrain:
        # train
        net.train(X_train, y_train, X_test, y_test, weightpath)
    else:
        if weightpath is None:
            print('set the weight path')
            return;
        net.test(X_test, y_test, weightpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-blocks', type=int, default=3, help='number of blocks, default=3')
    parser.add_argument('-weightpath', type=str, default=None, help='weight path')
    parser.add_argument('-epochs', type=int, default=100, help='epochs, default=100')
    # -aや--allが指定されれば True, なければ False が args.all に格納される
    parser.add_argument('-test', '--test', action='store_false')
    args = parser.parse_args()

    main("keras_resnet", n=args.blocks, istrain=args.test, epochs=args.epochs, weightpath=args.weightpath)
