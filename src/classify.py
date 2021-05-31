import argparse
import csv
import datetime
import json
import os

import keras.utils.data_utils
import numpy as np
import pandas as pd
# import oyaml as yaml
import pickle as pk

import keras
import tensorflow as tf
from keras.layers import Input, Dense, TimeDistributed, Dropout, BatchNormalization
from keras.models import Model
from keras import regularizers
from keras.optimizers import Adam
import keras.backend as K
from autopool import AutoPool1D
import random
from keras import Model
# Generators
from keras.models import Sequential
from data_gen import AudioUtil
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.random.seed(0)
random.seed(0)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)
tf.compat.v1.keras.backend.set_session(session)


## HELPERS

def load_embeddings(file_list, emb_dir):
    """
    Load saved embeddings from an embedding directory
    Parameters
    ----------
    file_list
    emb_dir
    Returns
    -------
    embeddings
    ignore_idxs
    """
    embeddings = []
    for idx, filename in enumerate(file_list):
        emb_path = os.path.join(emb_dir, os.path.splitext(filename)[0] + '.npz')
        embeddings.append(np.load(emb_path)['embedding'])

    return embeddings


## MODEL CONSTRUCTION
def construct_mlp(input_size, num_classes,
                  dropout_size=0.5, ef_mode=4, l2_reg=1e-5):
    """
    Construct a MLP model for urban sound tagging.
    Parameters
    ----------
    input_size
    num_classes
    dropout_size
    ef_mode
    l2_reg
    Returns
    -------
    model
    """

    # Add hidden layers
    from keras.layers import Flatten, Conv1D, Conv2D, GlobalMaxPooling1D, GlobalAveragePooling1D, LSTM, Concatenate, \
        GlobalAveragePooling2D, LeakyReLU

    import efficientnet.keras as efn
    base_model = None

    if ef_mode == 0:
        base_model = efn.EfficientNetB0(weights='noisy-student', include_top=False, pooling='avg')
    elif ef_mode == 1:
        base_model = efn.EfficientNetB1(weights='noisy-student', include_top=False, pooling='avg')
    elif ef_mode == 2:
        base_model = efn.EfficientNetB2(weights='noisy-student', include_top=False, pooling='avg')
    elif ef_mode == 3:
        base_model = efn.EfficientNetB3(weights='noisy-student', include_top=False, pooling='avg')
    elif ef_mode == 4:
        base_model = efn.EfficientNetB4(weights='noisy-student', include_top=False,
                                        pooling='avg')  # imagenet or weights='noisy-student'
    elif ef_mode == 5:
        base_model = efn.EfficientNetB5(weights='noisy-student', include_top=False, pooling='avg')
    elif ef_mode == 6:
        base_model = efn.EfficientNetB6(weights='noisy-student', include_top=False, pooling='avg')
    elif ef_mode == 7:
        base_model = efn.EfficientNetB7(weights='noisy-student', include_top=False, pooling='avg')

    input1 = Input(shape=input_size, dtype='float32', name='input')
    # input2 = Input(shape=(num_frames, 85), dtype='float32', name='input2')  # 1621
    assert base_model is not None
    y = TimeDistributed(base_model)(input1)
    y = TimeDistributed(Dropout(dropout_size))(y)
    # y = Concatenate()([y, input2])
    y = TimeDistributed(Dense(num_classes, activation='sigmoid', kernel_regularizer=regularizers.l2(l2_reg)))(y)
    y = AutoPool1D(axis=1, name='output')(y)

    m = Model(inputs=input1, outputs=y)
    m.summary(line_length=100)
    # m.name = 'urban_sound_classifier'

    return m


def train_model(base_model, training_dataset, validation_dataset, output_dir,
                loss=None, num_epochs=100, patience=20,
                learning_rate=1e-4):
    """
    Train a model with the given data.
    Parameters
    ----------
    base_model
    training_dataset
    validation_dataset
    output_dir
    loss
    num_epochs
    patience
    learning_rate
    Returns
    -------
    history
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set up callbacks
    cb = []
    # checkpoint
    model_weight_file = os.path.join(output_dir, 'model_best.h5')
    cb.append(tf.keras.callbacks.ModelCheckpoint(output_dir + '\\{epoch:02d}-{val_loss:.2f}_model_best.h5', verbose=1,
                                                 save_weights_only=True,
                                                 save_best_only=False,
                                                 monitor='val_loss'))  # val_loss

    cb.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1,
                                               patience=patience))

    history_csv_file = os.path.join(output_dir, 'history.csv')
    cb.append(tf.keras.callbacks.CSVLogger(history_csv_file, append=True,
                                           separator=','))

    # model = ModelMGPU(base_model, gpus=2) FIXME:
    model = base_model
    model.compile(Adam(learning_rate=learning_rate), loss=loss)

    history = model.fit(training_dataset, validation_data=validation_dataset,
                        # steps_per_epoch=846,
                        epochs=num_epochs, callbacks=cb, verbose=1, shuffle=False, use_multiprocessing=True,
                        workers=8)
    return history


def train_model_ease(base_model, X, Y, batch_size, test_X, test_Y, output_dir,
                     loss=None, num_epochs=100, patience=20,
                     learning_rate=1e-4):
    os.makedirs(output_dir, exist_ok=True)

    # Set up callbacks
    cb = []
    # checkpoint
    model_weight_file = os.path.join(output_dir, 'model_best.h5')
    cb.append(tf.keras.callbacks.ModelCheckpoint(output_dir + '\\{epoch:02d}-{val_loss:.2f}_model_best.h5', verbose=1,
                                                 save_weights_only=True,
                                                 save_best_only=False,
                                                 monitor='val_loss'))  # val_loss

    cb.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1,
                                               patience=patience))

    history_csv_file = os.path.join(output_dir, 'history.csv')
    cb.append(tf.keras.callbacks.CSVLogger(history_csv_file, append=True,
                                           separator=','))

    # model = ModelMGPU(base_model, gpus=2) FIXME:
    model = base_model
    model.compile(Adam(learning_rate=learning_rate), loss=loss, metrics=[keras.metrics.sparse_categorical_accuracy])

    history = model.fit(X, Y, batch_size=batch_size, validation_data=(test_X, test_Y),
                        # steps_per_epoch=846,
                        epochs=num_epochs, callbacks=cb, verbose=1, shuffle=False, use_multiprocessing=True,
                        workers=8)
    return history



## MODEL TRAINING
def train(annotation_path, num_classes, emb_dir, output_dir, exp_id,
          batch_size=64, num_epochs=1000,
          patience=20, learning_rate=1e-4, dropout_size=0.5,
          ef_mode=4, l2_reg=1e-5, standardize=True,
          timestamp=None, random_state=0):
    """
    Train and evaluate a MIL MLP model.
    Parameters
    ----------
    annotation_path
    emb_dir
    output_dir
    batch_size
    num_epochs
    patience
    learning_rate
    dropout_size
    l2_reg
    standardize
    timestamp
    random_state
    Returns
    -------
    """
    np.random.seed(random_state)
    random.seed(random_state)

    # Load annotations and taxonomy
    print("* Loading dataset.")
    annotation_data = pd.read_csv(annotation_path).sort_values('filename')
    # with open(taxonomy_path, 'r') as f:
    #     taxonomy = yaml.load(f, Loader=yaml.Loader)

    annotation_data_trunc = annotation_data[['filename',
                                             'fold',
                                             'target',
                                             'category',
                                             'esc10',
                                             'src_file',
                                             'take']].drop_duplicates()
    file_list = annotation_data_trunc['filename'].to_list()

    labels = annotation_data_trunc['target'].to_list()

    print("* Preparing training data.")

    print(np.shape(labels))

    embeddings = load_embeddings(file_list, emb_dir)
    Conbin = list(zip(embeddings, labels))
    random.shuffle(Conbin)
    embeddings[:], labels[:] = zip(*Conbin)

    # embeddings = [AudioUtil.spectro_augment(x) for x in embeddings]

    embeddings = np.asarray(embeddings)
    print(embeddings.shape)
    labels = np.asarray(labels)

    train_X, test_X, train_Y, test_Y = train_test_split(embeddings, labels, test_size=0.2)

    print(np.shape(embeddings[0]))

    dim = np.shape(embeddings[0])
    num_frames = len(embeddings[0])
    print(num_frames)

    params = {'dim': dim,
              'batch_size': batch_size,
              'shuffle': True}
    print("#batch_size", batch_size, "#dim", dim)

    # scaler = None
    # training_dataset = SoundDS(train_files, train_targets,
    #                            num_classes, emb_dir, **params)
    # validation_dataset = SoundDS(test_files, test_targets,
    #                              num_classes, emb_dir, **params)

    model = construct_mlp(dim,
                          num_classes,
                          ef_mode=ef_mode,
                          dropout_size=dropout_size,
                          l2_reg=l2_reg)

    if not timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    results_dir = os.path.join(output_dir, exp_id, timestamp)

    loss_func = keras.backend.sparse_categorical_crossentropy
    training = True
    prediction = False

    if training:
        # history = train_model(model, training_dataset, validation_dataset,
        #                       results_dir, loss=loss_func,
        #                       num_epochs=num_epochs,
        #                       patience=patience, learning_rate=learning_rate)
        history = train_model_ease(model, train_X, train_Y, batch_size, test_X, test_Y,
                                   results_dir, loss=loss_func,
                                   num_epochs=num_epochs,
                                   patience=patience, learning_rate=learning_rate)
        test_loss, test_acc = model.evaluate(test_X, test_Y, batch_size, use_multiprocessing=True, workers=8)
        print("test_loss", test_loss, "test_acc", test_acc)
    # Reload checkpointed file
    if prediction:

        model = construct_mlp(dim,
                              num_classes,
                              ef_mode=ef_mode,
                              dropout_size=dropout_size,
                              l2_reg=l2_reg)
        params = {'dim': dim,
                  'batch_size': 1,
                  'shuffle': False}
        # training_dataset = SoundDS(train_files, train_targets,
        #                            num_classes, emb_dir, **params)
        # validation_dataset = SoundDS(test_files, test_targets,
        #                              num_classes, emb_dir, **params)

        out_dir = os.listdir(results_dir)
        out_dir.sort()
        count = 0

        for i in out_dir:
            if i[-2:] == "h5":
                model_weight_file = os.path.join(results_dir, i)
                model.load_weights(model_weight_file)

                print("* Saving model predictions.")
                results = {'train': model.predict(train_X, batch_size, use_multiprocessing=True, workers=8).tolist(),
                           'validate': model.predict(test_X, batch_size, use_multiprocessing=True, workers=8).tolist()}

                results_path = os.path.join(results_dir, "results.json")
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2)

                print(np.shape(results['validate']))

                count = count + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_path")
    parser.add_argument("num_classes", type=int)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("exp_id", type=str)

    parser.add_argument("--emb_dir", type=str)
    parser.add_argument("--dropout_size", type=float, default=0.5)  # keep_prob 1.14    # rate > 2.x
    parser.add_argument("--ef_mode", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)  # batch 64 1e-3, batch 32 1e-4
    parser.add_argument("--l2_reg", type=float, default=1e-4)  # batch 8 1e-4 epoch4,  batch 4 1e-5
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--no_standardize", action='store_true')
    parser.add_argument("--label_mode", type=str, choices=["fine", "coarse"],
                        default='fine')
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    # save args to disk
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    out_dir = os.path.join(args.output_dir, args.exp_id, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    kwarg_file = os.path.join(out_dir, "hyper_params.json")
    with open(kwarg_file, 'w') as f:
        json.dump(vars(args), f, indent=2)

    train(args.annotation_path,
          args.num_classes,
          args.emb_dir,
          args.output_dir,
          args.exp_id,
          batch_size=args.batch_size,
          num_epochs=args.num_epochs,
          patience=args.patience,
          learning_rate=args.learning_rate,
          dropout_size=args.dropout_size,
          ef_mode=args.ef_mode,
          l2_reg=args.l2_reg,
          standardize=(not args.no_standardize),
          timestamp=timestamp,
          random_state=args.random_state)
