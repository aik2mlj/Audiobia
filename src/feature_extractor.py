import argparse
import os
from multiprocessing import Pool

import librosa
import numpy as np

resampling = 44100
n_fft = 1024
win_length = 1024
hop_length = 512

n_mels = 128

PROCESSOR = 16
second = 5

parser = argparse.ArgumentParser()

parser.add_argument("input_path")
parser.add_argument("output_path")
args = parser.parse_args()

output_path = args.output_path
input_path = args.input_path


def feature_extractor(name, srate=resampling):
    name = name.split('?')
    filename = name[0]
    second = int(name[3])
    out_path = name[4]

    print("\tProgress: ", str((int(int(name[1]) / int(name[2]) * 100))), "%")
    try:
        sig, sr = librosa.load(filename, sr=srate, mono=True)
    except:
        raise IOError("Cannot read this audio file")

    assert sr == resampling

    try:
        sp_harmonic, sp_percussive = librosa.effects.hpss(sig)
        sp_y = librosa.util.normalize(sig, norm=np.inf, axis=None)
        sp_harmonic = librosa.util.normalize(sp_harmonic, norm=np.inf, axis=None)
        sp_percussive = librosa.util.normalize(sp_percussive, norm=np.inf, axis=None)

        inpt = librosa.power_to_db(
            librosa.feature.melspectrogram(y=sp_y, sr=srate, n_fft=n_fft, hop_length=hop_length,
                                           win_length=win_length, n_mels=n_mels))
        inpt2 = librosa.power_to_db(
            librosa.feature.melspectrogram(y=sp_harmonic, sr=srate, n_fft=n_fft, hop_length=hop_length,
                                           win_length=win_length, n_mels=n_mels))
        inpt3 = librosa.power_to_db(
            librosa.feature.melspectrogram(y=sp_percussive, sr=srate, n_fft=n_fft, hop_length=hop_length,
                                           win_length=win_length, n_mels=n_mels))

        # stack_inpt = np.stack(([inpt], [inpt2], [inpt3]), axis=3)
        stack_inpt = np.stack(([inpt], [inpt], [inpt]), axis=3)

        stack_inpt = librosa.util.normalize(stack_inpt, norm=np.inf, axis=None)

        # feature = np.reshape(stack_inpt, (stack_inpt.shape[1], stack_inpt.shape[2], stack_inpt.shape[3]))
        feature = stack_inpt

    except Exception as e:
        print(e)

    filename = filename.split('\\')[-1]
    print(out_path + filename[:-4] + '.npz')
    feature = np.asarray(feature)
    print(feature.shape)
    np.savez(out_path + filename[:-4], embedding=feature)
    return feature


if __name__ == '__main__':
    train_folder = sorted(os.listdir(input_path))
    print(train_folder[:5])
    print("Data Feature Extraction")

    p = Pool(processes=PROCESSOR)
    data = p.map(feature_extractor, [
        input_path + '\\' + train_folder[j] + '?' + str(j) + '?' + str(len(train_folder)) + '?' + str(
            second) + '?' + str(output_path) for j in range(len(train_folder))], chunksize=1)
    p.close()
