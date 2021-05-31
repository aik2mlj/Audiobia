import random
import numpy as np
import librosa
import tensorflow.keras.utils


class AudioUtil:
    @staticmethod
    def time_shift(sig, shift_limit):
        sig_len = sig.shape[0]
        shift_amount = int(random.random() * shift_limit * sig_len)
        return sig.roll(shift_amount)

    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps, num = spec.shape
        mask_value = spec[0, :, :, :].mean()
        aug_spec = np.array(spec)

        freq_mask_param = int(max_mask_pct * n_mels)
        # print(":", freq_mask_param)
        mask_row = np.ones((1, 1, n_steps, num)) * mask_value
        # print(mask_row)
        for _ in range(n_freq_masks):
            place = int(random.random() * (n_mels - freq_mask_param))
            aug_spec[:, place: place + freq_mask_param, :, :] = mask_row

        time_mask_param = int(max_mask_pct * n_steps)
        mask_column = np.ones((1, n_mels, 1, num)) * mask_value
        for _ in range(n_time_masks):
            # aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
            place = int(random.random() * (n_steps - time_mask_param))
            aug_spec[0, :, place: place + time_mask_param, :] = mask_column

        return aug_spec


class SoundDS(tensorflow.keras.utils.Sequence):
    def __init__(self, data_list, label_list, num_classes, data_path, batch_size=32, dim=(32, 32, 32), shuffle=True):
        self.data_list = data_list
        self.label_list = label_list
        self.num_classes = num_classes
        self.data_path = data_path
        self.sr = 44100
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """number of items in dataset"""
        return int(np.floor(len(self.data_list) / self.batch_size))

    def __getitem__(self, idx):
        """Generate one batch of data"""
        indices = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        X, y = self.__data_generation(indices)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indices = np.arange(len(self.data_list))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, data_list):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        y = np.empty(self.batch_size, dtype=int)
        X = []

        for i, ID in enumerate(data_list):
            x_emb = np.load(self.data_path + self.data_list[ID][:-4] + '.npz')['embedding']
            X.append(x_emb)
            y[i] = self.label_list[i]

        return np.array(X), y
