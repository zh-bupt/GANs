import pickle
import numpy as np
import os


class Cifar10DataReader():
    def __init__(self, path, batch_size=128, onehot=True):
        self.dataset_path = path
        self.onehot = onehot
        self.batch_size = batch_size
        self.read_next = True
        self.batch_index = 0
        self.unpickle()

    def unpickle(self):
        x_train = None
        for i in range(1, 6):
            with open(os.path.join(self.dataset_path, 'data_batch_' + str(i)), 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                if i == 1:
                    x_train = dict[b'data']
                else:
                    x_train = np.concatenate((x_train, dict[b'data']), axis=0)
        x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32')
        self.data = x_train

    def next_batch(self):
        if self.read_next:
            np.random.shuffle(self.data)

        if self.batch_index < len(self.data) // self.batch_size:
            batch = self.data[self.batch_index * self.batch_size:(self.batch_index + 1) * self.batch_size]
            self.batch_index += 1
            # batch_data, batch_label = self._decode(batch, self.onehot)
        else:
            self.batch_index = 0
            self.read_next = True
            return self.next_batch()

        return batch
        # return batch_data, batch_label

    def _decode(self, batch, onehot):
        data = list()
        label = list()
        if onehot:
            for d, l in batch:
                data.append(np.reshape(np.reshape(d, [3, 1024]).T, [32, 32, 3]))
                hot = np.zeros(10)
                hot[int(l)] = 1
                label.append(hot)
        else:
            for d, l in batch:
                data.append(np.reshape(np.reshape(d, [3, 1024]).T, [32, 32, 3]))
                label.append(int(l))
        return data, label

