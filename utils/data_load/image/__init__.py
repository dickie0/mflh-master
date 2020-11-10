import os
import cv2
import numpy as np


class Dataset(object):
    def __init__(self, model, data_root, path,config, train=True):
        self.lines = open(path, 'r').readlines()
        self.data_root = data_root
        self.n_samples = len(self.lines)
        self.train = train
        assert model == 'img'
        self.model = 'img'
        self._img = [0] * self.n_samples
        self._label = [0] * self.n_samples
        self._name = [0] * self.n_samples
        self._load = [0] * self.n_samples
        self._load_num = 0
        self._status = 0
        self.data = self.img_data
        self.all_data = self.img_all_data

    def get_img(self, i):
        path = os.path.join(self.data_root, self.lines[i].strip().split()[0])
        x = cv2.imread(path)
        return x

    def get_label(self, i):
        return [int(j) for j in self.lines[i].strip().split()[1:]]

    def get_name(self, i):
        return self.lines[i].strip().split()[0]

    def img_data(self, indexes):
        if self._status:
            ret_img = self._img[indexes, :]
            ret_lab = self._label[indexes, :]
            data = np.transpose(ret_img, (0, 3, 1, 2))
            data = data[:, ::-1, :, :]
            if self.train is True:
                ret_img = np.reshape(data, (192, -1))
            else:
                ret_img = np.reshape(data, (16, -1))
                ret_lab = np.reshape(ret_lab, (16, -1))
            return ret_img, ret_lab, self._name[indexes]
        else:
            ret_img = []
            ret_label = []
            ret_name = []
            for i in indexes:
                try:
                    if self.train:
                        if not self._load[i]:
                            self._img[i] = self.get_img(i)
                            self._label[i] = self.get_label(i)
                            self._name[i] = self.get_name(i)
                            self._load[i] = 1
                            self._load_num += 1
                        ret_img.append(self._img[i])
                        ret_label.append(self._label[i])
                        ret_name.append(self._name[i])
                    else:
                        self._label[i] = self.get_label(i)
                        self._name[i] = self.get_name(i)
                        ret_img.append(self.get_img(i))
                        ret_label.append(self._label[i])
                        ret_name.append(self._name[i])
                except Exception as e:
                    print('cannot open {}, exception: {}'.format(self.lines[i].strip(), e))

            if self._load_num == self.n_samples:
                self._status = 1
                self._img = np.asarray(self._img)
                self._label = np.asarray(self._label)
                self._name = np.asarray(self._name)

            data = np.transpose(ret_img, (0, 3, 1, 2))
            data = data[:, ::-1, :, :]
            if self.train is True:
                ret_img = np.reshape(data, (192, -1))
            else:
                ret_img = np.reshape(data, (16, -1))
                ret_label = np.reshape(ret_label, (16, -1))
            return np.asarray(ret_img), np.asarray(ret_label), np.asarray(ret_name)

    def img_all_data(self):
        if self._status:
            return self._img, self._label

    def get_labels(self):
        for i in range(self.n_samples):
            if self._label[i] is not list:
                self._label[i] = [int(j)
                                  for j in self.lines[i].strip().split()[1:]]
        return np.asarray(self._label)


def import_train(data_root, img_tr, config):
    return Dataset('img', data_root, img_tr, config, train=True)


def import_test(data_root, img_te, img_db, config):
    return (Dataset('img', data_root, img_te, config, train=False),
            Dataset('img', data_root, img_db, config, train=False))