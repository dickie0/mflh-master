import numpy as np
import math
from utils.metrics.np_v import metric


class Dataset(object):
    def __init__(self, dataset, output_dim):
        self._dataset = dataset
        self.n_samples = dataset.n_samples
        self._train = dataset.train
        self._output = np.zeros((self.n_samples, output_dim), dtype=np.float32)
        self._triplets = np.array([])
        self._trip_index_in_epoch = 0
        self._index_in_epoch = 0
        self._epochs_complete = 0
        self._perm = np.arange(self.n_samples)
        np.random.shuffle(self._perm)
        return

    def iter_triplets(self, n_part=10, dist='euclidean2', select_strategy='all'):
        n_samples = self.n_samples
        np.random.shuffle(self._perm)
        n_samples_per_part = int(math.ceil(n_samples / n_part))
        embedding = self._output[self._perm[:n_samples]]
        labels = self._dataset.get_labels()[self._perm[:n_samples]]
        triplets = []
        for i in range(n_part):
            start = n_samples_per_part * i
            end = min(n_samples_per_part * (i+1), n_samples)
            distance = metric(embedding[start:end], pair=True, dist=dist)
            for idx_anchor in range(0, end - start):
                label_anchor = np.copy(labels[idx_anchor+start, :])
                label_anchor[label_anchor==0] = -1
                all_pos = np.where(np.any(labels[start:end] == label_anchor, axis=1))[0]
                all_neg = np.array(list(set(range(end-start)) - set(all_pos)))
                if select_strategy == 'hard':
                    idx_pos = all_pos[np.argmax(distance[idx_anchor, all_pos])]
                    if idx_pos == idx_anchor:
                        continue
                    idx_neg = all_neg[np.argmin(distance[idx_anchor, all_neg])]
                    triplets.append((idx_anchor + start, idx_pos + start, idx_neg + start))
                    continue
                for idx_pos in all_pos:
                    if idx_pos == idx_anchor:
                        continue
                    if select_strategy == 'all':
                        selected_neg = all_neg
                    if selected_neg.shape[0] > 0:
                        idx_neg = np.random.choice(selected_neg)
                        triplets.append((idx_anchor + start, idx_pos + start, idx_neg + start))
        self._triplets = np.array(triplets)
        np.random.shuffle(self._triplets)
        anchor = labels[self._triplets[:, 0]]
        mapper = lambda anchor, other: np.any(anchor * (anchor == other), -1)
        assert(np.all(mapper(anchor, labels[self._triplets[:, 1]])))
        assert(np.all(np.invert(anchor, labels[self._triplets[:, 2]])))
        return

    def new_triplet(self, batch_size):
        start = self._trip_index_in_epoch
        self._trip_index_in_epoch += batch_size
        if self._trip_index_in_epoch > self.triplets.shape[0]:
            start = 0
            self._trip_index_in_epoch = batch_size
        end = self._trip_index_in_epoch
        arr = self.triplets[start:end]
        idx = self._perm[np.concatenate([arr[:, 0], arr[:, 1], arr[:, 2]], axis=0)]
        data, label, _ = self._dataset.data(idx)
        return data, label

    def new_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.n_samples:
            if self._train:
                self._epochs_complete += 1
                start = 0
                self._index_in_epoch = batch_size
            else:
                start = self.n_samples - batch_size
                self._index_in_epoch = self.n_samples
        end = self._index_in_epoch
        data, label, name = self._dataset.data(self._perm[start:end])
        return data, label, name

    def batch_out(self, bs, output):
        start = self._index_in_epoch - bs
        end = self._index_in_epoch
        self._output[self._perm[start:end], :] = output
        return

    def batch_triplet(self, bs, triplet_output):
        anchor, pos, neg = np.split(triplet_output, 3, axis=0)
        start = self._trip_index_in_epoch - bs
        end = self._trip_index_in_epoch
        idx = self._perm[self._triplets[start:end, :]]
        self._output[idx[:, 0]] = anchor
        self._output[idx[:, 1]] = pos
        self._output[idx[:, 2]] = neg
        return

    @property
    def output(self):
        return self._output

    @property
    def triplets(self):
        return self._triplets

    @property
    def label(self):
        return self._dataset.get_labels()

    def finish_epoch(self):
        self._index_in_epoch = 0