import numpy as np
from utils.util import sign
from utils.metrics.np_v import metric


class MAP:
    def __init__(self,  num):
        self.num = num

    def map_fr(self, database, query, nums=None, dist='inner_product'):
        if nums is None:
            nums = self.num
        q_output = query.output
        db_output = database.output
        return cal_MAP(q_output, query.label, db_output, database.label, nums, dist)

    def map_sign(self, database, query, nums=None, dist='inner_product'):
        if nums is None:
            nums = self.num
        q_output = sign(query.output)
        db_output = sign(database.output)
        return cal_MAP(q_output, query.label, db_output, database.label, nums, dist)


def cal_MAP(q_output, q_labels, db_output, db_labels, nums, dist):
    distance = metric(q_output, db_output, dist=dist, pair=True)
    unsorted_ids = np.argpartition(distance, nums - 1)[:, :nums]
    APx = []
    for i in range(distance.shape[0]):
        label = q_labels[i, :]
        label[label == 0] = -1
        idx = unsorted_ids[i, :]
        idx = idx[np.argsort(distance[i, :][idx])]
        imatch = np.sum(np.equal(db_labels[idx[0: nums], :], label), 1) > 0
        rel = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, nums + 1, 1)
        if rel != 0:
            APx.append(np.sum(Px * imatch) / rel)
    return np.mean(np.array(APx))
