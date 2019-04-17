import os
import numpy as np

class BASE(object):
    def __init__(self):
        self._split     = None
        self._db_inds   = []
        self._image_ids = []

        self._mean    = np.zeros((3, ), dtype=np.float32)
        self._std     = np.ones((3, ), dtype=np.float32)
        self._eig_val = np.ones((3, ), dtype=np.float32)
        self._eig_vec = np.zeros((3, 3), dtype=np.float32)

        self._configs = {}
        self._configs["data_aug"] = True

        self._data_rng = None

    @property
    def configs(self):
        return self._configs

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    @property
    def eig_val(self):
        return self._eig_val

    @property
    def eig_vec(self):
        return self._eig_vec

    @property
    def db_inds(self):
        return self._db_inds

    @property
    def split(self):
        return self._split

    def update_config(self, new):
        for key in new:
            if key in self._configs:
                self._configs[key] = new[key]

    def image_ids(self, ind):
        return self._image_ids[ind]

    def image_path(self, ind):
        pass

    def write_result(self, ind, all_bboxes, all_scores):
        pass

    def evaluate(self, name):
        pass

    def shuffle_inds(self, quiet=False):
        if self._data_rng is None:
            self._data_rng = np.random.RandomState(os.getpid())

        if not quiet:
            print("shuffling indices...")
        rand_perm = self._data_rng.permutation(len(self._db_inds))
        self._db_inds = self._db_inds[rand_perm]
