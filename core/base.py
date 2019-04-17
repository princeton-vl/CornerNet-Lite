import json

from .nnet.py_factory import NetworkFactory

class Base(object):
    def __init__(self, db, nnet, func, model=None):
        super(Base, self).__init__()

        self._db   = db
        self._nnet = nnet
        self._func = func

        if model is not None:
            self._nnet.load_pretrained_params(model)

        self._nnet.cuda()
        self._nnet.eval_mode()

    def _inference(self, image, *args, **kwargs):
        return self._func(self._db, self._nnet, image.copy(), *args, **kwargs)

    def __call__(self, image, *args, **kwargs):
        categories = self._db.configs["categories"]
        bboxes     = self._inference(image, *args, **kwargs)
        return {self._db.cls2name(j): bboxes[j] for j in range(1, categories + 1)}

def load_cfg(cfg_file):
    with open(cfg_file, "r") as f:
        cfg = json.load(f)

    cfg_sys = cfg["system"]
    cfg_db  = cfg["db"]
    return cfg_sys, cfg_db

def load_nnet(cfg_sys, model):
    return NetworkFactory(cfg_sys, model)
