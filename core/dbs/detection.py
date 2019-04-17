import numpy as np

from .base import BASE

class DETECTION(BASE):
    def __init__(self, db_config):
        super(DETECTION, self).__init__()

        # Configs for training
        self._configs["categories"]      = 80
        self._configs["rand_scales"]     = [1]
        self._configs["rand_scale_min"]  = 0.8
        self._configs["rand_scale_max"]  = 1.4
        self._configs["rand_scale_step"] = 0.2

        # Configs for both training and testing
        self._configs["input_size"]      = [383, 383]
        self._configs["output_sizes"]    = [[96, 96], [48, 48], [24, 24], [12, 12]]

        self._configs["score_threshold"] = 0.05
        self._configs["nms_threshold"]   = 0.7
        self._configs["max_per_set"]     = 40
        self._configs["max_per_image"]   = 100
        self._configs["top_k"]           = 20
        self._configs["ae_threshold"]    = 1
        self._configs["nms_kernel"]      = 3
        self._configs["num_dets"]        = 1000

        self._configs["nms_algorithm"]   = "exp_soft_nms"
        self._configs["weight_exp"]      = 8
        self._configs["merge_bbox"]      = False

        self._configs["data_aug"]        = True
        self._configs["lighting"]        = True

        self._configs["border"]          = 64
        self._configs["gaussian_bump"]   = False
        self._configs["gaussian_iou"]    = 0.7
        self._configs["gaussian_radius"] = -1
        self._configs["rand_crop"]       = False
        self._configs["rand_color"]      = False
        self._configs["rand_center"]     = True

        self._configs["init_sizes"]      = [192, 255]
        self._configs["view_sizes"]      = []

        self._configs["min_scale"]       = 16
        self._configs["max_scale"]       = 32

        self._configs["att_sizes"]       = [[16, 16], [32, 32], [64, 64]]
        self._configs["att_ranges"]      = [[96, 256], [32, 96], [0, 32]]
        self._configs["att_ratios"]      = [16, 8, 4]
        self._configs["att_scales"]      = [1, 1.5, 2]
        self._configs["att_thresholds"]  = [0.3, 0.3, 0.3, 0.3]
        self._configs["att_nms_ks"]      = [3, 3, 3]
        self._configs["att_max_crops"]   = 8
        self._configs["ref_dets"]        = True

        # Configs for testing
        self._configs["test_scales"]     = [1]
        self._configs["test_flipped"]    = True

        self.update_config(db_config)

        if self._configs["rand_scales"] is None:
            self._configs["rand_scales"] = np.arange(
                self._configs["rand_scale_min"], 
                self._configs["rand_scale_max"],
                self._configs["rand_scale_step"]
            )
