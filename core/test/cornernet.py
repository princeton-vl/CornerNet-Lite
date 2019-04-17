import os
import cv2
import json
import numpy as np
import torch

from tqdm import tqdm

from ..utils import Timer
from ..vis_utils import draw_bboxes
from ..sample.utils import crop_image
from ..external.nms import soft_nms, soft_nms_merge

def rescale_dets_(detections, ratios, borders, sizes):
    xs, ys = detections[..., 0:4:2], detections[..., 1:4:2]
    xs    /= ratios[:, 1][:, None, None]
    ys    /= ratios[:, 0][:, None, None]
    xs    -= borders[:, 2][:, None, None]
    ys    -= borders[:, 0][:, None, None]
    np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
    np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)

def decode(nnet, images, K, ae_threshold=0.5, kernel=3, num_dets=1000):
    detections = nnet.test([images], ae_threshold=ae_threshold, test=True, K=K, kernel=kernel, num_dets=num_dets)[0]
    return detections.data.cpu().numpy()

def cornernet(db, nnet, result_dir, debug=False, decode_func=decode):
    debug_dir = os.path.join(result_dir, "debug")
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    if db.split != "trainval2014":
        db_inds = db.db_inds[:100] if debug else db.db_inds
    else:
        db_inds = db.db_inds[:100] if debug else db.db_inds[:5000]

    num_images = db_inds.size
    categories = db.configs["categories"]

    timer = Timer()
    top_bboxes = {}
    for ind in tqdm(range(0, num_images), ncols=80, desc="locating kps"):
        db_ind = db_inds[ind]

        image_id   = db.image_ids(db_ind)
        image_path = db.image_path(db_ind)
        image      = cv2.imread(image_path)

        timer.tic()
        top_bboxes[image_id] = cornernet_inference(db, nnet, image)
        timer.toc()

        if debug:
            image_path = db.image_path(db_ind)
            image      = cv2.imread(image_path)
            bboxes     = {
                db.cls2name(j): top_bboxes[image_id][j] 
                for j in range(1, categories + 1)
            }
            image      = draw_bboxes(image, bboxes)
            debug_file = os.path.join(debug_dir, "{}.jpg".format(db_ind))
            cv2.imwrite(debug_file, image)
    print('average time: {}'.format(timer.average_time))

    result_json = os.path.join(result_dir, "results.json")
    detections  = db.convert_to_coco(top_bboxes)
    with open(result_json, "w") as f:
        json.dump(detections, f)

    cls_ids   = list(range(1, categories + 1))
    image_ids = [db.image_ids(ind) for ind in db_inds]
    db.evaluate(result_json, cls_ids, image_ids)
    return 0

def cornernet_inference(db, nnet, image, decode_func=decode):
    K             = db.configs["top_k"]
    ae_threshold  = db.configs["ae_threshold"]
    nms_kernel    = db.configs["nms_kernel"]
    num_dets      = db.configs["num_dets"]
    test_flipped  = db.configs["test_flipped"]

    input_size    = db.configs["input_size"]
    output_size   = db.configs["output_sizes"][0]
    
    scales        = db.configs["test_scales"]
    weight_exp    = db.configs["weight_exp"]
    merge_bbox    = db.configs["merge_bbox"]
    categories    = db.configs["categories"]
    nms_threshold = db.configs["nms_threshold"]
    max_per_image = db.configs["max_per_image"]
    nms_algorithm = {
        "nms": 0,
        "linear_soft_nms": 1, 
        "exp_soft_nms": 2
    }[db.configs["nms_algorithm"]]

    height, width = image.shape[0:2]

    height_scale  = (input_size[0] + 1) // output_size[0]
    width_scale   = (input_size[1] + 1) // output_size[1]

    im_mean = torch.cuda.FloatTensor(db.mean).reshape(1, 3, 1, 1)
    im_std  = torch.cuda.FloatTensor(db.std).reshape(1, 3, 1, 1)

    detections = []
    for scale in scales:
        new_height = int(height * scale)
        new_width  = int(width * scale)
        new_center = np.array([new_height // 2, new_width // 2])

        inp_height = new_height | 127
        inp_width  = new_width  | 127

        images  = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
        ratios  = np.zeros((1, 2), dtype=np.float32)
        borders = np.zeros((1, 4), dtype=np.float32)
        sizes   = np.zeros((1, 2), dtype=np.float32)

        out_height, out_width = (inp_height + 1) // height_scale, (inp_width + 1) // width_scale
        height_ratio = out_height / inp_height
        width_ratio  = out_width  / inp_width

        resized_image = cv2.resize(image, (new_width, new_height))
        resized_image, border, offset = crop_image(resized_image, new_center, [inp_height, inp_width])

        resized_image = resized_image / 255.

        images[0]  = resized_image.transpose((2, 0, 1))
        borders[0] = border
        sizes[0]   = [int(height * scale), int(width * scale)]
        ratios[0]  = [height_ratio, width_ratio]

        if test_flipped:
            images  = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images  = torch.from_numpy(images).cuda()
        images -= im_mean
        images /= im_std

        dets = decode_func(nnet, images, K, ae_threshold=ae_threshold, kernel=nms_kernel, num_dets=num_dets)
        if test_flipped:
            dets[1, :, [0, 2]] = out_width - dets[1, :, [2, 0]]
            dets = dets.reshape(1, -1, 8)

        rescale_dets_(dets, ratios, borders, sizes)
        dets[:, :, 0:4] /= scale
        detections.append(dets)

    detections = np.concatenate(detections, axis=1)

    classes    = detections[..., -1]
    classes    = classes[0]
    detections = detections[0]

    # reject detections with negative scores
    keep_inds  = (detections[:, 4] > -1)
    detections = detections[keep_inds]
    classes    = classes[keep_inds]

    top_bboxes = {}
    for j in range(categories):
        keep_inds = (classes == j)
        top_bboxes[j + 1] = detections[keep_inds][:, 0:7].astype(np.float32)
        if merge_bbox:
            soft_nms_merge(top_bboxes[j + 1], Nt=nms_threshold, method=nms_algorithm, weight_exp=weight_exp)
        else:
            soft_nms(top_bboxes[j + 1], Nt=nms_threshold, method=nms_algorithm)
        top_bboxes[j + 1] = top_bboxes[j + 1][:, 0:5]

    scores = np.hstack([top_bboxes[j][:, -1] for j in range(1, categories + 1)])
    if len(scores) > max_per_image:
        kth    = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, categories + 1):
            keep_inds     = (top_bboxes[j][:, -1] >= thresh)
            top_bboxes[j] = top_bboxes[j][keep_inds]
    return top_bboxes
