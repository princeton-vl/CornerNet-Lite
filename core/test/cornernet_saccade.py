import os
import cv2
import math
import json
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm

from ..utils import Timer
from ..vis_utils import draw_bboxes
from ..external.nms import soft_nms

def crop_image_gpu(image, center, size, out_image):
    cty, ctx            = center
    height, width       = size
    o_height, o_width   = out_image.shape[1:3]
    im_height, im_width = image.shape[1:3]

    scale  = o_height / max(height, width)
    x0, x1 = max(0, ctx - width // 2), min(ctx + width // 2, im_width)
    y0, y1 = max(0, cty - height // 2), min(cty + height // 2, im_height)

    left, right = ctx - x0, x1 - ctx
    top, bottom = cty - y0, y1 - cty

    cropped_cty, cropped_ctx = o_height // 2, o_width // 2
    out_y0, out_y1 = cropped_cty - int(top * scale), cropped_cty + int(bottom * scale)
    out_x0, out_x1 = cropped_ctx - int(left * scale), cropped_ctx + int(right * scale)

    new_height = out_y1 - out_y0
    new_width  = out_x1 - out_x0
    image      = image[:, y0:y1, x0:x1].unsqueeze(0)
    out_image[:, out_y0:out_y1, out_x0:out_x1] = nn.functional.interpolate(
        image, size=[new_height, new_width], mode='bilinear'
    )[0]

    return np.array([cty - height // 2, ctx - width  // 2])

def remap_dets_(detections, scales, offsets):
    xs, ys = detections[..., 0:4:2], detections[..., 1:4:2]
 
    xs /= scales.reshape(-1, 1, 1)
    ys /= scales.reshape(-1, 1, 1)
    xs += offsets[:, 1][:, None, None]
    ys += offsets[:, 0][:, None, None]

def att_nms(atts, ks):
    pads  = [(k - 1) // 2 for k in ks]
    pools = [nn.functional.max_pool2d(att, (k, k), stride=1, padding=pad) for k, att, pad in zip(ks, atts, pads)]
    keeps = [(att == pool).float() for att, pool in zip(atts, pools)]
    atts  = [att * keep for att, keep in zip(atts, keeps)]
    return atts

def batch_decode(db, nnet, images, no_att=False):
    K            = db.configs["top_k"]
    ae_threshold = db.configs["ae_threshold"]
    kernel       = db.configs["nms_kernel"]
    num_dets     = db.configs["num_dets"]

    att_nms_ks = db.configs["att_nms_ks"]
    att_ranges = db.configs["att_ranges"]

    num_images = images.shape[0]
    detections = []
    attentions = [[] for _ in range(len(att_ranges))]   

    batch_size = 32
    for b_ind in range(math.ceil(num_images / batch_size)):
        b_start = b_ind * batch_size
        b_end   = min(num_images, (b_ind + 1) * batch_size)

        b_images  = images[b_start:b_end]
        b_outputs = nnet.test(
                [b_images], ae_threshold=ae_threshold, K=K, kernel=kernel,
                test=True, num_dets=num_dets, no_border=True, no_att=no_att
        )
        if no_att:
            b_detections = b_outputs
        else:
            b_detections = b_outputs[0]
            b_attentions = b_outputs[1]
            b_attentions = att_nms(b_attentions, att_nms_ks)
            b_attentions = [b_attention.data.cpu().numpy() for b_attention in b_attentions]

        b_detections = b_detections.data.cpu().numpy()

        detections.append(b_detections)
        if not no_att:
            for attention, b_attention in zip(attentions, b_attentions):
                attention.append(b_attention)

    if not no_att:
        attentions = [np.concatenate(atts, axis=0) for atts in attentions] if detections else None
    detections = np.concatenate(detections, axis=0) if detections else np.zeros((0, num_dets, 8))
    return detections, attentions

def decode_atts(db, atts, att_scales, scales, offsets, height, width, thresh, ignore_same=False):
    att_ranges = db.configs["att_ranges"]
    att_ratios = db.configs["att_ratios"]
    input_size = db.configs["input_size"]

    next_ys, next_xs, next_scales, next_scores = [], [], [], []

    num_atts = atts[0].shape[0]
    for aind in range(num_atts):
        for att, att_range, att_ratio, att_scale in zip(atts, att_ranges, att_ratios, att_scales):
            ys, xs = np.where(att[aind, 0] > thresh)
            scores = att[aind, 0, ys, xs]

            ys = ys * att_ratio / scales[aind] + offsets[aind, 0]
            xs = xs * att_ratio / scales[aind] + offsets[aind, 1]

            keep = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
            ys, xs, scores = ys[keep], xs[keep], scores[keep]

            next_scale = att_scale * scales[aind]
            if (ignore_same and att_scale <= 1) or scales[aind] > 2 or next_scale > 4:
                continue

            next_scales += [next_scale] * len(xs)
            next_scores += scores.tolist()
            next_ys     += ys.tolist()
            next_xs     += xs.tolist()
    next_ys = np.array(next_ys)
    next_xs = np.array(next_xs)
    next_scales = np.array(next_scales)
    next_scores = np.array(next_scores)
    return np.stack((next_ys, next_xs, next_scales, next_scores), axis=1)

def get_ref_locs(dets):
    keep = dets[:, 4] > 0.5
    dets = dets[keep]

    ref_xs = (dets[:, 0] + dets[:, 2]) / 2
    ref_ys = (dets[:, 1] + dets[:, 3]) / 2

    ref_maxhws = np.maximum(dets[:, 2] - dets[:, 0], dets[:, 3] - dets[:, 1])
    ref_scales = np.zeros_like(ref_maxhws)
    ref_scores = dets[:, 4]

    large_inds  = ref_maxhws > 96
    medium_inds = (ref_maxhws > 32) & (ref_maxhws <= 96)
    small_inds  = ref_maxhws <= 32

    ref_scales[large_inds]  = 192 / ref_maxhws[large_inds]
    ref_scales[medium_inds] =  64 / ref_maxhws[medium_inds]
    ref_scales[small_inds]  =  24 / ref_maxhws[small_inds]

    new_locations = np.stack((ref_ys, ref_xs, ref_scales, ref_scores), axis=1)
    new_locations[:, 3] = 1
    return new_locations

def get_locs(db, nnet, image, im_mean, im_std, att_scales, thresh, sizes, ref_dets=True):
    att_ranges = db.configs["att_ranges"]
    att_ratios = db.configs["att_ratios"]
    input_size = db.configs["input_size"]

    height, width = image.shape[1:3]

    locations = []
    for size in sizes:
        scale    = size / max(height, width)
        location = [height // 2, width // 2, scale]
        locations.append(location)

    locations       = np.array(locations, dtype=np.float32)
    images, offsets = prepare_images(db, image, locations, flipped=False)

    images -= im_mean
    images /= im_std

    dets, atts = batch_decode(db, nnet, images)

    scales = locations[:, 2]
    next_locations = decode_atts(db, atts, att_scales, scales, offsets, height, width, thresh)

    rescale_dets_(db, dets)
    remap_dets_(dets, scales, offsets)

    dets = dets.reshape(-1, 8)
    keep = dets[:, 4] > 0.3
    dets = dets[keep]

    if ref_dets:
        ref_locations  = get_ref_locs(dets)
        next_locations = np.concatenate((next_locations, ref_locations), axis=0)
        next_locations = location_nms(next_locations, thresh=16)
    return dets, next_locations, atts

def location_nms(locations, thresh=15):
    next_locations = []
    sorted_inds    = np.argsort(locations[:, -1])[::-1]

    locations = locations[sorted_inds]
    ys = locations[:, 0]
    xs = locations[:, 1]
    scales = locations[:, 2]

    dist_ys = np.absolute(ys.reshape(-1, 1) - ys.reshape(1, -1))
    dist_xs = np.absolute(xs.reshape(-1, 1) - xs.reshape(1, -1))
    dists   = np.minimum(dist_ys, dist_xs)
    ratios  = scales.reshape(-1, 1) / scales.reshape(1, -1)
    while dists.shape[0] > 0:
        next_locations.append(locations[0])

        scale = scales[0]
        dist  = dists[0]
        ratio = ratios[0]

        keep  = (dist > (thresh / scale)) | (ratio > 1.2) | (ratio < 0.8)

        locations = locations[keep]

        scales = scales[keep]
        dists  = dists[keep, :]
        dists  = dists[:, keep]
        ratios = ratios[keep, :]
        ratios = ratios[:, keep]
    return np.stack(next_locations) if next_locations else np.zeros((0, 4))

def prepare_images(db, image, locs, flipped=True):
    input_size  = db.configs["input_size"]
    num_patches = locs.shape[0]

    images  = torch.cuda.FloatTensor(num_patches, 3, input_size[0], input_size[1]).fill_(0)
    offsets = np.zeros((num_patches, 2), dtype=np.float32)
    for ind, (y, x, scale) in enumerate(locs[:, :3]):
        crop_height  = int(input_size[0] / scale)
        crop_width   = int(input_size[1] / scale)
        offsets[ind] = crop_image_gpu(image, [int(y), int(x)], [crop_height, crop_width], images[ind])
    return images, offsets

def rescale_dets_(db, dets):
    input_size  = db.configs["input_size"]
    output_size = db.configs["output_sizes"][0]

    ratios = [o / i for o, i in zip(output_size, input_size)]
    dets[..., 0:4:2] /= ratios[1]
    dets[..., 1:4:2] /= ratios[0]

def cornernet_saccade(db, nnet, result_dir, debug=False, decode_func=batch_decode):
    debug_dir = os.path.join(result_dir, "debug")
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    if db.split != "trainval2014":
        db_inds = db.db_inds[:500] if debug else db.db_inds
    else:
        db_inds = db.db_inds[:100] if debug else db.db_inds[:5000]

    num_images = db_inds.size
    categories = db.configs["categories"]

    timer = Timer()
    top_bboxes = {}
    for k_ind in tqdm(range(0, num_images), ncols=80, desc="locating kps"):
        db_ind = db_inds[k_ind]

        image_id   = db.image_ids(db_ind)
        image_path = db.image_path(db_ind)
        image      = cv2.imread(image_path)

        timer.tic()
        top_bboxes[image_id] = cornernet_saccade_inference(db, nnet, image)
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

def cornernet_saccade_inference(db, nnet, image, decode_func=batch_decode): 
    init_sizes  = db.configs["init_sizes"]
    ref_dets    = db.configs["ref_dets"]

    att_thresholds = db.configs["att_thresholds"]
    att_scales     = db.configs["att_scales"]
    att_max_crops  = db.configs["att_max_crops"]

    categories    = db.configs["categories"]
    nms_threshold = db.configs["nms_threshold"]
    max_per_image = db.configs["max_per_image"]
    nms_algorithm = {
        "nms": 0,
        "linear_soft_nms": 1, 
        "exp_soft_nms": 2
    }[db.configs["nms_algorithm"]]

    num_iterations = len(att_thresholds)

    im_mean = torch.cuda.FloatTensor(db.mean).reshape(1, 3, 1, 1)
    im_std  = torch.cuda.FloatTensor(db.std).reshape(1, 3, 1, 1)

    detections    = []
    height, width = image.shape[0:2]

    image = image / 255.
    image = image.transpose((2, 0, 1)).copy()
    image = torch.from_numpy(image).cuda(non_blocking=True)

    dets, locations, atts = get_locs(
        db, nnet, image, im_mean, im_std, 
        att_scales[0], att_thresholds[0], 
        init_sizes, ref_dets=ref_dets
    )

    detections  = [dets]
    num_patches = locations.shape[0]

    num_crops = 0
    for ind in range(1, num_iterations + 1):
        if num_patches == 0:
            break

        if num_crops + num_patches > att_max_crops:
            max_crops = min(att_max_crops - num_crops, num_patches)
            locations = locations[:max_crops]

        num_patches = locations.shape[0]
        num_crops  += locations.shape[0]
        no_att = (ind == num_iterations)

        images, offsets = prepare_images(db, image, locations, flipped=False)
        images -= im_mean
        images /= im_std

        dets, atts = decode_func(db, nnet, images, no_att=no_att)
        dets = dets.reshape(num_patches, -1, 8)

        rescale_dets_(db, dets)
        remap_dets_(dets, locations[:, 2], offsets)

        dets  = dets.reshape(-1, 8)
        keeps = (dets[:, 4] > -1)
        dets  = dets[keeps]

        detections.append(dets)

        if num_crops == att_max_crops:
            break

        if ind < num_iterations:
            att_threshold  = att_thresholds[ind]
            att_scale      = att_scales[ind]

            next_locations = decode_atts(
                db, atts, att_scale, locations[:, 2], offsets, height, width, att_threshold, ignore_same=True
            )

            if ref_dets:
                ref_locations  = get_ref_locs(dets)
                next_locations = np.concatenate((next_locations, ref_locations), axis=0)
                next_locations = location_nms(next_locations, thresh=16)

            locations   = next_locations
            num_patches = locations.shape[0]

    detections = np.concatenate(detections, axis=0)
    classes    = detections[..., -1]

    top_bboxes = {}
    for j in range(categories):
        keep_inds = (classes == j)
        top_bboxes[j + 1] = detections[keep_inds][:, 0:7].astype(np.float32)
        keep_inds = soft_nms(top_bboxes[j + 1], Nt=nms_threshold, method=nms_algorithm, sigma=0.7)
        top_bboxes[j + 1] = top_bboxes[j + 1][keep_inds, 0:5]

    scores = np.hstack([top_bboxes[j][:, -1] for j in range(1, categories + 1)])
    if len(scores) > max_per_image:
        kth    = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, categories + 1):
            keep_inds = (top_bboxes[j][:, -1] >= thresh)
            top_bboxes[j] = top_bboxes[j][keep_inds]
    return top_bboxes
