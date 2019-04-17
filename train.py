#!/usr/bin/env python
import os
import json
import torch
import numpy as np
import queue
import pprint
import random 
import argparse
import importlib
import threading
import traceback
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from torch.multiprocessing import Process, Queue, Pool

from core.dbs import datasets
from core.utils import stdout_to_tqdm
from core.config import SystemConfig
from core.sample import data_sampling_func
from core.nnet.py_factory import NetworkFactory

torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument("--iter", dest="start_iter",
                        help="train at iteration i",
                        default=0, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--initialize", action="store_true")

    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--world-size", default=-1, type=int,
                        help="number of nodes of distributed training")
    parser.add_argument("--rank", default=0, type=int,
                        help="node rank for distributed training")
    parser.add_argument("--dist-url", default=None, type=str,
                        help="url used to set up distributed training")
    parser.add_argument("--dist-backend", default="nccl", type=str)

    args = parser.parse_args()
    return args

def prefetch_data(system_config, db, queue, sample_data, data_aug):
    ind = 0
    print("start prefetching data...")
    np.random.seed(os.getpid())
    while True:
        try:
            data, ind = sample_data(system_config, db, ind, data_aug=data_aug)
            queue.put(data)
        except Exception as e:
            traceback.print_exc()
            raise e

def _pin_memory(ts):
    if type(ts) is list:
        return [t.pin_memory() for t in ts]
    return ts.pin_memory()

def pin_memory(data_queue, pinned_data_queue, sema):
    while True:
        data = data_queue.get()

        data["xs"] = [_pin_memory(x) for x in data["xs"]]
        data["ys"] = [_pin_memory(y) for y in data["ys"]]

        pinned_data_queue.put(data)

        if sema.acquire(blocking=False):
            return

def init_parallel_jobs(system_config, dbs, queue, fn, data_aug):
    tasks = [Process(target=prefetch_data, args=(system_config, db, queue, fn, data_aug)) for db in dbs]
    for task in tasks:
        task.daemon = True
        task.start()
    return tasks

def terminate_tasks(tasks):
    for task in tasks:
        task.terminate()

def train(training_dbs, validation_db, system_config, model, args):
    # reading arguments from command
    start_iter  = args.start_iter
    distributed = args.distributed
    world_size  = args.world_size
    initialize  = args.initialize
    gpu         = args.gpu
    rank        = args.rank

    # reading arguments from json file
    batch_size       = system_config.batch_size
    learning_rate    = system_config.learning_rate
    max_iteration    = system_config.max_iter
    pretrained_model = system_config.pretrain
    stepsize         = system_config.stepsize
    snapshot         = system_config.snapshot
    val_iter         = system_config.val_iter
    display          = system_config.display
    decay_rate       = system_config.decay_rate
    stepsize         = system_config.stepsize

    print("Process {}: building model...".format(rank))
    nnet = NetworkFactory(system_config, model, distributed=distributed, gpu=gpu)
    if initialize:
        nnet.save_params(0)
        exit(0)

    # queues storing data for training
    training_queue   = Queue(system_config.prefetch_size)
    validation_queue = Queue(5)

    # queues storing pinned data for training
    pinned_training_queue   = queue.Queue(system_config.prefetch_size)
    pinned_validation_queue = queue.Queue(5)

    # allocating resources for parallel reading
    training_tasks = init_parallel_jobs(system_config, training_dbs, training_queue, data_sampling_func, True)
    if val_iter:
        validation_tasks = init_parallel_jobs(system_config, [validation_db], validation_queue, data_sampling_func, False)

    training_pin_semaphore   = threading.Semaphore()
    validation_pin_semaphore = threading.Semaphore()
    training_pin_semaphore.acquire()
    validation_pin_semaphore.acquire()

    training_pin_args   = (training_queue, pinned_training_queue, training_pin_semaphore)
    training_pin_thread = threading.Thread(target=pin_memory, args=training_pin_args)
    training_pin_thread.daemon = True
    training_pin_thread.start()

    validation_pin_args   = (validation_queue, pinned_validation_queue, validation_pin_semaphore)
    validation_pin_thread = threading.Thread(target=pin_memory, args=validation_pin_args)
    validation_pin_thread.daemon = True
    validation_pin_thread.start()

    if pretrained_model is not None:
        if not os.path.exists(pretrained_model):
            raise ValueError("pretrained model does not exist")
        print("Process {}: loading from pretrained model".format(rank))
        nnet.load_pretrained_params(pretrained_model)

    if start_iter:
        nnet.load_params(start_iter)
        learning_rate /= (decay_rate ** (start_iter // stepsize))
        nnet.set_lr(learning_rate)
        print("Process {}: training starts from iteration {} with learning_rate {}".format(rank, start_iter + 1, learning_rate))
    else:
        nnet.set_lr(learning_rate)

    if rank == 0:
        print("training start...")
    nnet.cuda()
    nnet.train_mode()
    with stdout_to_tqdm() as save_stdout:
        for iteration in tqdm(range(start_iter + 1, max_iteration + 1), file=save_stdout, ncols=80):
            training = pinned_training_queue.get(block=True)
            training_loss = nnet.train(**training)

            if display and iteration % display == 0:
                print("Process {}: training loss at iteration {}: {}".format(rank, iteration, training_loss.item()))
            del training_loss

            if val_iter and validation_db.db_inds.size and iteration % val_iter == 0:
                nnet.eval_mode()
                validation = pinned_validation_queue.get(block=True)
                validation_loss = nnet.validate(**validation)
                print("Process {}: validation loss at iteration {}: {}".format(rank, iteration, validation_loss.item()))
                nnet.train_mode()

            if iteration % snapshot == 0 and rank == 0:
                nnet.save_params(iteration)

            if iteration % stepsize == 0:
                learning_rate /= decay_rate
                nnet.set_lr(learning_rate)

    # sending signal to kill the thread
    training_pin_semaphore.release()
    validation_pin_semaphore.release()

    # terminating data fetching processes
    terminate_tasks(training_tasks)
    terminate_tasks(validation_tasks)

def main(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    rank = args.rank

    cfg_file = os.path.join("./configs", args.cfg_file + ".json")
    with open(cfg_file, "r") as f:
        config = json.load(f)

    config["system"]["snapshot_name"] = args.cfg_file
    system_config = SystemConfig().update_config(config["system"])

    model_file  = "core.models.{}".format(args.cfg_file)
    model_file  = importlib.import_module(model_file)
    model       = model_file.model()

    train_split = system_config.train_split
    val_split   = system_config.val_split

    print("Process {}: loading all datasets...".format(rank))
    dataset = system_config.dataset
    workers = args.workers
    print("Process {}: using {} workers".format(rank, workers))
    training_dbs = [datasets[dataset](config["db"], split=train_split, sys_config=system_config) for _ in range(workers)]
    validation_db = datasets[dataset](config["db"], split=val_split, sys_config=system_config)

    if rank == 0:
        print("system config...")
        pprint.pprint(system_config.full)

        print("db config...")
        pprint.pprint(training_dbs[0].configs)

        print("len of db: {}".format(len(training_dbs[0].db_inds)))
        print("distributed: {}".format(args.distributed))

    train(training_dbs, validation_db, system_config, model, args)

if __name__ == "__main__":
    args = parse_args()

    distributed = args.distributed
    world_size  = args.world_size

    if distributed and world_size < 0:
        raise ValueError("world size must be greater than 0 in distributed training")

    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main(None, ngpus_per_node, args)
