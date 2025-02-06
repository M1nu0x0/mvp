from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
import torchvision.transforms as transforms
import argparse
import os
import pprint

import _init_paths
from core.config import config
from core.config import update_config
from core.function import validate_3d
from utils.utils import create_logger
import lib.utils.misc as utils
from mmcv.runner import get_dist_info
from torch.utils.data import DistributedSampler
from models.util.misc import is_main_process, collect_results

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg', help='experiment configure file name',
                        default='configs/panoptic/best_model_config.yaml', type=str)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--model_path', default='models/model_best_5view.pth.tar', type=str,
                        help='pass model path for evaluation')

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args


def main():
    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'validate')
    device = torch.device(args.device)

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if is_main_process():
        logger.info(pprint.pformat(args))
        logger.info(pprint.pformat(config))

    print('=> Loading data ..')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        rank, world_size = get_dist_info()
        sampler_val = DistributedSampler(test_dataset, world_size, rank,
                                         shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(test_dataset)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE,
        sampler=sampler_val,
        pin_memory=True,
        num_workers=config.WORKERS)

    num_views = test_dataset.num_views
    assert config.DATASET.CAMERA_NUM == num_views, \
        'inconsistent number of views. {} != {}'.format(
            config.DATASET.CAMERA_NUM, num_views)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    print('=> Constructing models ..')
    model = eval('models.' + 'multi_view_pose_transformer' + '.get_mvp')(
        config, is_train=True)
    model.to(device)
    # with torch.no_grad():
    #     model = torch.nn.DataParallel(model, device_ids=0).cuda()
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)

    if args.model_path is not None:
        logger.info('=> load models state {}'.format(args.model_path))
        model.module.load_state_dict(torch.load(args.model_path))
    elif os.path.isfile(
            os.path.join(final_output_dir, config.TEST.MODEL_FILE)):
        test_model_file = \
            os.path.join(final_output_dir, config.TEST.MODEL_FILE)
        logger.info('=> load models state {}'.format(test_model_file))
        model.module.load_state_dict(torch.load(test_model_file))
    else:
        raise ValueError('Check the model file for testing!')

    for thr in config.DECODER.inference_conf_thr:
        model.eval()
        
        preds = []
        meta_image_files = []
        with torch.no_grad():
            end = time.time()
            for i, (inputs, meta) in enumerate(test_loader):
                assert len(inputs) == num_views
                
                output = model(views=inputs, meta=meta)
                
        preds_single, meta_image_files_single = validate_3d(
            config, model, test_loader, final_output_dir, thr,
            num_views=num_views)
        preds = collect_results(preds_single, len(test_dataset))

        if is_main_process():
            actor_pcp, avg_pcp, mpjpe, recall = test_loader.dataset.evaluate(preds)
            logger.info('actor_pcp')
            logger.info(actor_pcp)
            logger.info('avg_pcp')
            logger.info(avg_pcp)
            logger.info('mpjpe')
            logger.info(mpjpe)
            logger.info('recall')
            logger.info(recall)


def setup_distrubuted():
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    # dist.init_process_group(
    #     backend="nccl",  # 또는 "gloo" (CPU만 사용 시)
    #     rank=rank,
    #     world_size=world_size,
    #     init_method="env://"
    # )


if __name__ == '__main__':
    setup_distrubuted()
    main()
