import glob
import os
import time

import torch
from PIL import Image
from vizer.draw import draw_boxes

#from ssd.config import cfg
#from ssd.data.datasets import COCODataset, VOCDataset
import argparse
import numpy as np
from ext.utils import mkdir


#######
from mmdet.apis import init_detector, inference_detector
import mmcv
import pycocotools.mask as maskUtils


def show_result_viz(img,
                result,
                class_names,
                score_thr=0.3,
                wait_time=0,
                show=True,
                out_file=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        wait_time (int): Value of waitKey param.
        show (bool, optional): Whether to show the image with opencv or not.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """
    assert isinstance(class_names, (tuple, list))
    img = np.array(Image.open(img).convert("RGB"))
    #img = mmcv.imread(img)
    #img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)

    # uses vizer instead
    boxes = bboxes[:, 0:4]
    scores = bboxes[:, 4]
    indices = scores > score_thr
    boxes = boxes[indices]
    labels = labels[indices]
    scores = scores[indices]
    print(labels)

    drawn_image = draw_boxes(img, boxes, labels, scores, class_names).astype(np.uint8)
    Image.fromarray(drawn_image).save(out_file)

    # mmcv.imshow_det_bboxes(
    #     img,
    #     bboxes,
    #     labels,
    #     class_names=class_names,
    #     score_thr=score_thr,
    #     show=show,
    #     wait_time=wait_time,
    #     out_file=out_file)
    if not (show or out_file):
        return img



@torch.no_grad()
def run_demo(cfg, ckpt, score_threshold, images_dir, output_dir, dataset_type, device_type=None):
    # if dataset_type == "voc":
    #     class_names = VOCDataset.class_names
    # elif dataset_type == 'coco':
    #     class_names = COCODataset.class_names
    # else:
    #     raise NotImplementedError('unknown dataset type.')

    if device_type is None:
    	device = torch.device(cfg.MODEL.DEVICE)
    else:
        device = torch.device(device_type)


    ######
    config_file = cfg
    # download the checkpoint from model zoo and put it in `checkpoints/`
    checkpoint_file = ckpt
    model = init_detector(config_file, checkpoint_file, device=device)
    # test a single image
    ##################


    print('Loaded weights from {}'.format(ckpt))
    image_paths = []

    # for ending in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.JPEG']:
    #     image_paths.append(glob.glob(os.path.join(images_dir, ending)))
    image_paths = (glob.glob(os.path.join(images_dir, '*.jpg')))
    mkdir(output_dir)
    print('Images found: ' + str(len(image_paths)))

    cpu_device = torch.device("cpu")
    #transforms = build_transforms(cfg, is_train=False)

    for i, image_path in enumerate(image_paths):
        #start = time.time()
        image_name = os.path.basename(image_path)

        #image = np.array(Image.open(image_path).convert("RGB"))
        #height, width = image.shape[:2]
        # images = transforms(image)[0].unsqueeze(0)
        # load_time = time.time() - start
        #
        start = time.time()
        # result = model(images.to(device))[0]

        #
        # result = result.resize((width, height)).to(cpu_device).numpy()
        # boxes, labels, scores = result['boxes'], result['labels'], result['scores']

        result = inference_detector(model, image_path)
        inference_time = time.time() - start

        # boxes = np.vstack(result)
        # print(boxes)
        #
        # indices = scores > score_threshold
        # boxes = boxes[indices]
        # labels = labels[indices]
        # scores = scores[indices]
        meters = ' | '.join(
            [
                #'objects {:02d}'.format(len(boxes)),
                'inference {:03d}ms'.format(round(inference_time * 1000)),
                'FPS {}'.format(1.0 / inference_time)
            ]
        )
        print('({:04d}/{:04d}) {}: {}'.format(i + 1, len(image_paths), image_name, meters))

        show_result_viz(image_path, result, model.CLASSES, out_file=os.path.join(output_dir, image_name), show=False,
                    score_thr=score_threshold)
        # #drawn_image = draw_boxes(image, boxes, labels, scores, class_names).astype(np.uint8)
        # #Image.fromarray(drawn_image).save(os.path.join(output_dir, image_name))


def main():


    parser = argparse.ArgumentParser(description="Demo.")
    parser.add_argument(
        "--config_file",
        #default='../configs/hrnet/cascade_rcnn_hrnetv2p_w32_20e.py',
        default='../configs/hrnet/fcos_hrnetv2p_w32_gn_1x_1gpu_oi.py',
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    #parser.add_argument("--ckpt", type=str, default='../checkpoints/cascade_rcnn_hrnetv2p_w32_20e_20190522-55bec4ee.pth', help="Trained weights.")
    parser.add_argument("--ckpt", type=str, default='../work_dirs/fcos_hrnetv2p_w32_gn_1x_1gpu_oi/latest.pth', help="Trained weights.")
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--images_dir", default='inputs', type=str, help='Specify a image dir to do prediction.')
    parser.add_argument("--output_dir", default='results', type=str, help='Specify a image dir to save predicted images.')
    parser.add_argument("--dataset_type", default="coco", type=str, help='Specify dataset type. Currently supported: voc, coco.')
    parser.add_argument("--device", default="cuda", type=str, help='cuda or cpu')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )
    args = parser.parse_args()
    print(args)

    # cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    # cfg.freeze()

    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(args.config_file))

    run_demo(cfg=args.config_file,
             ckpt=args.ckpt,
             score_threshold=args.score_threshold,
             images_dir=args.images_dir,
             output_dir=args.output_dir,
             dataset_type=args.dataset_type,
             device_type = args.device)


if __name__ == '__main__':
    main()
