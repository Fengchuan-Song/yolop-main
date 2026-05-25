import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from lib.config import cfg, update_config
from lib.core.general import non_max_suppression, scale_coords
from lib.models import get_net
from lib.utils import letterbox_for_img
from lib.utils.utils import select_device


IMG_FORMATS = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff'}


def parse_args():
    parser = argparse.ArgumentParser(description='Save YOLOP prediction results')
    parser.add_argument('--weights', type=str, default='/data/yolop/weights/yolop_weight.pth',
                        help='pretrained checkpoint path')
    parser.add_argument('--split', type=str, default='/data_ssd/datasets/WaterScenes/MIPC_shipOnly/2007_test.txt',
                        help='test split txt. Each line can be an image path, relative path, or image id')
    parser.add_argument('--data-root', type=str, default='/data_ssd/datasets/WaterScenes',
                        help='dataset root path')
    parser.add_argument('--image-folder', type=str, default='images',
                        help='image folder name under data-root')
    parser.add_argument('--output-dir', type=str, default='/data/yolop/predicted_results',
                        help='directory used to save DetectionResults and SegmentationClass')
    parser.add_argument('--image-format', type=str, default='jpg',
                        help='image suffix used when split lines contain only ids')
    parser.add_argument('--img-size', type=int, default=320,
                        help='inference image size')
    parser.add_argument('--seg-classes', type=int, default=9,
                        help='number of semantic segmentation classes expected in VOC masks')
    parser.add_argument('--conf-thres', type=float, default=0.35,
                        help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.35,
                        help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 inference on CUDA')
    parser.add_argument('--det-class-names', type=str, default='',
                        help='optional comma-separated detection class names')
    parser.add_argument('--modelDir', type=str, default='', help=argparse.SUPPRESS)
    parser.add_argument('--logDir', type=str, default='', help=argparse.SUPPRESS)
    parser.add_argument('--dataDir', type=str, default='', help=argparse.SUPPRESS)
    parser.add_argument('--prevModelDir', type=str, default='', help=argparse.SUPPRESS)
    return parser.parse_args()


def voc_palette(num_classes=256):
    palette = []
    for j in range(num_classes):
        lab = j
        r = g = b = 0
        i = 0
        while lab:
            r |= (((lab >> 0) & 1) << (7 - i))
            g |= (((lab >> 1) & 1) << (7 - i))
            b |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
        palette.extend([r, g, b])
    return palette


def read_split_images(split, data_root, image_folder, image_format):
    split = Path(split)
    data_root = Path(data_root)
    image_root = data_root / image_folder
    suffix = image_format if image_format.startswith('.') else '.' + image_format

    with split.open('r') as f:
        items = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Some WaterScenes split files store annotations after the image path:
            # image.jpg x1,y1,x2,y2,cls ...
            items.append(line.split()[0])

    image_paths = []
    for item in items:
        item_path = Path(item)
        candidates = []
        if item_path.is_absolute():
            candidates.append(item_path)
        else:
            candidates.extend([
                data_root / item_path,
                image_root / item_path,
            ])
        if not item_path.suffix:
            candidates.append(image_root / (item + suffix))

        image_path = next((p for p in candidates if p.exists()), candidates[-1])
        if image_path.suffix.lower() not in IMG_FORMATS:
            raise ValueError('Unsupported image format: {}'.format(image_path))
        if not image_path.exists():
            raise FileNotFoundError('Image not found for split item "{}": {}'.format(item, image_path))
        image_paths.append(image_path)
    return image_paths


def load_checkpoint(model, weights, device):
    checkpoint = torch.load(weights, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    model_dict = model.state_dict()
    model_dict.update({k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape})
    model.load_state_dict(model_dict)
    return model


def make_names(model, opt):
    if opt.det_class_names:
        return [x.strip() for x in opt.det_class_names.split(',') if x.strip()]
    return model.module.names if hasattr(model, 'module') else model.names


def preprocess_image(image_bgr, img_size, transform):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_lb, ratio, pad = letterbox_for_img(image_rgb, new_shape=img_size, auto=True)
    image_lb = np.ascontiguousarray(image_lb)
    tensor = transform(image_lb).unsqueeze(0)
    shapes = (image_bgr.shape[0], image_bgr.shape[1]), ((ratio[1], ratio[0]), pad)
    return tensor, image_lb.shape[:2], shapes


def save_detection_txt(path, detections, names):
    with path.open('w') as f:
        for *xyxy, conf, cls in detections:
            cls = int(cls)
            name = names[cls] if cls < len(names) else str(cls)
            x1, y1, x2, y2 = [int(round(float(x))) for x in xyxy]
            f.write('{} {:.6f} {} {} {} {}\n'.format(name, float(conf), x1, y1, x2, y2))


def save_voc_mask(path, mask, seg_classes):
    mask = np.clip(mask, 0, seg_classes - 1).astype(np.uint8)
    image = Image.fromarray(mask, mode='P')
    image.putpalette(voc_palette())
    image.save(path)


def predict(cfg, opt):
    device = select_device(None, opt.device)
    half = opt.half and device.type != 'cpu'
    cudnn.benchmark = device.type != 'cpu'

    model = get_net(cfg)
    model = load_checkpoint(model, opt.weights, device).to(device).eval()
    if half:
        model.half()

    names = make_names(model, opt)
    det_dir = Path(opt.output_dir) / 'DetectionResults'
    seg_dir = Path(opt.output_dir) / 'SegmentationClass'
    det_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    image_paths = read_split_images(opt.split, opt.data_root, opt.image_folder, opt.image_format)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    with torch.no_grad():
        for image_path in tqdm(image_paths, desc='Predict'):
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if image_bgr is None:
                raise FileNotFoundError('Image not readable: {}'.format(image_path))

            img, input_hw, shapes = preprocess_image(image_bgr, opt.img_size, transform)
            img = img.to(device)
            img = img.half() if half else img.float()

            det_out, seg_out, _ = model(img)
            inf_out, _ = det_out
            detections = non_max_suppression(
                inf_out,
                conf_thres=opt.conf_thres,
                iou_thres=opt.iou_thres,
                classes=None,
                agnostic=False
            )[0]

            predn = detections.clone()
            if len(predn):
                scale_coords(input_hw, predn[:, :4], image_bgr.shape, shapes[1]).round()
            save_detection_txt(det_dir / (image_path.stem + '.txt'), predn.tolist(), names)

            pad_w, pad_h = shapes[1][1]
            pad_w, pad_h = int(pad_w), int(pad_h)
            h, w = input_hw
            seg_crop = seg_out[:, :, pad_h:h - pad_h, pad_w:w - pad_w]
            seg_resized = F.interpolate(
                seg_crop,
                size=image_bgr.shape[:2],
                mode='bilinear',
                align_corners=False
            )
            seg_mask = torch.argmax(seg_resized, dim=1).squeeze(0).cpu().numpy()
            save_voc_mask(seg_dir / (image_path.stem + '.png'), seg_mask, opt.seg_classes)

    print('Detection results saved to {}'.format(det_dir))
    print('Segmentation results saved to {}'.format(seg_dir))


def main():
    opt = parse_args()
    update_config(cfg, opt)
    with torch.no_grad():
        predict(cfg, opt)


if __name__ == '__main__':
    main()
