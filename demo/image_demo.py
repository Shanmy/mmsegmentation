# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
#from mmseg.models.segmentors import encode_decode
import cv2 
import torch
import numpy as np

from pdb import set_trace as st


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_segmentor(model, args.img)
    # show probability
    img = torch.tensor(cv2.cvtColor(cv2.imread(args.img), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None, :]).cuda().float() / 255
    probs = model.encode_decode(img, img_metas={})
    classes = np.argmax(probs.cpu().detach().numpy(), axis=1)[0]
    norm_probs = np.clip(((probs[0, 1, :, :].cpu().detach().numpy() + 27) * 7), 0, 255).astype(np.uint8)
    blur_vis = cv2.applyColorMap(norm_probs, cv2.COLORMAP_JET)
    cv2.imwrite("heat.png", blur_vis)
    # st()
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result,
        get_palette(args.palette),
        opacity=args.opacity,
        out_file=args.out_file)


if __name__ == '__main__':
    main()
