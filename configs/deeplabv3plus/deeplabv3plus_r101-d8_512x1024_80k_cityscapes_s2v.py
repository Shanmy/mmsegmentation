_base_ = './deeplabv3plus_r50-d8_512x1024_80k_cityscapes_s2v.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
