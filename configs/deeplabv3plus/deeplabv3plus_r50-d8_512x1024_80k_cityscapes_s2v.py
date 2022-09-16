_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8_s2v.py',
    '../_base_/datasets/cityscapes_s2v.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
