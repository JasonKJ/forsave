_base_ = '../cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'

# 1. dataset settings
dataset_type = 'CocoDataset'
classes = ('traffic_light', 'traffic_sign', 'traffic_information')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file='train_road/label/coco_road_info.json',
        img_prefix='train_road/image'),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file='test_road/label/coco_road_info.json',
        img_prefix='test_road/image'),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='test_road/label/coco_road_info.json',
        img_prefix='test_road/image'))

