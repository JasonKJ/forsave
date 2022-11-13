/usr/local/lib/python3.7/dist-packages/mmcv/__init__.py:21: UserWarning: On January 1, 2023, MMCV will release v2.0.0, in which it will remove components related to the training process and add a data transformation module. In addition, it will rename the package names mmcv to mmcv-lite and mmcv-full to mmcv. See https://github.com/open-mmlab/mmcv/blob/master/docs/en/compatibility.md for more details.
  'On January 1, 2023, MMCV will release v2.0.0, in which it will remove '
/content/mmdetection/mmdet/utils/setup_env.py:39: UserWarning: Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
  f'Setting OMP_NUM_THREADS environment variable for each process '
/content/mmdetection/mmdet/utils/setup_env.py:49: UserWarning: Setting MKL_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
  f'Setting MKL_NUM_THREADS environment variable for each process '
2022-11-12 18:44:37,755 - mmdet - INFO - Environment info:
------------------------------------------------------------
sys.platform: linux
Python: 3.7.15 (default, Oct 12 2022, 19:14:55) [GCC 7.5.0]
CUDA available: True
GPU 0: Tesla T4
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 11.2, V11.2.152
GCC: x86_64-linux-gnu-gcc (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
PyTorch: 1.9.0+cu111
PyTorch compiling details: PyTorch built with:
  - GCC 7.3
  - C++ Version: 201402
  - Intel(R) Math Kernel Library Version 2020.0.0 Product Build 20191122 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v2.1.2 (Git Hash 98be7e8afa711dc9b66c8ff3504129cb82013cdb)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - NNPACK is enabled
  - CPU capability usage: AVX2
  - CUDA Runtime 11.1
  - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86
  - CuDNN 8.0.5
  - Magma 2.5.2
  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=11.1, CUDNN_VERSION=8.0.5, CXX_COMPILER=/opt/rh/devtoolset-7/root/usr/bin/c++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.9.0, USE_CUDA=ON, USE_CUDNN=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON,

TorchVision: 0.10.0+cu111
OpenCV: 4.6.0
MMCV: 1.7.0
MMCV Compiler: GCC 7.3
MMCV CUDA Compiler: 11.1
MMDetection: 2.25.3+e71b499
------------------------------------------------------------

2022-11-12 18:44:38,170 - mmdet - INFO - Distributed training: False
2022-11-12 18:44:38,562 - mmdet - INFO - Config:
model = dict(
    type='CascadeRCNN',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=3,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=3,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=3,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file='train_road/label/coco_road_info.json',
        img_prefix='train_road/image',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        classes=('traffic_light', 'traffic_sign', 'traffic_information')),
    val=dict(
        type='CocoDataset',
        ann_file='test_road/label/coco_road_info.json',
        img_prefix='test_road/image',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('traffic_light', 'traffic_sign', 'traffic_information')),
    test=dict(
        type='CocoDataset',
        ann_file='test_road/label/coco_road_info.json',
        img_prefix='test_road/image',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('traffic_light', 'traffic_sign', 'traffic_information')))
evaluation = dict(interval=1, metric='bbox')
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
classes = ('traffic_light', 'traffic_sign', 'traffic_information')
work_dir = './work_dirs/customized_model_road'
auto_resume = False
gpu_ids = [0]

2022-11-12 18:44:38,613 - mmdet - INFO - Set random seed to 1524488120, deterministic: False
2022-11-12 18:44:39,434 - mmdet - INFO - initialize ResNet with init_cfg {'type': 'Pretrained', 'checkpoint': 'torchvision://resnet50'}
2022-11-12 18:44:39,434 - mmcv - INFO - load model from: torchvision://resnet50
2022-11-12 18:44:39,434 - mmcv - INFO - load checkpoint from torchvision path: torchvision://resnet50
2022-11-12 18:44:39,554 - mmcv - WARNING - The model and loaded state dict do not match exactly

unexpected key in source state_dict: fc.weight, fc.bias

missing keys in source state_dict: layer3.6.conv1.weight, layer3.6.bn1.weight, layer3.6.bn1.bias, layer3.6.bn1.running_mean, layer3.6.bn1.running_var, layer3.6.conv2.weight, layer3.6.bn2.weight, layer3.6.bn2.bias, layer3.6.bn2.running_mean, layer3.6.bn2.running_var, layer3.6.conv3.weight, layer3.6.bn3.weight, layer3.6.bn3.bias, layer3.6.bn3.running_mean, layer3.6.bn3.running_var, layer3.7.conv1.weight, layer3.7.bn1.weight, layer3.7.bn1.bias, layer3.7.bn1.running_mean, layer3.7.bn1.running_var, layer3.7.conv2.weight, layer3.7.bn2.weight, layer3.7.bn2.bias, layer3.7.bn2.running_mean, layer3.7.bn2.running_var, layer3.7.conv3.weight, layer3.7.bn3.weight, layer3.7.bn3.bias, layer3.7.bn3.running_mean, layer3.7.bn3.running_var, layer3.8.conv1.weight, layer3.8.bn1.weight, layer3.8.bn1.bias, layer3.8.bn1.running_mean, layer3.8.bn1.running_var, layer3.8.conv2.weight, layer3.8.bn2.weight, layer3.8.bn2.bias, layer3.8.bn2.running_mean, layer3.8.bn2.running_var, layer3.8.conv3.weight, layer3.8.bn3.weight, layer3.8.bn3.bias, layer3.8.bn3.running_mean, layer3.8.bn3.running_var, layer3.9.conv1.weight, layer3.9.bn1.weight, layer3.9.bn1.bias, layer3.9.bn1.running_mean, layer3.9.bn1.running_var, layer3.9.conv2.weight, layer3.9.bn2.weight, layer3.9.bn2.bias, layer3.9.bn2.running_mean, layer3.9.bn2.running_var, layer3.9.conv3.weight, layer3.9.bn3.weight, layer3.9.bn3.bias, layer3.9.bn3.running_mean, layer3.9.bn3.running_var, layer3.10.conv1.weight, layer3.10.bn1.weight, layer3.10.bn1.bias, layer3.10.bn1.running_mean, layer3.10.bn1.running_var, layer3.10.conv2.weight, layer3.10.bn2.weight, layer3.10.bn2.bias, layer3.10.bn2.running_mean, layer3.10.bn2.running_var, layer3.10.conv3.weight, layer3.10.bn3.weight, layer3.10.bn3.bias, layer3.10.bn3.running_mean, layer3.10.bn3.running_var, layer3.11.conv1.weight, layer3.11.bn1.weight, layer3.11.bn1.bias, layer3.11.bn1.running_mean, layer3.11.bn1.running_var, layer3.11.conv2.weight, layer3.11.bn2.weight, layer3.11.bn2.bias, layer3.11.bn2.running_mean, layer3.11.bn2.running_var, layer3.11.conv3.weight, layer3.11.bn3.weight, layer3.11.bn3.bias, layer3.11.bn3.running_mean, layer3.11.bn3.running_var, layer3.12.conv1.weight, layer3.12.bn1.weight, layer3.12.bn1.bias, layer3.12.bn1.running_mean, layer3.12.bn1.running_var, layer3.12.conv2.weight, layer3.12.bn2.weight, layer3.12.bn2.bias, layer3.12.bn2.running_mean, layer3.12.bn2.running_var, layer3.12.conv3.weight, layer3.12.bn3.weight, layer3.12.bn3.bias, layer3.12.bn3.running_mean, layer3.12.bn3.running_var, layer3.13.conv1.weight, layer3.13.bn1.weight, layer3.13.bn1.bias, layer3.13.bn1.running_mean, layer3.13.bn1.running_var, layer3.13.conv2.weight, layer3.13.bn2.weight, layer3.13.bn2.bias, layer3.13.bn2.running_mean, layer3.13.bn2.running_var, layer3.13.conv3.weight, layer3.13.bn3.weight, layer3.13.bn3.bias, layer3.13.bn3.running_mean, layer3.13.bn3.running_var, layer3.14.conv1.weight, layer3.14.bn1.weight, layer3.14.bn1.bias, layer3.14.bn1.running_mean, layer3.14.bn1.running_var, layer3.14.conv2.weight, layer3.14.bn2.weight, layer3.14.bn2.bias, layer3.14.bn2.running_mean, layer3.14.bn2.running_var, layer3.14.conv3.weight, layer3.14.bn3.weight, layer3.14.bn3.bias, layer3.14.bn3.running_mean, layer3.14.bn3.running_var, layer3.15.conv1.weight, layer3.15.bn1.weight, layer3.15.bn1.bias, layer3.15.bn1.running_mean, layer3.15.bn1.running_var, layer3.15.conv2.weight, layer3.15.bn2.weight, layer3.15.bn2.bias, layer3.15.bn2.running_mean, layer3.15.bn2.running_var, layer3.15.conv3.weight, layer3.15.bn3.weight, layer3.15.bn3.bias, layer3.15.bn3.running_mean, layer3.15.bn3.running_var, layer3.16.conv1.weight, layer3.16.bn1.weight, layer3.16.bn1.bias, layer3.16.bn1.running_mean, layer3.16.bn1.running_var, layer3.16.conv2.weight, layer3.16.bn2.weight, layer3.16.bn2.bias, layer3.16.bn2.running_mean, layer3.16.bn2.running_var, layer3.16.conv3.weight, layer3.16.bn3.weight, layer3.16.bn3.bias, layer3.16.bn3.running_mean, layer3.16.bn3.running_var, layer3.17.conv1.weight, layer3.17.bn1.weight, layer3.17.bn1.bias, layer3.17.bn1.running_mean, layer3.17.bn1.running_var, layer3.17.conv2.weight, layer3.17.bn2.weight, layer3.17.bn2.bias, layer3.17.bn2.running_mean, layer3.17.bn2.running_var, layer3.17.conv3.weight, layer3.17.bn3.weight, layer3.17.bn3.bias, layer3.17.bn3.running_mean, layer3.17.bn3.running_var, layer3.18.conv1.weight, layer3.18.bn1.weight, layer3.18.bn1.bias, layer3.18.bn1.running_mean, layer3.18.bn1.running_var, layer3.18.conv2.weight, layer3.18.bn2.weight, layer3.18.bn2.bias, layer3.18.bn2.running_mean, layer3.18.bn2.running_var, layer3.18.conv3.weight, layer3.18.bn3.weight, layer3.18.bn3.bias, layer3.18.bn3.running_mean, layer3.18.bn3.running_var, layer3.19.conv1.weight, layer3.19.bn1.weight, layer3.19.bn1.bias, layer3.19.bn1.running_mean, layer3.19.bn1.running_var, layer3.19.conv2.weight, layer3.19.bn2.weight, layer3.19.bn2.bias, layer3.19.bn2.running_mean, layer3.19.bn2.running_var, layer3.19.conv3.weight, layer3.19.bn3.weight, layer3.19.bn3.bias, layer3.19.bn3.running_mean, layer3.19.bn3.running_var, layer3.20.conv1.weight, layer3.20.bn1.weight, layer3.20.bn1.bias, layer3.20.bn1.running_mean, layer3.20.bn1.running_var, layer3.20.conv2.weight, layer3.20.bn2.weight, layer3.20.bn2.bias, layer3.20.bn2.running_mean, layer3.20.bn2.running_var, layer3.20.conv3.weight, layer3.20.bn3.weight, layer3.20.bn3.bias, layer3.20.bn3.running_mean, layer3.20.bn3.running_var, layer3.21.conv1.weight, layer3.21.bn1.weight, layer3.21.bn1.bias, layer3.21.bn1.running_mean, layer3.21.bn1.running_var, layer3.21.conv2.weight, layer3.21.bn2.weight, layer3.21.bn2.bias, layer3.21.bn2.running_mean, layer3.21.bn2.running_var, layer3.21.conv3.weight, layer3.21.bn3.weight, layer3.21.bn3.bias, layer3.21.bn3.running_mean, layer3.21.bn3.running_var, layer3.22.conv1.weight, layer3.22.bn1.weight, layer3.22.bn1.bias, layer3.22.bn1.running_mean, layer3.22.bn1.running_var, layer3.22.conv2.weight, layer3.22.bn2.weight, layer3.22.bn2.bias, layer3.22.bn2.running_mean, layer3.22.bn2.running_var, layer3.22.conv3.weight, layer3.22.bn3.weight, layer3.22.bn3.bias, layer3.22.bn3.running_mean, layer3.22.bn3.running_var

2022-11-12 18:44:39,606 - mmdet - INFO - initialize FPN with init_cfg {'type': 'Xavier', 'layer': 'Conv2d', 'distribution': 'uniform'}
2022-11-12 18:44:39,630 - mmdet - INFO - initialize RPNHead with init_cfg {'type': 'Normal', 'layer': 'Conv2d', 'std': 0.01}
2022-11-12 18:44:39,635 - mmdet - INFO - initialize Shared2FCBBoxHead with init_cfg [{'type': 'Normal', 'std': 0.01, 'override': {'name': 'fc_cls'}}, {'type': 'Normal', 'std': 0.001, 'override': {'name': 'fc_reg'}}, {'type': 'Xavier', 'distribution': 'uniform', 'override': [{'name': 'shared_fcs'}, {'name': 'cls_fcs'}, {'name': 'reg_fcs'}]}]
2022-11-12 18:44:39,753 - mmdet - INFO - initialize Shared2FCBBoxHead with init_cfg [{'type': 'Normal', 'std': 0.01, 'override': {'name': 'fc_cls'}}, {'type': 'Normal', 'std': 0.001, 'override': {'name': 'fc_reg'}}, {'type': 'Xavier', 'distribution': 'uniform', 'override': [{'name': 'shared_fcs'}, {'name': 'cls_fcs'}, {'name': 'reg_fcs'}]}]
2022-11-12 18:44:39,865 - mmdet - INFO - initialize Shared2FCBBoxHead with init_cfg [{'type': 'Normal', 'std': 0.01, 'override': {'name': 'fc_cls'}}, {'type': 'Normal', 'std': 0.001, 'override': {'name': 'fc_reg'}}, {'type': 'Xavier', 'distribution': 'uniform', 'override': [{'name': 'shared_fcs'}, {'name': 'cls_fcs'}, {'name': 'reg_fcs'}]}]
loading annotations into memory...
Done (t=0.03s)
creating index...
index created!
2022-11-12 18:44:44,467 - mmdet - INFO - Automatic scaling of learning rate (LR) has been disabled.
loading annotations into memory...
Done (t=0.00s)
creating index...
index created!
2022-11-12 18:44:44,478 - mmdet - INFO - Start running, host: root@b915b7d09177, work_dir: /content/mmdetection/work_dirs/customized_model_road
2022-11-12 18:44:44,479 - mmdet - INFO - Hooks will be executed in the following order:
before_run:
(VERY_HIGH   ) StepLrUpdaterHook
(NORMAL      ) CheckpointHook
(LOW         ) EvalHook
(VERY_LOW    ) TextLoggerHook
 --------------------
before_train_epoch:
(VERY_HIGH   ) StepLrUpdaterHook
(NORMAL      ) NumClassCheckHook
(LOW         ) IterTimerHook
(LOW         ) EvalHook
(VERY_LOW    ) TextLoggerHook
 --------------------
before_train_iter:
(VERY_HIGH   ) StepLrUpdaterHook
(LOW         ) IterTimerHook
(LOW         ) EvalHook
 --------------------
after_train_iter:
(ABOVE_NORMAL) OptimizerHook
(NORMAL      ) CheckpointHook
(LOW         ) IterTimerHook
(LOW         ) EvalHook
(VERY_LOW    ) TextLoggerHook
 --------------------
after_train_epoch:
(NORMAL      ) CheckpointHook
(LOW         ) EvalHook
(VERY_LOW    ) TextLoggerHook
 --------------------
before_val_epoch:
(NORMAL      ) NumClassCheckHook
(LOW         ) IterTimerHook
(VERY_LOW    ) TextLoggerHook
 --------------------
before_val_iter:
(LOW         ) IterTimerHook
 --------------------
after_val_iter:
(LOW         ) IterTimerHook
 --------------------
after_val_epoch:
(VERY_LOW    ) TextLoggerHook
 --------------------
after_run:
(VERY_LOW    ) TextLoggerHook
 --------------------
2022-11-12 18:44:44,479 - mmdet - INFO - workflow: [('train', 1)], max: 12 epochs
2022-11-12 18:44:44,479 - mmdet - INFO - Checkpoints will be saved to /content/mmdetection/work_dirs/customized_model_road by HardDiskBackend.
/usr/local/lib/python3.7/dist-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.)
  return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
2022-11-12 18:45:27,590 - mmdet - INFO - Epoch [1][50/962]	lr: 1.978e-03, eta: 2:45:05, time: 0.862, data_time: 0.058, memory: 5062, loss_rpn_cls: 0.3404, loss_rpn_bbox: 0.0274, s0.loss_cls: 0.4566, s0.acc: 86.6973, s0.loss_bbox: 0.1237, s1.loss_cls: 0.1925, s1.acc: 88.9062, s1.loss_bbox: 0.0483, s2.loss_cls: 0.0940, s2.acc: 88.6973, s2.loss_bbox: 0.0075, loss: 1.2903
2022-11-12 18:46:10,725 - mmdet - INFO - Epoch [1][100/962]	lr: 3.976e-03, eta: 2:44:27, time: 0.863, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0396, loss_rpn_bbox: 0.0196, s0.loss_cls: 0.2298, s0.acc: 92.2793, s0.loss_bbox: 0.1638, s1.loss_cls: 0.0885, s1.acc: 94.7891, s1.loss_bbox: 0.0977, s2.loss_cls: 0.0282, s2.acc: 97.3867, s2.loss_bbox: 0.0218, loss: 0.6892
2022-11-12 18:46:56,038 - mmdet - INFO - Epoch [1][150/962]	lr: 5.974e-03, eta: 2:46:31, time: 0.906, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0251, loss_rpn_bbox: 0.0211, s0.loss_cls: 0.2150, s0.acc: 92.2539, s0.loss_bbox: 0.1594, s1.loss_cls: 0.0894, s1.acc: 94.3555, s1.loss_bbox: 0.0997, s2.loss_cls: 0.0301, s2.acc: 96.8965, s2.loss_bbox: 0.0264, loss: 0.6661
2022-11-12 18:47:40,934 - mmdet - INFO - Epoch [1][200/962]	lr: 7.972e-03, eta: 2:46:47, time: 0.898, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0200, loss_rpn_bbox: 0.0185, s0.loss_cls: 0.2124, s0.acc: 92.6562, s0.loss_bbox: 0.1486, s1.loss_cls: 0.0895, s1.acc: 94.2520, s1.loss_bbox: 0.0992, s2.loss_cls: 0.0317, s2.acc: 96.6230, s2.loss_bbox: 0.0293, loss: 0.6493
2022-11-12 18:48:26,067 - mmdet - INFO - Epoch [1][250/962]	lr: 9.970e-03, eta: 2:46:49, time: 0.903, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0267, loss_rpn_bbox: 0.0188, s0.loss_cls: 0.2283, s0.acc: 92.2500, s0.loss_bbox: 0.1515, s1.loss_cls: 0.0960, s1.acc: 93.8379, s1.loss_bbox: 0.1028, s2.loss_cls: 0.0343, s2.acc: 96.2441, s2.loss_bbox: 0.0320, loss: 0.6903
2022-11-12 18:49:11,094 - mmdet - INFO - Epoch [1][300/962]	lr: 1.197e-02, eta: 2:46:31, time: 0.901, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0272, loss_rpn_bbox: 0.0219, s0.loss_cls: 0.2064, s0.acc: 92.9180, s0.loss_bbox: 0.1422, s1.loss_cls: 0.0862, s1.acc: 94.6914, s1.loss_bbox: 0.0877, s2.loss_cls: 0.0314, s2.acc: 96.7148, s2.loss_bbox: 0.0267, loss: 0.6296
2022-11-12 18:49:56,785 - mmdet - INFO - Epoch [1][350/962]	lr: 1.397e-02, eta: 2:46:27, time: 0.914, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0260, loss_rpn_bbox: 0.0216, s0.loss_cls: 0.2293, s0.acc: 92.2559, s0.loss_bbox: 0.1537, s1.loss_cls: 0.0931, s1.acc: 94.3242, s1.loss_bbox: 0.0976, s2.loss_cls: 0.0328, s2.acc: 96.6484, s2.loss_bbox: 0.0285, loss: 0.6825
2022-11-12 18:50:42,013 - mmdet - INFO - Epoch [1][400/962]	lr: 1.596e-02, eta: 2:46:00, time: 0.905, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0299, loss_rpn_bbox: 0.0237, s0.loss_cls: 0.2400, s0.acc: 92.6641, s0.loss_bbox: 0.1555, s1.loss_cls: 0.0913, s1.acc: 95.0801, s1.loss_bbox: 0.0847, s2.loss_cls: 0.0303, s2.acc: 97.2520, s2.loss_bbox: 0.0219, loss: 0.6773
2022-11-12 18:51:26,876 - mmdet - INFO - Epoch [1][450/962]	lr: 1.796e-02, eta: 2:45:19, time: 0.897, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0335, loss_rpn_bbox: 0.0250, s0.loss_cls: 0.2554, s0.acc: 92.5430, s0.loss_bbox: 0.1637, s1.loss_cls: 0.0931, s1.acc: 94.9121, s1.loss_bbox: 0.0914, s2.loss_cls: 0.0291, s2.acc: 97.3262, s2.loss_bbox: 0.0209, loss: 0.7121
2022-11-12 18:52:11,359 - mmdet - INFO - Epoch [1][500/962]	lr: 1.996e-02, eta: 2:44:30, time: 0.890, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0349, loss_rpn_bbox: 0.0249, s0.loss_cls: 0.2415, s0.acc: 92.7031, s0.loss_bbox: 0.1596, s1.loss_cls: 0.0878, s1.acc: 95.4121, s1.loss_bbox: 0.0826, s2.loss_cls: 0.0280, s2.acc: 97.6016, s2.loss_bbox: 0.0186, loss: 0.6778
2022-11-12 18:52:56,646 - mmdet - INFO - Epoch [1][550/962]	lr: 2.000e-02, eta: 2:43:57, time: 0.906, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0352, loss_rpn_bbox: 0.0257, s0.loss_cls: 0.2669, s0.acc: 90.5859, s0.loss_bbox: 0.1978, s1.loss_cls: 0.0995, s1.acc: 94.0781, s1.loss_bbox: 0.1052, s2.loss_cls: 0.0329, s2.acc: 96.8711, s2.loss_bbox: 0.0251, loss: 0.7883
2022-11-12 18:53:41,717 - mmdet - INFO - Epoch [1][600/962]	lr: 2.000e-02, eta: 2:43:18, time: 0.901, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0304, loss_rpn_bbox: 0.0249, s0.loss_cls: 0.2267, s0.acc: 92.1484, s0.loss_bbox: 0.1618, s1.loss_cls: 0.0909, s1.acc: 94.3770, s1.loss_bbox: 0.0996, s2.loss_cls: 0.0305, s2.acc: 96.9160, s2.loss_bbox: 0.0252, loss: 0.6899
2022-11-12 18:54:26,852 - mmdet - INFO - Epoch [1][650/962]	lr: 2.000e-02, eta: 2:42:40, time: 0.903, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0320, loss_rpn_bbox: 0.0191, s0.loss_cls: 0.2194, s0.acc: 92.7031, s0.loss_bbox: 0.1532, s1.loss_cls: 0.0838, s1.acc: 95.1543, s1.loss_bbox: 0.0838, s2.loss_cls: 0.0274, s2.acc: 97.2715, s2.loss_bbox: 0.0219, loss: 0.6406
2022-11-12 18:55:11,457 - mmdet - INFO - Epoch [1][700/962]	lr: 2.000e-02, eta: 2:41:52, time: 0.892, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0195, loss_rpn_bbox: 0.0210, s0.loss_cls: 0.2106, s0.acc: 92.6172, s0.loss_bbox: 0.1476, s1.loss_cls: 0.0912, s1.acc: 94.2617, s1.loss_bbox: 0.0969, s2.loss_cls: 0.0322, s2.acc: 96.5723, s2.loss_bbox: 0.0286, loss: 0.6476
2022-11-12 18:55:56,390 - mmdet - INFO - Epoch [1][750/962]	lr: 2.000e-02, eta: 2:41:09, time: 0.899, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0315, loss_rpn_bbox: 0.0216, s0.loss_cls: 0.2319, s0.acc: 91.9102, s0.loss_bbox: 0.1655, s1.loss_cls: 0.0951, s1.acc: 93.9785, s1.loss_bbox: 0.1026, s2.loss_cls: 0.0328, s2.acc: 96.4551, s2.loss_bbox: 0.0291, loss: 0.7099
2022-11-12 18:56:41,783 - mmdet - INFO - Epoch [1][800/962]	lr: 2.000e-02, eta: 2:40:33, time: 0.908, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0255, loss_rpn_bbox: 0.0213, s0.loss_cls: 0.2366, s0.acc: 91.4043, s0.loss_bbox: 0.1715, s1.loss_cls: 0.0999, s1.acc: 93.3965, s1.loss_bbox: 0.1121, s2.loss_cls: 0.0354, s2.acc: 96.0957, s2.loss_bbox: 0.0339, loss: 0.7364
2022-11-12 18:57:26,823 - mmdet - INFO - Epoch [1][850/962]	lr: 2.000e-02, eta: 2:39:50, time: 0.901, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0215, loss_rpn_bbox: 0.0164, s0.loss_cls: 0.2176, s0.acc: 92.0820, s0.loss_bbox: 0.1617, s1.loss_cls: 0.0921, s1.acc: 93.7109, s1.loss_bbox: 0.1053, s2.loss_cls: 0.0333, s2.acc: 96.1953, s2.loss_bbox: 0.0322, loss: 0.6801
2022-11-12 18:58:12,006 - mmdet - INFO - Epoch [1][900/962]	lr: 2.000e-02, eta: 2:39:10, time: 0.904, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0210, loss_rpn_bbox: 0.0170, s0.loss_cls: 0.2140, s0.acc: 92.2324, s0.loss_bbox: 0.1529, s1.loss_cls: 0.0887, s1.acc: 93.8965, s1.loss_bbox: 0.1030, s2.loss_cls: 0.0331, s2.acc: 95.9141, s2.loss_bbox: 0.0346, loss: 0.6642
2022-11-12 18:58:57,266 - mmdet - INFO - Epoch [1][950/962]	lr: 2.000e-02, eta: 2:38:29, time: 0.905, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0246, loss_rpn_bbox: 0.0229, s0.loss_cls: 0.2206, s0.acc: 91.6875, s0.loss_bbox: 0.1574, s1.loss_cls: 0.0988, s1.acc: 92.8457, s1.loss_bbox: 0.1152, s2.loss_cls: 0.0375, s2.acc: 95.1133, s2.loss_bbox: 0.0413, loss: 0.7182
2022-11-12 18:59:08,070 - mmdet - INFO - Saving checkpoint at 1 epochs
[>>] 200/200, 5.1 task/s, elapsed: 39s, ETA:     0s2022-11-12 18:59:53,503 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.38s).
Accumulating evaluation results...
DONE (t=0.08s).
2022-11-12 18:59:54,113 - mmdet - INFO -
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.021
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.094
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.001
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.021
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.125
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.125
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.125
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.125

2022-11-12 18:59:54,118 - mmdet - INFO - Exp name: customized_model_road.py
2022-11-12 18:59:54,118 - mmdet - INFO - Epoch(val) [1][200]	bbox_mAP: 0.0210, bbox_mAP_50: 0.0940, bbox_mAP_75: 0.0010, bbox_mAP_s: -1.0000, bbox_mAP_m: -1.0000, bbox_mAP_l: 0.0210, bbox_mAP_copypaste: 0.021 0.094 0.001 -1.000 -1.000 0.021
2022-11-12 19:00:42,595 - mmdet - INFO - Epoch [2][50/962]	lr: 2.000e-02, eta: 2:36:19, time: 0.969, data_time: 0.061, memory: 5062, loss_rpn_cls: 0.0290, loss_rpn_bbox: 0.0177, s0.loss_cls: 0.1979, s0.acc: 92.6914, s0.loss_bbox: 0.1398, s1.loss_cls: 0.0884, s1.acc: 93.7539, s1.loss_bbox: 0.1061, s2.loss_cls: 0.0337, s2.acc: 95.6328, s2.loss_bbox: 0.0382, loss: 0.6508
2022-11-12 19:01:27,643 - mmdet - INFO - Epoch [2][100/962]	lr: 2.000e-02, eta: 2:35:39, time: 0.901, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0210, loss_rpn_bbox: 0.0221, s0.loss_cls: 0.1997, s0.acc: 92.0723, s0.loss_bbox: 0.1428, s1.loss_cls: 0.0910, s1.acc: 92.7969, s1.loss_bbox: 0.1229, s2.loss_cls: 0.0379, s2.acc: 94.4355, s2.loss_bbox: 0.0505, loss: 0.6879
2022-11-12 19:02:12,460 - mmdet - INFO - Epoch [2][150/962]	lr: 2.000e-02, eta: 2:34:57, time: 0.896, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0171, loss_rpn_bbox: 0.0161, s0.loss_cls: 0.2103, s0.acc: 91.8418, s0.loss_bbox: 0.1481, s1.loss_cls: 0.0970, s1.acc: 92.4395, s1.loss_bbox: 0.1199, s2.loss_cls: 0.0386, s2.acc: 94.3457, s2.loss_bbox: 0.0484, loss: 0.6954
2022-11-12 19:02:57,320 - mmdet - INFO - Epoch [2][200/962]	lr: 2.000e-02, eta: 2:34:15, time: 0.897, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0199, loss_rpn_bbox: 0.0174, s0.loss_cls: 0.1915, s0.acc: 92.6270, s0.loss_bbox: 0.1328, s1.loss_cls: 0.0843, s1.acc: 93.5586, s1.loss_bbox: 0.1054, s2.loss_cls: 0.0343, s2.acc: 95.0371, s2.loss_bbox: 0.0448, loss: 0.6304
2022-11-12 19:03:42,829 - mmdet - INFO - Epoch [2][250/962]	lr: 2.000e-02, eta: 2:33:39, time: 0.910, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0161, loss_rpn_bbox: 0.0153, s0.loss_cls: 0.2051, s0.acc: 92.2363, s0.loss_bbox: 0.1386, s1.loss_cls: 0.0940, s1.acc: 93.0176, s1.loss_bbox: 0.1145, s2.loss_cls: 0.0376, s2.acc: 94.7305, s2.loss_bbox: 0.0458, loss: 0.6671
2022-11-12 19:04:27,703 - mmdet - INFO - Epoch [2][300/962]	lr: 2.000e-02, eta: 2:32:56, time: 0.897, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0276, loss_rpn_bbox: 0.0193, s0.loss_cls: 0.2165, s0.acc: 91.4531, s0.loss_bbox: 0.1551, s1.loss_cls: 0.0968, s1.acc: 92.6504, s1.loss_bbox: 0.1232, s2.loss_cls: 0.0391, s2.acc: 94.3262, s2.loss_bbox: 0.0496, loss: 0.7273
2022-11-12 19:05:12,715 - mmdet - INFO - Epoch [2][350/962]	lr: 2.000e-02, eta: 2:32:14, time: 0.900, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0221, loss_rpn_bbox: 0.0177, s0.loss_cls: 0.1948, s0.acc: 92.9941, s0.loss_bbox: 0.1288, s1.loss_cls: 0.0872, s1.acc: 93.9531, s1.loss_bbox: 0.1014, s2.loss_cls: 0.0330, s2.acc: 95.6562, s2.loss_bbox: 0.0385, loss: 0.6235
2022-11-12 19:05:57,918 - mmdet - INFO - Epoch [2][400/962]	lr: 2.000e-02, eta: 2:31:34, time: 0.904, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0173, loss_rpn_bbox: 0.0124, s0.loss_cls: 0.1907, s0.acc: 92.8848, s0.loss_bbox: 0.1318, s1.loss_cls: 0.0861, s1.acc: 93.6133, s1.loss_bbox: 0.1081, s2.loss_cls: 0.0347, s2.acc: 94.8555, s2.loss_bbox: 0.0473, loss: 0.6284
2022-11-12 19:06:43,344 - mmdet - INFO - Epoch [2][450/962]	lr: 2.000e-02, eta: 2:30:55, time: 0.909, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0185, loss_rpn_bbox: 0.0127, s0.loss_cls: 0.2026, s0.acc: 92.2676, s0.loss_bbox: 0.1372, s1.loss_cls: 0.0934, s1.acc: 92.8398, s1.loss_bbox: 0.1169, s2.loss_cls: 0.0381, s2.acc: 94.3164, s2.loss_bbox: 0.0492, loss: 0.6686
2022-11-12 19:07:29,415 - mmdet - INFO - Epoch [2][500/962]	lr: 2.000e-02, eta: 2:30:20, time: 0.921, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0164, loss_rpn_bbox: 0.0145, s0.loss_cls: 0.1864, s0.acc: 92.6602, s0.loss_bbox: 0.1289, s1.loss_cls: 0.0866, s1.acc: 93.1543, s1.loss_bbox: 0.1120, s2.loss_cls: 0.0373, s2.acc: 94.2480, s2.loss_bbox: 0.0556, loss: 0.6377
2022-11-12 19:08:14,672 - mmdet - INFO - Epoch [2][550/962]	lr: 2.000e-02, eta: 2:29:38, time: 0.905, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0159, loss_rpn_bbox: 0.0136, s0.loss_cls: 0.1946, s0.acc: 92.5957, s0.loss_bbox: 0.1451, s1.loss_cls: 0.0901, s1.acc: 93.2070, s1.loss_bbox: 0.1203, s2.loss_cls: 0.0380, s2.acc: 94.3730, s2.loss_bbox: 0.0543, loss: 0.6719
2022-11-12 19:08:59,985 - mmdet - INFO - Epoch [2][600/962]	lr: 2.000e-02, eta: 2:28:57, time: 0.906, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0194, loss_rpn_bbox: 0.0166, s0.loss_cls: 0.2114, s0.acc: 91.9180, s0.loss_bbox: 0.1557, s1.loss_cls: 0.0957, s1.acc: 92.7930, s1.loss_bbox: 0.1303, s2.loss_cls: 0.0401, s2.acc: 94.0898, s2.loss_bbox: 0.0566, loss: 0.7260
2022-11-12 19:09:45,323 - mmdet - INFO - Epoch [2][650/962]	lr: 2.000e-02, eta: 2:28:16, time: 0.907, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0177, loss_rpn_bbox: 0.0123, s0.loss_cls: 0.1952, s0.acc: 92.6445, s0.loss_bbox: 0.1339, s1.loss_cls: 0.0876, s1.acc: 93.4219, s1.loss_bbox: 0.1151, s2.loss_cls: 0.0368, s2.acc: 94.5371, s2.loss_bbox: 0.0543, loss: 0.6529
2022-11-12 19:10:30,529 - mmdet - INFO - Epoch [2][700/962]	lr: 2.000e-02, eta: 2:27:34, time: 0.904, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0227, loss_rpn_bbox: 0.0173, s0.loss_cls: 0.1933, s0.acc: 92.8965, s0.loss_bbox: 0.1335, s1.loss_cls: 0.0878, s1.acc: 93.6816, s1.loss_bbox: 0.1130, s2.loss_cls: 0.0357, s2.acc: 94.9043, s2.loss_bbox: 0.0497, loss: 0.6529
2022-11-12 19:11:15,964 - mmdet - INFO - Epoch [2][750/962]	lr: 2.000e-02, eta: 2:26:53, time: 0.909, data_time: 0.010, memory: 5062, loss_rpn_cls: 0.0139, loss_rpn_bbox: 0.0137, s0.loss_cls: 0.1982, s0.acc: 92.3027, s0.loss_bbox: 0.1442, s1.loss_cls: 0.0935, s1.acc: 92.9863, s1.loss_bbox: 0.1230, s2.loss_cls: 0.0394, s2.acc: 94.2227, s2.loss_bbox: 0.0548, loss: 0.6807
2022-11-12 19:12:01,010 - mmdet - INFO - Epoch [2][800/962]	lr: 2.000e-02, eta: 2:26:09, time: 0.901, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0172, loss_rpn_bbox: 0.0156, s0.loss_cls: 0.1887, s0.acc: 92.9219, s0.loss_bbox: 0.1278, s1.loss_cls: 0.0867, s1.acc: 93.5332, s1.loss_bbox: 0.1110, s2.loss_cls: 0.0369, s2.acc: 94.6016, s2.loss_bbox: 0.0510, loss: 0.6349
2022-11-12 19:12:46,353 - mmdet - INFO - Epoch [2][850/962]	lr: 2.000e-02, eta: 2:25:27, time: 0.907, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0162, loss_rpn_bbox: 0.0137, s0.loss_cls: 0.1809, s0.acc: 93.0137, s0.loss_bbox: 0.1258, s1.loss_cls: 0.0863, s1.acc: 93.2539, s1.loss_bbox: 0.1131, s2.loss_cls: 0.0378, s2.acc: 94.1016, s2.loss_bbox: 0.0556, loss: 0.6294
2022-11-12 19:13:31,907 - mmdet - INFO - Epoch [2][900/962]	lr: 2.000e-02, eta: 2:24:46, time: 0.911, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0203, loss_rpn_bbox: 0.0133, s0.loss_cls: 0.1883, s0.acc: 92.9336, s0.loss_bbox: 0.1305, s1.loss_cls: 0.0863, s1.acc: 93.6562, s1.loss_bbox: 0.1061, s2.loss_cls: 0.0369, s2.acc: 94.3457, s2.loss_bbox: 0.0495, loss: 0.6312
2022-11-12 19:14:17,778 - mmdet - INFO - Epoch [2][950/962]	lr: 2.000e-02, eta: 2:24:06, time: 0.917, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0181, loss_rpn_bbox: 0.0138, s0.loss_cls: 0.1900, s0.acc: 93.1211, s0.loss_bbox: 0.1262, s1.loss_cls: 0.0864, s1.acc: 93.7168, s1.loss_bbox: 0.1069, s2.loss_cls: 0.0380, s2.acc: 94.5332, s2.loss_bbox: 0.0531, loss: 0.6326
2022-11-12 19:14:28,628 - mmdet - INFO - Saving checkpoint at 2 epochs
[>>] 200/200, 5.1 task/s, elapsed: 39s, ETA:     0s2022-11-12 19:15:14,210 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.55s).
Accumulating evaluation results...
DONE (t=0.10s).
2022-11-12 19:15:14,894 - mmdet - INFO -
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.056
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.152
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.044
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.056
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.161
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.161
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.161
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.161

2022-11-12 19:15:14,901 - mmdet - INFO - Exp name: customized_model_road.py
2022-11-12 19:15:14,901 - mmdet - INFO - Epoch(val) [2][200]	bbox_mAP: 0.0560, bbox_mAP_50: 0.1520, bbox_mAP_75: 0.0440, bbox_mAP_s: -1.0000, bbox_mAP_m: -1.0000, bbox_mAP_l: 0.0560, bbox_mAP_copypaste: 0.056 0.152 0.044 -1.000 -1.000 0.056
2022-11-12 19:16:02,485 - mmdet - INFO - Epoch [3][50/962]	lr: 2.000e-02, eta: 2:22:31, time: 0.951, data_time: 0.059, memory: 5062, loss_rpn_cls: 0.0195, loss_rpn_bbox: 0.0135, s0.loss_cls: 0.1813, s0.acc: 93.3242, s0.loss_bbox: 0.1190, s1.loss_cls: 0.0851, s1.acc: 93.8594, s1.loss_bbox: 0.1037, s2.loss_cls: 0.0362, s2.acc: 94.7891, s2.loss_bbox: 0.0502, loss: 0.6085
2022-11-12 19:16:47,787 - mmdet - INFO - Epoch [3][100/962]	lr: 2.000e-02, eta: 2:21:49, time: 0.906, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0152, loss_rpn_bbox: 0.0111, s0.loss_cls: 0.1993, s0.acc: 92.6934, s0.loss_bbox: 0.1326, s1.loss_cls: 0.0910, s1.acc: 93.3633, s1.loss_bbox: 0.1197, s2.loss_cls: 0.0388, s2.acc: 94.3047, s2.loss_bbox: 0.0583, loss: 0.6659
2022-11-12 19:17:32,890 - mmdet - INFO - Epoch [3][150/962]	lr: 2.000e-02, eta: 2:21:07, time: 0.902, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0148, loss_rpn_bbox: 0.0117, s0.loss_cls: 0.1994, s0.acc: 92.4727, s0.loss_bbox: 0.1416, s1.loss_cls: 0.0922, s1.acc: 93.1680, s1.loss_bbox: 0.1207, s2.loss_cls: 0.0403, s2.acc: 94.1797, s2.loss_bbox: 0.0586, loss: 0.6793
2022-11-12 19:18:18,452 - mmdet - INFO - Epoch [3][200/962]	lr: 2.000e-02, eta: 2:20:26, time: 0.911, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0137, loss_rpn_bbox: 0.0130, s0.loss_cls: 0.1661, s0.acc: 93.8203, s0.loss_bbox: 0.1196, s1.loss_cls: 0.0758, s1.acc: 94.3789, s1.loss_bbox: 0.1080, s2.loss_cls: 0.0327, s2.acc: 95.1270, s2.loss_bbox: 0.0541, loss: 0.5831
2022-11-12 19:19:03,646 - mmdet - INFO - Epoch [3][250/962]	lr: 2.000e-02, eta: 2:19:43, time: 0.904, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0126, loss_rpn_bbox: 0.0128, s0.loss_cls: 0.1709, s0.acc: 93.4551, s0.loss_bbox: 0.1174, s1.loss_cls: 0.0825, s1.acc: 93.7168, s1.loss_bbox: 0.1112, s2.loss_cls: 0.0347, s2.acc: 94.9336, s2.loss_bbox: 0.0530, loss: 0.5952
2022-11-12 19:19:48,468 - mmdet - INFO - Epoch [3][300/962]	lr: 2.000e-02, eta: 2:18:59, time: 0.896, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0165, loss_rpn_bbox: 0.0144, s0.loss_cls: 0.1815, s0.acc: 93.0684, s0.loss_bbox: 0.1307, s1.loss_cls: 0.0863, s1.acc: 93.3066, s1.loss_bbox: 0.1139, s2.loss_cls: 0.0373, s2.acc: 94.3359, s2.loss_bbox: 0.0554, loss: 0.6361
2022-11-12 19:20:33,455 - mmdet - INFO - Epoch [3][350/962]	lr: 2.000e-02, eta: 2:18:15, time: 0.900, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0152, loss_rpn_bbox: 0.0141, s0.loss_cls: 0.1839, s0.acc: 93.0234, s0.loss_bbox: 0.1243, s1.loss_cls: 0.0853, s1.acc: 93.6582, s1.loss_bbox: 0.1075, s2.loss_cls: 0.0367, s2.acc: 94.8516, s2.loss_bbox: 0.0513, loss: 0.6183
2022-11-12 19:21:19,056 - mmdet - INFO - Epoch [3][400/962]	lr: 2.000e-02, eta: 2:17:34, time: 0.912, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0139, loss_rpn_bbox: 0.0123, s0.loss_cls: 0.1872, s0.acc: 92.6562, s0.loss_bbox: 0.1332, s1.loss_cls: 0.0897, s1.acc: 93.1719, s1.loss_bbox: 0.1182, s2.loss_cls: 0.0396, s2.acc: 94.1523, s2.loss_bbox: 0.0587, loss: 0.6528
2022-11-12 19:22:04,412 - mmdet - INFO - Epoch [3][450/962]	lr: 2.000e-02, eta: 2:16:51, time: 0.907, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0130, loss_rpn_bbox: 0.0111, s0.loss_cls: 0.1627, s0.acc: 93.8750, s0.loss_bbox: 0.1160, s1.loss_cls: 0.0748, s1.acc: 94.2598, s1.loss_bbox: 0.1041, s2.loss_cls: 0.0335, s2.acc: 94.7832, s2.loss_bbox: 0.0542, loss: 0.5693
2022-11-12 19:22:49,548 - mmdet - INFO - Epoch [3][500/962]	lr: 2.000e-02, eta: 2:16:08, time: 0.903, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0117, loss_rpn_bbox: 0.0100, s0.loss_cls: 0.1592, s0.acc: 94.0098, s0.loss_bbox: 0.1077, s1.loss_cls: 0.0742, s1.acc: 94.3848, s1.loss_bbox: 0.0981, s2.loss_cls: 0.0326, s2.acc: 95.1348, s2.loss_bbox: 0.0528, loss: 0.5462
2022-11-12 19:23:34,393 - mmdet - INFO - Epoch [3][550/962]	lr: 2.000e-02, eta: 2:15:23, time: 0.897, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0158, loss_rpn_bbox: 0.0131, s0.loss_cls: 0.1895, s0.acc: 92.6680, s0.loss_bbox: 0.1344, s1.loss_cls: 0.0916, s1.acc: 93.0664, s1.loss_bbox: 0.1209, s2.loss_cls: 0.0397, s2.acc: 94.2930, s2.loss_bbox: 0.0584, loss: 0.6634
2022-11-12 19:24:19,521 - mmdet - INFO - Epoch [3][600/962]	lr: 2.000e-02, eta: 2:14:40, time: 0.903, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0117, loss_rpn_bbox: 0.0131, s0.loss_cls: 0.1645, s0.acc: 93.6816, s0.loss_bbox: 0.1118, s1.loss_cls: 0.0786, s1.acc: 94.0312, s1.loss_bbox: 0.0986, s2.loss_cls: 0.0344, s2.acc: 94.8535, s2.loss_bbox: 0.0515, loss: 0.5642
2022-11-12 19:25:05,335 - mmdet - INFO - Epoch [3][650/962]	lr: 2.000e-02, eta: 2:13:58, time: 0.916, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0166, loss_rpn_bbox: 0.0125, s0.loss_cls: 0.1842, s0.acc: 92.9785, s0.loss_bbox: 0.1262, s1.loss_cls: 0.0846, s1.acc: 93.5098, s1.loss_bbox: 0.1073, s2.loss_cls: 0.0373, s2.acc: 94.4141, s2.loss_bbox: 0.0549, loss: 0.6236
2022-11-12 19:25:50,628 - mmdet - INFO - Epoch [3][700/962]	lr: 2.000e-02, eta: 2:13:15, time: 0.906, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0171, loss_rpn_bbox: 0.0147, s0.loss_cls: 0.1738, s0.acc: 93.3926, s0.loss_bbox: 0.1244, s1.loss_cls: 0.0811, s1.acc: 93.7988, s1.loss_bbox: 0.1130, s2.loss_cls: 0.0366, s2.acc: 94.3711, s2.loss_bbox: 0.0571, loss: 0.6178
2022-11-12 19:26:35,914 - mmdet - INFO - Epoch [3][750/962]	lr: 2.000e-02, eta: 2:12:32, time: 0.906, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0140, loss_rpn_bbox: 0.0113, s0.loss_cls: 0.1759, s0.acc: 93.5859, s0.loss_bbox: 0.1213, s1.loss_cls: 0.0798, s1.acc: 94.3086, s1.loss_bbox: 0.1031, s2.loss_cls: 0.0339, s2.acc: 95.2871, s2.loss_bbox: 0.0518, loss: 0.5911
2022-11-12 19:27:20,698 - mmdet - INFO - Epoch [3][800/962]	lr: 2.000e-02, eta: 2:11:47, time: 0.896, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0136, loss_rpn_bbox: 0.0129, s0.loss_cls: 0.1722, s0.acc: 93.2910, s0.loss_bbox: 0.1170, s1.loss_cls: 0.0832, s1.acc: 93.5508, s1.loss_bbox: 0.1118, s2.loss_cls: 0.0378, s2.acc: 94.3066, s2.loss_bbox: 0.0592, loss: 0.6076
2022-11-12 19:28:05,794 - mmdet - INFO - Epoch [3][850/962]	lr: 2.000e-02, eta: 2:11:03, time: 0.902, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0118, loss_rpn_bbox: 0.0117, s0.loss_cls: 0.1749, s0.acc: 93.2695, s0.loss_bbox: 0.1222, s1.loss_cls: 0.0814, s1.acc: 93.7080, s1.loss_bbox: 0.1145, s2.loss_cls: 0.0366, s2.acc: 94.3586, s2.loss_bbox: 0.0612, loss: 0.6142
2022-11-12 19:28:51,502 - mmdet - INFO - Epoch [3][900/962]	lr: 2.000e-02, eta: 2:10:21, time: 0.914, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0148, loss_rpn_bbox: 0.0108, s0.loss_cls: 0.1826, s0.acc: 93.3047, s0.loss_bbox: 0.1268, s1.loss_cls: 0.0824, s1.acc: 93.9512, s1.loss_bbox: 0.1106, s2.loss_cls: 0.0364, s2.acc: 94.7461, s2.loss_bbox: 0.0552, loss: 0.6195
2022-11-12 19:29:36,856 - mmdet - INFO - Epoch [3][950/962]	lr: 2.000e-02, eta: 2:09:37, time: 0.907, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0145, loss_rpn_bbox: 0.0131, s0.loss_cls: 0.1627, s0.acc: 93.8867, s0.loss_bbox: 0.1109, s1.loss_cls: 0.0757, s1.acc: 94.3203, s1.loss_bbox: 0.1000, s2.loss_cls: 0.0335, s2.acc: 95.0391, s2.loss_bbox: 0.0525, loss: 0.5629
2022-11-12 19:29:47,760 - mmdet - INFO - Saving checkpoint at 3 epochs
[>>] 200/200, 5.1 task/s, elapsed: 39s, ETA:     0s2022-11-12 19:30:33,341 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.43s).
Accumulating evaluation results...
DONE (t=0.07s).
2022-11-12 19:30:33,864 - mmdet - INFO -
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.093
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.252
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.055
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.093
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.201
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.201
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.201
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.201

2022-11-12 19:30:33,869 - mmdet - INFO - Exp name: customized_model_road.py
2022-11-12 19:30:33,869 - mmdet - INFO - Epoch(val) [3][200]	bbox_mAP: 0.0930, bbox_mAP_50: 0.2520, bbox_mAP_75: 0.0550, bbox_mAP_s: -1.0000, bbox_mAP_m: -1.0000, bbox_mAP_l: 0.0930, bbox_mAP_copypaste: 0.093 0.252 0.055 -1.000 -1.000 0.093
2022-11-12 19:31:21,588 - mmdet - INFO - Epoch [4][50/962]	lr: 2.000e-02, eta: 2:08:19, time: 0.954, data_time: 0.059, memory: 5062, loss_rpn_cls: 0.0116, loss_rpn_bbox: 0.0102, s0.loss_cls: 0.1565, s0.acc: 93.9258, s0.loss_bbox: 0.1080, s1.loss_cls: 0.0741, s1.acc: 94.2832, s1.loss_bbox: 0.0979, s2.loss_cls: 0.0337, s2.acc: 94.7695, s2.loss_bbox: 0.0539, loss: 0.5460
2022-11-12 19:32:07,326 - mmdet - INFO - Epoch [4][100/962]	lr: 2.000e-02, eta: 2:07:37, time: 0.915, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0185, loss_rpn_bbox: 0.0114, s0.loss_cls: 0.1684, s0.acc: 93.6016, s0.loss_bbox: 0.1125, s1.loss_cls: 0.0774, s1.acc: 94.2129, s1.loss_bbox: 0.1016, s2.loss_cls: 0.0340, s2.acc: 94.8457, s2.loss_bbox: 0.0560, loss: 0.5798
2022-11-12 19:32:52,434 - mmdet - INFO - Epoch [4][150/962]	lr: 2.000e-02, eta: 2:06:53, time: 0.902, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0157, loss_rpn_bbox: 0.0121, s0.loss_cls: 0.1738, s0.acc: 93.5684, s0.loss_bbox: 0.1217, s1.loss_cls: 0.0812, s1.acc: 94.1406, s1.loss_bbox: 0.1135, s2.loss_cls: 0.0367, s2.acc: 94.5898, s2.loss_bbox: 0.0609, loss: 0.6156
2022-11-12 19:33:37,554 - mmdet - INFO - Epoch [4][200/962]	lr: 2.000e-02, eta: 2:06:09, time: 0.902, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0137, loss_rpn_bbox: 0.0132, s0.loss_cls: 0.1663, s0.acc: 93.6973, s0.loss_bbox: 0.1272, s1.loss_cls: 0.0753, s1.acc: 94.2285, s1.loss_bbox: 0.1103, s2.loss_cls: 0.0339, s2.acc: 94.9238, s2.loss_bbox: 0.0562, loss: 0.5961
2022-11-12 19:34:22,825 - mmdet - INFO - Epoch [4][250/962]	lr: 2.000e-02, eta: 2:05:26, time: 0.905, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0131, loss_rpn_bbox: 0.0122, s0.loss_cls: 0.1579, s0.acc: 93.9434, s0.loss_bbox: 0.1141, s1.loss_cls: 0.0748, s1.acc: 94.2051, s1.loss_bbox: 0.1117, s2.loss_cls: 0.0341, s2.acc: 94.7539, s2.loss_bbox: 0.0608, loss: 0.5786
2022-11-12 19:35:08,149 - mmdet - INFO - Epoch [4][300/962]	lr: 2.000e-02, eta: 2:04:43, time: 0.906, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0136, loss_rpn_bbox: 0.0102, s0.loss_cls: 0.1520, s0.acc: 94.1562, s0.loss_bbox: 0.1039, s1.loss_cls: 0.0757, s1.acc: 94.1953, s1.loss_bbox: 0.0954, s2.loss_cls: 0.0347, s2.acc: 94.7051, s2.loss_bbox: 0.0503, loss: 0.5358
2022-11-12 19:35:53,492 - mmdet - INFO - Epoch [4][350/962]	lr: 2.000e-02, eta: 2:03:59, time: 0.907, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0122, loss_rpn_bbox: 0.0115, s0.loss_cls: 0.1676, s0.acc: 93.5137, s0.loss_bbox: 0.1170, s1.loss_cls: 0.0801, s1.acc: 93.9238, s1.loss_bbox: 0.1072, s2.loss_cls: 0.0348, s2.acc: 94.6973, s2.loss_bbox: 0.0564, loss: 0.5867
2022-11-12 19:36:38,768 - mmdet - INFO - Epoch [4][400/962]	lr: 2.000e-02, eta: 2:03:16, time: 0.906, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0112, loss_rpn_bbox: 0.0106, s0.loss_cls: 0.1652, s0.acc: 93.7441, s0.loss_bbox: 0.1165, s1.loss_cls: 0.0766, s1.acc: 94.1504, s1.loss_bbox: 0.1064, s2.loss_cls: 0.0334, s2.acc: 94.8379, s2.loss_bbox: 0.0553, loss: 0.5753
2022-11-12 19:37:24,066 - mmdet - INFO - Epoch [4][450/962]	lr: 2.000e-02, eta: 2:02:32, time: 0.906, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0140, loss_rpn_bbox: 0.0101, s0.loss_cls: 0.1559, s0.acc: 94.1211, s0.loss_bbox: 0.1106, s1.loss_cls: 0.0706, s1.acc: 94.8535, s1.loss_bbox: 0.0947, s2.loss_cls: 0.0308, s2.acc: 95.6621, s2.loss_bbox: 0.0476, loss: 0.5344
2022-11-12 19:38:09,436 - mmdet - INFO - Epoch [4][500/962]	lr: 2.000e-02, eta: 2:01:49, time: 0.907, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0130, loss_rpn_bbox: 0.0111, s0.loss_cls: 0.1398, s0.acc: 94.7090, s0.loss_bbox: 0.0987, s1.loss_cls: 0.0671, s1.acc: 94.9141, s1.loss_bbox: 0.0939, s2.loss_cls: 0.0313, s2.acc: 95.2949, s2.loss_bbox: 0.0515, loss: 0.5066
2022-11-12 19:38:54,638 - mmdet - INFO - Epoch [4][550/962]	lr: 2.000e-02, eta: 2:01:05, time: 0.904, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0114, loss_rpn_bbox: 0.0110, s0.loss_cls: 0.1518, s0.acc: 94.3594, s0.loss_bbox: 0.1045, s1.loss_cls: 0.0706, s1.acc: 94.7871, s1.loss_bbox: 0.0958, s2.loss_cls: 0.0309, s2.acc: 95.4727, s2.loss_bbox: 0.0518, loss: 0.5278
2022-11-12 19:39:39,971 - mmdet - INFO - Epoch [4][600/962]	lr: 2.000e-02, eta: 2:00:22, time: 0.907, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0119, loss_rpn_bbox: 0.0113, s0.loss_cls: 0.1535, s0.acc: 94.1387, s0.loss_bbox: 0.1082, s1.loss_cls: 0.0723, s1.acc: 94.4023, s1.loss_bbox: 0.1040, s2.loss_cls: 0.0328, s2.acc: 95.0645, s2.loss_bbox: 0.0556, loss: 0.5496
2022-11-12 19:40:25,231 - mmdet - INFO - Epoch [4][650/962]	lr: 2.000e-02, eta: 1:59:38, time: 0.905, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0142, loss_rpn_bbox: 0.0100, s0.loss_cls: 0.1540, s0.acc: 94.1953, s0.loss_bbox: 0.1042, s1.loss_cls: 0.0691, s1.acc: 94.9062, s1.loss_bbox: 0.0962, s2.loss_cls: 0.0305, s2.acc: 95.5527, s2.loss_bbox: 0.0548, loss: 0.5330
2022-11-12 19:41:10,319 - mmdet - INFO - Epoch [4][700/962]	lr: 2.000e-02, eta: 1:58:54, time: 0.902, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0116, loss_rpn_bbox: 0.0103, s0.loss_cls: 0.1585, s0.acc: 93.6504, s0.loss_bbox: 0.1086, s1.loss_cls: 0.0761, s1.acc: 93.9492, s1.loss_bbox: 0.1007, s2.loss_cls: 0.0354, s2.acc: 94.4531, s2.loss_bbox: 0.0543, loss: 0.5555
2022-11-12 19:41:55,305 - mmdet - INFO - Epoch [4][750/962]	lr: 2.000e-02, eta: 1:58:09, time: 0.900, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0106, loss_rpn_bbox: 0.0114, s0.loss_cls: 0.1594, s0.acc: 94.0488, s0.loss_bbox: 0.1105, s1.loss_cls: 0.0721, s1.acc: 94.5781, s1.loss_bbox: 0.1023, s2.loss_cls: 0.0326, s2.acc: 95.1094, s2.loss_bbox: 0.0586, loss: 0.5576
2022-11-12 19:42:40,655 - mmdet - INFO - Epoch [4][800/962]	lr: 2.000e-02, eta: 1:57:25, time: 0.907, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0128, loss_rpn_bbox: 0.0107, s0.loss_cls: 0.1526, s0.acc: 94.2480, s0.loss_bbox: 0.1111, s1.loss_cls: 0.0718, s1.acc: 94.5742, s1.loss_bbox: 0.1030, s2.loss_cls: 0.0316, s2.acc: 95.2070, s2.loss_bbox: 0.0556, loss: 0.5492
2022-11-12 19:43:26,153 - mmdet - INFO - Epoch [4][850/962]	lr: 2.000e-02, eta: 1:56:42, time: 0.910, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0112, loss_rpn_bbox: 0.0120, s0.loss_cls: 0.1465, s0.acc: 94.5176, s0.loss_bbox: 0.0939, s1.loss_cls: 0.0683, s1.acc: 94.7969, s1.loss_bbox: 0.0912, s2.loss_cls: 0.0309, s2.acc: 95.3398, s2.loss_bbox: 0.0497, loss: 0.5037
2022-11-12 19:44:11,436 - mmdet - INFO - Epoch [4][900/962]	lr: 2.000e-02, eta: 1:55:58, time: 0.906, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0217, loss_rpn_bbox: 0.0141, s0.loss_cls: 0.1731, s0.acc: 93.7949, s0.loss_bbox: 0.1143, s1.loss_cls: 0.0798, s1.acc: 94.2988, s1.loss_bbox: 0.0968, s2.loss_cls: 0.0343, s2.acc: 95.0332, s2.loss_bbox: 0.0489, loss: 0.5830
2022-11-12 19:44:56,908 - mmdet - INFO - Epoch [4][950/962]	lr: 2.000e-02, eta: 1:55:14, time: 0.909, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0126, loss_rpn_bbox: 0.0113, s0.loss_cls: 0.1500, s0.acc: 94.4570, s0.loss_bbox: 0.1094, s1.loss_cls: 0.0700, s1.acc: 94.7891, s1.loss_bbox: 0.0981, s2.loss_cls: 0.0311, s2.acc: 95.1406, s2.loss_bbox: 0.0527, loss: 0.5352
2022-11-12 19:45:07,915 - mmdet - INFO - Saving checkpoint at 4 epochs
[>>] 200/200, 5.2 task/s, elapsed: 39s, ETA:     0s2022-11-12 19:45:53,162 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.77s).
Accumulating evaluation results...
DONE (t=0.13s).
2022-11-12 19:45:54,098 - mmdet - INFO -
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.089
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.226
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.061
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.089
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.213
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.213
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.213
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.213

2022-11-12 19:45:54,104 - mmdet - INFO - Exp name: customized_model_road.py
2022-11-12 19:45:54,104 - mmdet - INFO - Epoch(val) [4][200]	bbox_mAP: 0.0890, bbox_mAP_50: 0.2260, bbox_mAP_75: 0.0610, bbox_mAP_s: -1.0000, bbox_mAP_m: -1.0000, bbox_mAP_l: 0.0890, bbox_mAP_copypaste: 0.089 0.226 0.061 -1.000 -1.000 0.089
2022-11-12 19:46:42,121 - mmdet - INFO - Epoch [5][50/962]	lr: 2.000e-02, eta: 1:54:04, time: 0.960, data_time: 0.067, memory: 5062, loss_rpn_cls: 0.0115, loss_rpn_bbox: 0.0097, s0.loss_cls: 0.1459, s0.acc: 94.4277, s0.loss_bbox: 0.1019, s1.loss_cls: 0.0658, s1.acc: 94.9238, s1.loss_bbox: 0.0956, s2.loss_cls: 0.0300, s2.acc: 95.4590, s2.loss_bbox: 0.0549, loss: 0.5152
2022-11-12 19:47:27,442 - mmdet - INFO - Epoch [5][100/962]	lr: 2.000e-02, eta: 1:53:20, time: 0.906, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0155, loss_rpn_bbox: 0.0119, s0.loss_cls: 0.1409, s0.acc: 94.4043, s0.loss_bbox: 0.0951, s1.loss_cls: 0.0676, s1.acc: 94.5938, s1.loss_bbox: 0.0933, s2.loss_cls: 0.0307, s2.acc: 95.0879, s2.loss_bbox: 0.0532, loss: 0.5082
2022-11-12 19:48:12,878 - mmdet - INFO - Epoch [5][150/962]	lr: 2.000e-02, eta: 1:52:37, time: 0.909, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0111, loss_rpn_bbox: 0.0089, s0.loss_cls: 0.1372, s0.acc: 94.8164, s0.loss_bbox: 0.0907, s1.loss_cls: 0.0646, s1.acc: 94.9609, s1.loss_bbox: 0.0864, s2.loss_cls: 0.0303, s2.acc: 95.3613, s2.loss_bbox: 0.0539, loss: 0.4832
2022-11-12 19:48:58,109 - mmdet - INFO - Epoch [5][200/962]	lr: 2.000e-02, eta: 1:51:53, time: 0.905, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0121, loss_rpn_bbox: 0.0118, s0.loss_cls: 0.1591, s0.acc: 93.8477, s0.loss_bbox: 0.1082, s1.loss_cls: 0.0746, s1.acc: 94.2422, s1.loss_bbox: 0.1018, s2.loss_cls: 0.0333, s2.acc: 94.9766, s2.loss_bbox: 0.0533, loss: 0.5542
2022-11-12 19:49:43,494 - mmdet - INFO - Epoch [5][250/962]	lr: 2.000e-02, eta: 1:51:09, time: 0.908, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0156, loss_rpn_bbox: 0.0142, s0.loss_cls: 0.1441, s0.acc: 94.5215, s0.loss_bbox: 0.0984, s1.loss_cls: 0.0658, s1.acc: 95.0254, s1.loss_bbox: 0.0902, s2.loss_cls: 0.0291, s2.acc: 95.7598, s2.loss_bbox: 0.0507, loss: 0.5081
2022-11-12 19:50:28,853 - mmdet - INFO - Epoch [5][300/962]	lr: 2.000e-02, eta: 1:50:25, time: 0.907, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0107, loss_rpn_bbox: 0.0123, s0.loss_cls: 0.1626, s0.acc: 93.5859, s0.loss_bbox: 0.1145, s1.loss_cls: 0.0760, s1.acc: 94.2061, s1.loss_bbox: 0.1051, s2.loss_cls: 0.0350, s2.acc: 94.6476, s2.loss_bbox: 0.0602, loss: 0.5763
2022-11-12 19:51:14,275 - mmdet - INFO - Epoch [5][350/962]	lr: 2.000e-02, eta: 1:49:42, time: 0.908, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0142, loss_rpn_bbox: 0.0137, s0.loss_cls: 0.1536, s0.acc: 94.2402, s0.loss_bbox: 0.1055, s1.loss_cls: 0.0685, s1.acc: 94.7734, s1.loss_bbox: 0.0974, s2.loss_cls: 0.0309, s2.acc: 95.3066, s2.loss_bbox: 0.0540, loss: 0.5379
2022-11-12 19:51:59,529 - mmdet - INFO - Epoch [5][400/962]	lr: 2.000e-02, eta: 1:48:58, time: 0.905, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0123, loss_rpn_bbox: 0.0098, s0.loss_cls: 0.1607, s0.acc: 93.7266, s0.loss_bbox: 0.1093, s1.loss_cls: 0.0757, s1.acc: 93.9219, s1.loss_bbox: 0.1013, s2.loss_cls: 0.0350, s2.acc: 94.5410, s2.loss_bbox: 0.0576, loss: 0.5615
2022-11-12 19:52:44,792 - mmdet - INFO - Epoch [5][450/962]	lr: 2.000e-02, eta: 1:48:14, time: 0.905, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0099, loss_rpn_bbox: 0.0087, s0.loss_cls: 0.1466, s0.acc: 94.3359, s0.loss_bbox: 0.1031, s1.loss_cls: 0.0668, s1.acc: 94.8555, s1.loss_bbox: 0.0940, s2.loss_cls: 0.0308, s2.acc: 95.1367, s2.loss_bbox: 0.0510, loss: 0.5109
2022-11-12 19:53:29,982 - mmdet - INFO - Epoch [5][500/962]	lr: 2.000e-02, eta: 1:47:29, time: 0.904, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0095, loss_rpn_bbox: 0.0110, s0.loss_cls: 0.1538, s0.acc: 93.9824, s0.loss_bbox: 0.1069, s1.loss_cls: 0.0698, s1.acc: 94.5254, s1.loss_bbox: 0.1020, s2.loss_cls: 0.0319, s2.acc: 94.8965, s2.loss_bbox: 0.0571, loss: 0.5420
2022-11-12 19:54:15,179 - mmdet - INFO - Epoch [5][550/962]	lr: 2.000e-02, eta: 1:46:45, time: 0.904, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0091, loss_rpn_bbox: 0.0091, s0.loss_cls: 0.1390, s0.acc: 94.6133, s0.loss_bbox: 0.0967, s1.loss_cls: 0.0668, s1.acc: 95.0215, s1.loss_bbox: 0.0910, s2.loss_cls: 0.0321, s2.acc: 95.2305, s2.loss_bbox: 0.0524, loss: 0.4962
2022-11-12 19:55:00,330 - mmdet - INFO - Epoch [5][600/962]	lr: 2.000e-02, eta: 1:46:01, time: 0.903, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0180, loss_rpn_bbox: 0.0134, s0.loss_cls: 0.1585, s0.acc: 94.1523, s0.loss_bbox: 0.1087, s1.loss_cls: 0.0731, s1.acc: 94.6348, s1.loss_bbox: 0.1016, s2.loss_cls: 0.0331, s2.acc: 94.9766, s2.loss_bbox: 0.0542, loss: 0.5607
2022-11-12 19:55:45,539 - mmdet - INFO - Epoch [5][650/962]	lr: 2.000e-02, eta: 1:45:17, time: 0.904, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0122, loss_rpn_bbox: 0.0099, s0.loss_cls: 0.1283, s0.acc: 94.9805, s0.loss_bbox: 0.0855, s1.loss_cls: 0.0599, s1.acc: 95.4395, s1.loss_bbox: 0.0819, s2.loss_cls: 0.0280, s2.acc: 95.6348, s2.loss_bbox: 0.0479, loss: 0.4536
2022-11-12 19:56:30,691 - mmdet - INFO - Epoch [5][700/962]	lr: 2.000e-02, eta: 1:44:32, time: 0.903, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0138, loss_rpn_bbox: 0.0111, s0.loss_cls: 0.1555, s0.acc: 94.1055, s0.loss_bbox: 0.1105, s1.loss_cls: 0.0710, s1.acc: 94.5176, s1.loss_bbox: 0.1006, s2.loss_cls: 0.0325, s2.acc: 95.1797, s2.loss_bbox: 0.0523, loss: 0.5473
2022-11-12 19:57:15,908 - mmdet - INFO - Epoch [5][750/962]	lr: 2.000e-02, eta: 1:43:48, time: 0.904, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0092, loss_rpn_bbox: 0.0091, s0.loss_cls: 0.1423, s0.acc: 94.6230, s0.loss_bbox: 0.1026, s1.loss_cls: 0.0674, s1.acc: 94.8027, s1.loss_bbox: 0.0972, s2.loss_cls: 0.0308, s2.acc: 95.2910, s2.loss_bbox: 0.0547, loss: 0.5134
2022-11-12 19:58:00,966 - mmdet - INFO - Epoch [5][800/962]	lr: 2.000e-02, eta: 1:43:04, time: 0.901, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0133, loss_rpn_bbox: 0.0086, s0.loss_cls: 0.1433, s0.acc: 94.5176, s0.loss_bbox: 0.0989, s1.loss_cls: 0.0670, s1.acc: 95.0734, s1.loss_bbox: 0.0935, s2.loss_cls: 0.0300, s2.acc: 95.3744, s2.loss_bbox: 0.0519, loss: 0.5065
2022-11-12 19:58:46,109 - mmdet - INFO - Epoch [5][850/962]	lr: 2.000e-02, eta: 1:42:19, time: 0.903, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0121, loss_rpn_bbox: 0.0092, s0.loss_cls: 0.1288, s0.acc: 95.3613, s0.loss_bbox: 0.0881, s1.loss_cls: 0.0586, s1.acc: 95.7207, s1.loss_bbox: 0.0856, s2.loss_cls: 0.0273, s2.acc: 95.9023, s2.loss_bbox: 0.0503, loss: 0.4599
2022-11-12 19:59:31,265 - mmdet - INFO - Epoch [5][900/962]	lr: 2.000e-02, eta: 1:41:35, time: 0.903, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0116, loss_rpn_bbox: 0.0123, s0.loss_cls: 0.1469, s0.acc: 94.3984, s0.loss_bbox: 0.1015, s1.loss_cls: 0.0704, s1.acc: 94.5620, s1.loss_bbox: 0.0952, s2.loss_cls: 0.0315, s2.acc: 95.1792, s2.loss_bbox: 0.0518, loss: 0.5211
2022-11-12 20:00:16,257 - mmdet - INFO - Epoch [5][950/962]	lr: 2.000e-02, eta: 1:40:50, time: 0.900, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0118, loss_rpn_bbox: 0.0100, s0.loss_cls: 0.1491, s0.acc: 94.5879, s0.loss_bbox: 0.0991, s1.loss_cls: 0.0674, s1.acc: 95.0508, s1.loss_bbox: 0.0871, s2.loss_cls: 0.0299, s2.acc: 95.5605, s2.loss_bbox: 0.0484, loss: 0.5028
2022-11-12 20:00:27,085 - mmdet - INFO - Saving checkpoint at 5 epochs
[>>] 200/200, 5.1 task/s, elapsed: 39s, ETA:     0s2022-11-12 20:01:12,163 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.29s).
Accumulating evaluation results...
DONE (t=0.05s).
2022-11-12 20:01:12,514 - mmdet - INFO -
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.107
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.246
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.084
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.107
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.211
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.211
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.211
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.211

2022-11-12 20:01:12,517 - mmdet - INFO - Exp name: customized_model_road.py
2022-11-12 20:01:12,517 - mmdet - INFO - Epoch(val) [5][200]	bbox_mAP: 0.1070, bbox_mAP_50: 0.2460, bbox_mAP_75: 0.0840, bbox_mAP_s: -1.0000, bbox_mAP_m: -1.0000, bbox_mAP_l: 0.1070, bbox_mAP_copypaste: 0.107 0.246 0.084 -1.000 -1.000 0.107
2022-11-12 20:02:00,206 - mmdet - INFO - Epoch [6][50/962]	lr: 2.000e-02, eta: 1:39:44, time: 0.953, data_time: 0.059, memory: 5062, loss_rpn_cls: 0.0113, loss_rpn_bbox: 0.0103, s0.loss_cls: 0.1431, s0.acc: 94.7324, s0.loss_bbox: 0.0992, s1.loss_cls: 0.0634, s1.acc: 95.3084, s1.loss_bbox: 0.0933, s2.loss_cls: 0.0294, s2.acc: 95.7068, s2.loss_bbox: 0.0526, loss: 0.5025
2022-11-12 20:02:45,228 - mmdet - INFO - Epoch [6][100/962]	lr: 2.000e-02, eta: 1:38:59, time: 0.900, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0103, loss_rpn_bbox: 0.0087, s0.loss_cls: 0.1394, s0.acc: 94.5723, s0.loss_bbox: 0.0987, s1.loss_cls: 0.0669, s1.acc: 94.7402, s1.loss_bbox: 0.0948, s2.loss_cls: 0.0312, s2.acc: 94.9844, s2.loss_bbox: 0.0546, loss: 0.5045
2022-11-12 20:03:30,440 - mmdet - INFO - Epoch [6][150/962]	lr: 2.000e-02, eta: 1:38:15, time: 0.904, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0095, loss_rpn_bbox: 0.0111, s0.loss_cls: 0.1402, s0.acc: 94.7402, s0.loss_bbox: 0.0997, s1.loss_cls: 0.0643, s1.acc: 95.0898, s1.loss_bbox: 0.0975, s2.loss_cls: 0.0309, s2.acc: 95.1992, s2.loss_bbox: 0.0574, loss: 0.5106
2022-11-12 20:04:15,505 - mmdet - INFO - Epoch [6][200/962]	lr: 2.000e-02, eta: 1:37:31, time: 0.901, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0107, loss_rpn_bbox: 0.0103, s0.loss_cls: 0.1299, s0.acc: 95.1562, s0.loss_bbox: 0.0881, s1.loss_cls: 0.0598, s1.acc: 95.7500, s1.loss_bbox: 0.0814, s2.loss_cls: 0.0269, s2.acc: 96.0820, s2.loss_bbox: 0.0485, loss: 0.4557
2022-11-12 20:05:00,673 - mmdet - INFO - Epoch [6][250/962]	lr: 2.000e-02, eta: 1:36:46, time: 0.903, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0106, loss_rpn_bbox: 0.0105, s0.loss_cls: 0.1362, s0.acc: 94.9355, s0.loss_bbox: 0.0912, s1.loss_cls: 0.0629, s1.acc: 95.2266, s1.loss_bbox: 0.0866, s2.loss_cls: 0.0293, s2.acc: 95.5059, s2.loss_bbox: 0.0506, loss: 0.4778
2022-11-12 20:05:45,892 - mmdet - INFO - Epoch [6][300/962]	lr: 2.000e-02, eta: 1:36:02, time: 0.904, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0102, loss_rpn_bbox: 0.0159, s0.loss_cls: 0.1471, s0.acc: 94.3887, s0.loss_bbox: 0.1082, s1.loss_cls: 0.0683, s1.acc: 94.7694, s1.loss_bbox: 0.1029, s2.loss_cls: 0.0314, s2.acc: 95.1911, s2.loss_bbox: 0.0577, loss: 0.5418
2022-11-12 20:06:30,948 - mmdet - INFO - Epoch [6][350/962]	lr: 2.000e-02, eta: 1:35:18, time: 0.901, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0118, loss_rpn_bbox: 0.0080, s0.loss_cls: 0.1196, s0.acc: 95.4082, s0.loss_bbox: 0.0826, s1.loss_cls: 0.0566, s1.acc: 95.3944, s1.loss_bbox: 0.0858, s2.loss_cls: 0.0268, s2.acc: 95.6385, s2.loss_bbox: 0.0524, loss: 0.4435
2022-11-12 20:07:16,079 - mmdet - INFO - Epoch [6][400/962]	lr: 2.000e-02, eta: 1:34:33, time: 0.903, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0097, loss_rpn_bbox: 0.0080, s0.loss_cls: 0.1206, s0.acc: 95.2207, s0.loss_bbox: 0.0845, s1.loss_cls: 0.0555, s1.acc: 95.7520, s1.loss_bbox: 0.0844, s2.loss_cls: 0.0251, s2.acc: 95.9941, s2.loss_bbox: 0.0496, loss: 0.4373
2022-11-12 20:08:01,076 - mmdet - INFO - Epoch [6][450/962]	lr: 2.000e-02, eta: 1:33:49, time: 0.900, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0102, loss_rpn_bbox: 0.0084, s0.loss_cls: 0.1302, s0.acc: 94.8535, s0.loss_bbox: 0.0884, s1.loss_cls: 0.0628, s1.acc: 95.0735, s1.loss_bbox: 0.0868, s2.loss_cls: 0.0288, s2.acc: 95.5735, s2.loss_bbox: 0.0492, loss: 0.4648
2022-11-12 20:08:46,177 - mmdet - INFO - Epoch [6][500/962]	lr: 2.000e-02, eta: 1:33:04, time: 0.902, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0093, loss_rpn_bbox: 0.0108, s0.loss_cls: 0.1371, s0.acc: 94.7988, s0.loss_bbox: 0.1013, s1.loss_cls: 0.0623, s1.acc: 95.3105, s1.loss_bbox: 0.0953, s2.loss_cls: 0.0293, s2.acc: 95.6328, s2.loss_bbox: 0.0575, loss: 0.5029
2022-11-12 20:09:31,254 - mmdet - INFO - Epoch [6][550/962]	lr: 2.000e-02, eta: 1:32:20, time: 0.902, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0084, loss_rpn_bbox: 0.0086, s0.loss_cls: 0.1241, s0.acc: 95.2266, s0.loss_bbox: 0.0840, s1.loss_cls: 0.0595, s1.acc: 95.4893, s1.loss_bbox: 0.0851, s2.loss_cls: 0.0281, s2.acc: 95.6300, s2.loss_bbox: 0.0507, loss: 0.4485
2022-11-12 20:10:16,348 - mmdet - INFO - Epoch [6][600/962]	lr: 2.000e-02, eta: 1:31:35, time: 0.902, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0117, loss_rpn_bbox: 0.0095, s0.loss_cls: 0.1407, s0.acc: 94.3691, s0.loss_bbox: 0.1035, s1.loss_cls: 0.0674, s1.acc: 94.7246, s1.loss_bbox: 0.0977, s2.loss_cls: 0.0324, s2.acc: 95.0645, s2.loss_bbox: 0.0539, loss: 0.5167
2022-11-12 20:11:01,442 - mmdet - INFO - Epoch [6][650/962]	lr: 2.000e-02, eta: 1:30:51, time: 0.902, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0120, loss_rpn_bbox: 0.0100, s0.loss_cls: 0.1442, s0.acc: 94.5781, s0.loss_bbox: 0.0997, s1.loss_cls: 0.0652, s1.acc: 95.0879, s1.loss_bbox: 0.0924, s2.loss_cls: 0.0302, s2.acc: 95.4238, s2.loss_bbox: 0.0529, loss: 0.5066
2022-11-12 20:11:46,521 - mmdet - INFO - Epoch [6][700/962]	lr: 2.000e-02, eta: 1:30:06, time: 0.902, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0109, loss_rpn_bbox: 0.0096, s0.loss_cls: 0.1363, s0.acc: 94.8633, s0.loss_bbox: 0.0890, s1.loss_cls: 0.0635, s1.acc: 95.2196, s1.loss_bbox: 0.0840, s2.loss_cls: 0.0292, s2.acc: 95.6040, s2.loss_bbox: 0.0485, loss: 0.4711
2022-11-12 20:12:31,623 - mmdet - INFO - Epoch [6][750/962]	lr: 2.000e-02, eta: 1:29:22, time: 0.902, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0135, loss_rpn_bbox: 0.0128, s0.loss_cls: 0.1430, s0.acc: 94.6836, s0.loss_bbox: 0.0932, s1.loss_cls: 0.0657, s1.acc: 95.2285, s1.loss_bbox: 0.0900, s2.loss_cls: 0.0283, s2.acc: 95.7910, s2.loss_bbox: 0.0513, loss: 0.4978
2022-11-12 20:13:16,760 - mmdet - INFO - Epoch [6][800/962]	lr: 2.000e-02, eta: 1:28:38, time: 0.903, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0147, loss_rpn_bbox: 0.0116, s0.loss_cls: 0.1380, s0.acc: 94.8418, s0.loss_bbox: 0.0963, s1.loss_cls: 0.0616, s1.acc: 95.3379, s1.loss_bbox: 0.0939, s2.loss_cls: 0.0284, s2.acc: 95.7344, s2.loss_bbox: 0.0554, loss: 0.5000
2022-11-12 20:14:01,981 - mmdet - INFO - Epoch [6][850/962]	lr: 2.000e-02, eta: 1:27:53, time: 0.904, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0141, loss_rpn_bbox: 0.0107, s0.loss_cls: 0.1344, s0.acc: 94.8984, s0.loss_bbox: 0.0961, s1.loss_cls: 0.0604, s1.acc: 95.4512, s1.loss_bbox: 0.0901, s2.loss_cls: 0.0279, s2.acc: 95.7402, s2.loss_bbox: 0.0545, loss: 0.4882
2022-11-12 20:14:46,971 - mmdet - INFO - Epoch [6][900/962]	lr: 2.000e-02, eta: 1:27:08, time: 0.900, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0101, loss_rpn_bbox: 0.0072, s0.loss_cls: 0.1261, s0.acc: 95.4043, s0.loss_bbox: 0.0776, s1.loss_cls: 0.0608, s1.acc: 95.5449, s1.loss_bbox: 0.0772, s2.loss_cls: 0.0283, s2.acc: 95.7617, s2.loss_bbox: 0.0451, loss: 0.4325
2022-11-12 20:15:31,999 - mmdet - INFO - Epoch [6][950/962]	lr: 2.000e-02, eta: 1:26:24, time: 0.901, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0114, loss_rpn_bbox: 0.0099, s0.loss_cls: 0.1314, s0.acc: 95.0371, s0.loss_bbox: 0.0875, s1.loss_cls: 0.0603, s1.acc: 95.3729, s1.loss_bbox: 0.0835, s2.loss_cls: 0.0282, s2.acc: 95.6580, s2.loss_bbox: 0.0515, loss: 0.4637
2022-11-12 20:15:42,807 - mmdet - INFO - Saving checkpoint at 6 epochs
[>>] 200/200, 5.1 task/s, elapsed: 39s, ETA:     0s2022-11-12 20:16:28,011 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.38s).
Accumulating evaluation results...
DONE (t=0.06s).
2022-11-12 20:16:28,587 - mmdet - INFO -
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.092
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.259
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.041
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.092
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.199
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.199
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.199
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.199

2022-11-12 20:16:28,591 - mmdet - INFO - Exp name: customized_model_road.py
2022-11-12 20:16:28,591 - mmdet - INFO - Epoch(val) [6][200]	bbox_mAP: 0.0920, bbox_mAP_50: 0.2590, bbox_mAP_75: 0.0410, bbox_mAP_s: -1.0000, bbox_mAP_m: -1.0000, bbox_mAP_l: 0.0920, bbox_mAP_copypaste: 0.092 0.259 0.041 -1.000 -1.000 0.092
2022-11-12 20:17:16,258 - mmdet - INFO - Epoch [7][50/962]	lr: 2.000e-02, eta: 1:25:21, time: 0.953, data_time: 0.059, memory: 5062, loss_rpn_cls: 0.0137, loss_rpn_bbox: 0.0085, s0.loss_cls: 0.1261, s0.acc: 95.2852, s0.loss_bbox: 0.0866, s1.loss_cls: 0.0568, s1.acc: 95.7266, s1.loss_bbox: 0.0860, s2.loss_cls: 0.0265, s2.acc: 95.9199, s2.loss_bbox: 0.0508, loss: 0.4549
2022-11-12 20:18:01,488 - mmdet - INFO - Epoch [7][100/962]	lr: 2.000e-02, eta: 1:24:36, time: 0.905, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0122, loss_rpn_bbox: 0.0131, s0.loss_cls: 0.1306, s0.acc: 95.0508, s0.loss_bbox: 0.0862, s1.loss_cls: 0.0577, s1.acc: 95.4872, s1.loss_bbox: 0.0873, s2.loss_cls: 0.0274, s2.acc: 95.6550, s2.loss_bbox: 0.0545, loss: 0.4688
2022-11-12 20:18:46,574 - mmdet - INFO - Epoch [7][150/962]	lr: 2.000e-02, eta: 1:23:52, time: 0.902, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0112, loss_rpn_bbox: 0.0114, s0.loss_cls: 0.1269, s0.acc: 95.1484, s0.loss_bbox: 0.0928, s1.loss_cls: 0.0597, s1.acc: 95.3594, s1.loss_bbox: 0.0928, s2.loss_cls: 0.0277, s2.acc: 95.5859, s2.loss_bbox: 0.0564, loss: 0.4789
2022-11-12 20:19:31,734 - mmdet - INFO - Epoch [7][200/962]	lr: 2.000e-02, eta: 1:23:07, time: 0.903, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0122, loss_rpn_bbox: 0.0110, s0.loss_cls: 0.1266, s0.acc: 95.0977, s0.loss_bbox: 0.0917, s1.loss_cls: 0.0579, s1.acc: 95.3809, s1.loss_bbox: 0.0915, s2.loss_cls: 0.0282, s2.acc: 95.5605, s2.loss_bbox: 0.0565, loss: 0.4757
2022-11-12 20:20:16,967 - mmdet - INFO - Epoch [7][250/962]	lr: 2.000e-02, eta: 1:22:23, time: 0.905, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0098, loss_rpn_bbox: 0.0097, s0.loss_cls: 0.1241, s0.acc: 95.2285, s0.loss_bbox: 0.0885, s1.loss_cls: 0.0584, s1.acc: 95.4381, s1.loss_bbox: 0.0926, s2.loss_cls: 0.0272, s2.acc: 95.8701, s2.loss_bbox: 0.0543, loss: 0.4647
2022-11-12 20:21:02,281 - mmdet - INFO - Epoch [7][300/962]	lr: 2.000e-02, eta: 1:21:39, time: 0.906, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0102, loss_rpn_bbox: 0.0133, s0.loss_cls: 0.1359, s0.acc: 94.9023, s0.loss_bbox: 0.0971, s1.loss_cls: 0.0625, s1.acc: 95.1719, s1.loss_bbox: 0.0957, s2.loss_cls: 0.0296, s2.acc: 95.3926, s2.loss_bbox: 0.0560, loss: 0.5002
2022-11-12 20:21:47,401 - mmdet - INFO - Epoch [7][350/962]	lr: 2.000e-02, eta: 1:20:54, time: 0.902, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0103, loss_rpn_bbox: 0.0086, s0.loss_cls: 0.1218, s0.acc: 95.1680, s0.loss_bbox: 0.0857, s1.loss_cls: 0.0588, s1.acc: 95.4473, s1.loss_bbox: 0.0885, s2.loss_cls: 0.0274, s2.acc: 95.7754, s2.loss_bbox: 0.0540, loss: 0.4551
2022-11-12 20:22:32,617 - mmdet - INFO - Epoch [7][400/962]	lr: 2.000e-02, eta: 1:20:10, time: 0.904, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0086, loss_rpn_bbox: 0.0076, s0.loss_cls: 0.1257, s0.acc: 95.1953, s0.loss_bbox: 0.0883, s1.loss_cls: 0.0585, s1.acc: 95.4281, s1.loss_bbox: 0.0869, s2.loss_cls: 0.0284, s2.acc: 95.6155, s2.loss_bbox: 0.0538, loss: 0.4578
2022-11-12 20:23:17,521 - mmdet - INFO - Epoch [7][450/962]	lr: 2.000e-02, eta: 1:19:25, time: 0.898, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0094, loss_rpn_bbox: 0.0079, s0.loss_cls: 0.1164, s0.acc: 95.5723, s0.loss_bbox: 0.0808, s1.loss_cls: 0.0532, s1.acc: 95.9557, s1.loss_bbox: 0.0767, s2.loss_cls: 0.0251, s2.acc: 95.9656, s2.loss_bbox: 0.0465, loss: 0.4160
2022-11-12 20:24:02,628 - mmdet - INFO - Epoch [7][500/962]	lr: 2.000e-02, eta: 1:18:41, time: 0.902, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0126, loss_rpn_bbox: 0.0105, s0.loss_cls: 0.1413, s0.acc: 94.7227, s0.loss_bbox: 0.0939, s1.loss_cls: 0.0657, s1.acc: 95.0918, s1.loss_bbox: 0.0928, s2.loss_cls: 0.0298, s2.acc: 95.5977, s2.loss_bbox: 0.0543, loss: 0.5009
2022-11-12 20:24:47,806 - mmdet - INFO - Epoch [7][550/962]	lr: 2.000e-02, eta: 1:17:56, time: 0.904, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0114, loss_rpn_bbox: 0.0096, s0.loss_cls: 0.1227, s0.acc: 95.0176, s0.loss_bbox: 0.0875, s1.loss_cls: 0.0575, s1.acc: 95.4961, s1.loss_bbox: 0.0833, s2.loss_cls: 0.0280, s2.acc: 95.6562, s2.loss_bbox: 0.0466, loss: 0.4468
2022-11-12 20:25:32,887 - mmdet - INFO - Epoch [7][600/962]	lr: 2.000e-02, eta: 1:17:12, time: 0.902, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0104, loss_rpn_bbox: 0.0090, s0.loss_cls: 0.1182, s0.acc: 95.6152, s0.loss_bbox: 0.0800, s1.loss_cls: 0.0533, s1.acc: 96.1074, s1.loss_bbox: 0.0788, s2.loss_cls: 0.0251, s2.acc: 96.3613, s2.loss_bbox: 0.0493, loss: 0.4240
2022-11-12 20:26:17,941 - mmdet - INFO - Epoch [7][650/962]	lr: 2.000e-02, eta: 1:16:27, time: 0.901, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0119, loss_rpn_bbox: 0.0108, s0.loss_cls: 0.1499, s0.acc: 94.1074, s0.loss_bbox: 0.1057, s1.loss_cls: 0.0683, s1.acc: 94.7305, s1.loss_bbox: 0.0933, s2.loss_cls: 0.0319, s2.acc: 95.0176, s2.loss_bbox: 0.0549, loss: 0.5267
2022-11-12 20:27:03,080 - mmdet - INFO - Epoch [7][700/962]	lr: 2.000e-02, eta: 1:15:43, time: 0.903, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0127, loss_rpn_bbox: 0.0117, s0.loss_cls: 0.1375, s0.acc: 94.7969, s0.loss_bbox: 0.0944, s1.loss_cls: 0.0652, s1.acc: 95.0137, s1.loss_bbox: 0.0911, s2.loss_cls: 0.0307, s2.acc: 95.3340, s2.loss_bbox: 0.0548, loss: 0.4981
2022-11-12 20:27:48,262 - mmdet - INFO - Epoch [7][750/962]	lr: 2.000e-02, eta: 1:14:58, time: 0.904, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0090, loss_rpn_bbox: 0.0082, s0.loss_cls: 0.1146, s0.acc: 95.7090, s0.loss_bbox: 0.0859, s1.loss_cls: 0.0539, s1.acc: 95.9316, s1.loss_bbox: 0.0829, s2.loss_cls: 0.0248, s2.acc: 96.1797, s2.loss_bbox: 0.0495, loss: 0.4287
2022-11-12 20:28:33,363 - mmdet - INFO - Epoch [7][800/962]	lr: 2.000e-02, eta: 1:14:14, time: 0.902, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0084, loss_rpn_bbox: 0.0083, s0.loss_cls: 0.1276, s0.acc: 95.1973, s0.loss_bbox: 0.0870, s1.loss_cls: 0.0582, s1.acc: 95.6793, s1.loss_bbox: 0.0849, s2.loss_cls: 0.0271, s2.acc: 95.9840, s2.loss_bbox: 0.0525, loss: 0.4539
2022-11-12 20:29:18,488 - mmdet - INFO - Epoch [7][850/962]	lr: 2.000e-02, eta: 1:13:29, time: 0.903, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0117, loss_rpn_bbox: 0.0088, s0.loss_cls: 0.1322, s0.acc: 94.7871, s0.loss_bbox: 0.0923, s1.loss_cls: 0.0619, s1.acc: 95.0193, s1.loss_bbox: 0.0921, s2.loss_cls: 0.0296, s2.acc: 95.1600, s2.loss_bbox: 0.0548, loss: 0.4833
2022-11-12 20:30:03,569 - mmdet - INFO - Epoch [7][900/962]	lr: 2.000e-02, eta: 1:12:45, time: 0.902, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0102, loss_rpn_bbox: 0.0095, s0.loss_cls: 0.1272, s0.acc: 95.1543, s0.loss_bbox: 0.0907, s1.loss_cls: 0.0607, s1.acc: 95.5097, s1.loss_bbox: 0.0875, s2.loss_cls: 0.0295, s2.acc: 95.6854, s2.loss_bbox: 0.0518, loss: 0.4670
2022-11-12 20:30:48,736 - mmdet - INFO - Epoch [7][950/962]	lr: 2.000e-02, eta: 1:12:00, time: 0.903, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0086, loss_rpn_bbox: 0.0088, s0.loss_cls: 0.1140, s0.acc: 95.7578, s0.loss_bbox: 0.0759, s1.loss_cls: 0.0502, s1.acc: 96.2051, s1.loss_bbox: 0.0775, s2.loss_cls: 0.0239, s2.acc: 96.4004, s2.loss_bbox: 0.0475, loss: 0.4064
2022-11-12 20:30:59,572 - mmdet - INFO - Saving checkpoint at 7 epochs
[>>] 200/200, 5.1 task/s, elapsed: 39s, ETA:     0s2022-11-12 20:31:44,672 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.26s).
Accumulating evaluation results...
DONE (t=0.05s).
2022-11-12 20:31:44,996 - mmdet - INFO -
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.111
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.270
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.081
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.111
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.214
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.214
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.214
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.214

2022-11-12 20:31:45,000 - mmdet - INFO - Exp name: customized_model_road.py
2022-11-12 20:31:45,000 - mmdet - INFO - Epoch(val) [7][200]	bbox_mAP: 0.1110, bbox_mAP_50: 0.2700, bbox_mAP_75: 0.0810, bbox_mAP_s: -1.0000, bbox_mAP_m: -1.0000, bbox_mAP_l: 0.1110, bbox_mAP_copypaste: 0.111 0.270 0.081 -1.000 -1.000 0.111
2022-11-12 20:32:32,745 - mmdet - INFO - Epoch [8][50/962]	lr: 2.000e-02, eta: 1:10:59, time: 0.954, data_time: 0.060, memory: 5062, loss_rpn_cls: 0.0083, loss_rpn_bbox: 0.0080, s0.loss_cls: 0.1017, s0.acc: 96.1660, s0.loss_bbox: 0.0728, s1.loss_cls: 0.0464, s1.acc: 96.4492, s1.loss_bbox: 0.0754, s2.loss_cls: 0.0219, s2.acc: 96.6250, s2.loss_bbox: 0.0464, loss: 0.3808
2022-11-12 20:33:18,519 - mmdet - INFO - Epoch [8][100/962]	lr: 2.000e-02, eta: 1:10:15, time: 0.915, data_time: 0.010, memory: 5062, loss_rpn_cls: 0.0096, loss_rpn_bbox: 0.0110, s0.loss_cls: 0.1127, s0.acc: 95.7227, s0.loss_bbox: 0.0791, s1.loss_cls: 0.0516, s1.acc: 96.1016, s1.loss_bbox: 0.0764, s2.loss_cls: 0.0245, s2.acc: 96.2422, s2.loss_bbox: 0.0469, loss: 0.4118
2022-11-12 20:34:03,588 - mmdet - INFO - Epoch [8][150/962]	lr: 2.000e-02, eta: 1:09:30, time: 0.901, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0103, loss_rpn_bbox: 0.0088, s0.loss_cls: 0.1220, s0.acc: 95.3184, s0.loss_bbox: 0.0911, s1.loss_cls: 0.0535, s1.acc: 95.9570, s1.loss_bbox: 0.0883, s2.loss_cls: 0.0249, s2.acc: 96.1348, s2.loss_bbox: 0.0545, loss: 0.4533
2022-11-12 20:34:48,696 - mmdet - INFO - Epoch [8][200/962]	lr: 2.000e-02, eta: 1:08:46, time: 0.902, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0100, loss_rpn_bbox: 0.0075, s0.loss_cls: 0.1319, s0.acc: 94.8770, s0.loss_bbox: 0.0863, s1.loss_cls: 0.0625, s1.acc: 95.1750, s1.loss_bbox: 0.0800, s2.loss_cls: 0.0294, s2.acc: 95.4661, s2.loss_bbox: 0.0485, loss: 0.4561
2022-11-12 20:35:33,679 - mmdet - INFO - Epoch [8][250/962]	lr: 2.000e-02, eta: 1:08:01, time: 0.900, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0100, loss_rpn_bbox: 0.0098, s0.loss_cls: 0.1128, s0.acc: 95.6895, s0.loss_bbox: 0.0815, s1.loss_cls: 0.0519, s1.acc: 96.1026, s1.loss_bbox: 0.0833, s2.loss_cls: 0.0250, s2.acc: 96.0986, s2.loss_bbox: 0.0542, loss: 0.4286
2022-11-12 20:36:18,781 - mmdet - INFO - Epoch [8][300/962]	lr: 2.000e-02, eta: 1:07:17, time: 0.902, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0114, loss_rpn_bbox: 0.0093, s0.loss_cls: 0.1231, s0.acc: 95.6641, s0.loss_bbox: 0.0793, s1.loss_cls: 0.0538, s1.acc: 96.2637, s1.loss_bbox: 0.0718, s2.loss_cls: 0.0257, s2.acc: 96.3555, s2.loss_bbox: 0.0455, loss: 0.4198
2022-11-12 20:37:03,891 - mmdet - INFO - Epoch [8][350/962]	lr: 2.000e-02, eta: 1:06:32, time: 0.902, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0087, loss_rpn_bbox: 0.0070, s0.loss_cls: 0.1143, s0.acc: 95.6523, s0.loss_bbox: 0.0760, s1.loss_cls: 0.0524, s1.acc: 95.9785, s1.loss_bbox: 0.0757, s2.loss_cls: 0.0255, s2.acc: 95.8672, s2.loss_bbox: 0.0481, loss: 0.4078
2022-11-12 20:37:48,862 - mmdet - INFO - Epoch [8][400/962]	lr: 2.000e-02, eta: 1:05:48, time: 0.899, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0081, loss_rpn_bbox: 0.0089, s0.loss_cls: 0.1229, s0.acc: 95.2305, s0.loss_bbox: 0.0818, s1.loss_cls: 0.0573, s1.acc: 95.2769, s1.loss_bbox: 0.0818, s2.loss_cls: 0.0277, s2.acc: 95.5111, s2.loss_bbox: 0.0491, loss: 0.4377
2022-11-12 20:38:33,783 - mmdet - INFO - Epoch [8][450/962]	lr: 2.000e-02, eta: 1:05:03, time: 0.898, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0105, loss_rpn_bbox: 0.0110, s0.loss_cls: 0.1301, s0.acc: 95.0391, s0.loss_bbox: 0.0892, s1.loss_cls: 0.0592, s1.acc: 95.4844, s1.loss_bbox: 0.0814, s2.loss_cls: 0.0283, s2.acc: 95.5893, s2.loss_bbox: 0.0493, loss: 0.4588
2022-11-12 20:39:18,736 - mmdet - INFO - Epoch [8][500/962]	lr: 2.000e-02, eta: 1:04:18, time: 0.899, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0100, loss_rpn_bbox: 0.0091, s0.loss_cls: 0.1265, s0.acc: 95.1660, s0.loss_bbox: 0.0807, s1.loss_cls: 0.0615, s1.acc: 95.3390, s1.loss_bbox: 0.0818, s2.loss_cls: 0.0288, s2.acc: 95.5738, s2.loss_bbox: 0.0496, loss: 0.4479
2022-11-12 20:40:03,668 - mmdet - INFO - Epoch [8][550/962]	lr: 2.000e-02, eta: 1:03:34, time: 0.899, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0099, loss_rpn_bbox: 0.0103, s0.loss_cls: 0.1224, s0.acc: 95.3203, s0.loss_bbox: 0.0898, s1.loss_cls: 0.0547, s1.acc: 95.8364, s1.loss_bbox: 0.0860, s2.loss_cls: 0.0260, s2.acc: 96.1156, s2.loss_bbox: 0.0526, loss: 0.4518
2022-11-12 20:40:48,713 - mmdet - INFO - Epoch [8][600/962]	lr: 2.000e-02, eta: 1:02:49, time: 0.901, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0095, loss_rpn_bbox: 0.0087, s0.loss_cls: 0.1104, s0.acc: 95.8184, s0.loss_bbox: 0.0790, s1.loss_cls: 0.0487, s1.acc: 96.2471, s1.loss_bbox: 0.0805, s2.loss_cls: 0.0225, s2.acc: 96.4212, s2.loss_bbox: 0.0505, loss: 0.4098
2022-11-12 20:41:33,726 - mmdet - INFO - Epoch [8][650/962]	lr: 2.000e-02, eta: 1:02:04, time: 0.900, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0105, loss_rpn_bbox: 0.0089, s0.loss_cls: 0.1218, s0.acc: 95.6758, s0.loss_bbox: 0.0830, s1.loss_cls: 0.0539, s1.acc: 96.1172, s1.loss_bbox: 0.0798, s2.loss_cls: 0.0260, s2.acc: 96.1328, s2.loss_bbox: 0.0491, loss: 0.4329
2022-11-12 20:42:18,686 - mmdet - INFO - Epoch [8][700/962]	lr: 2.000e-02, eta: 1:01:20, time: 0.899, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0091, loss_rpn_bbox: 0.0084, s0.loss_cls: 0.1006, s0.acc: 96.0840, s0.loss_bbox: 0.0714, s1.loss_cls: 0.0469, s1.acc: 96.3320, s1.loss_bbox: 0.0732, s2.loss_cls: 0.0231, s2.acc: 96.4219, s2.loss_bbox: 0.0470, loss: 0.3796
2022-11-12 20:43:03,435 - mmdet - INFO - Epoch [8][750/962]	lr: 2.000e-02, eta: 1:00:35, time: 0.895, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0070, loss_rpn_bbox: 0.0077, s0.loss_cls: 0.1037, s0.acc: 96.0078, s0.loss_bbox: 0.0759, s1.loss_cls: 0.0484, s1.acc: 96.2687, s1.loss_bbox: 0.0785, s2.loss_cls: 0.0246, s2.acc: 96.4367, s2.loss_bbox: 0.0492, loss: 0.3950
2022-11-12 20:43:48,325 - mmdet - INFO - Epoch [8][800/962]	lr: 2.000e-02, eta: 0:59:50, time: 0.898, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0101, loss_rpn_bbox: 0.0105, s0.loss_cls: 0.1312, s0.acc: 95.0508, s0.loss_bbox: 0.0912, s1.loss_cls: 0.0597, s1.acc: 95.5645, s1.loss_bbox: 0.0877, s2.loss_cls: 0.0268, s2.acc: 95.8945, s2.loss_bbox: 0.0505, loss: 0.4677
2022-11-12 20:44:33,533 - mmdet - INFO - Epoch [8][850/962]	lr: 2.000e-02, eta: 0:59:06, time: 0.904, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0122, loss_rpn_bbox: 0.0152, s0.loss_cls: 0.1381, s0.acc: 94.7363, s0.loss_bbox: 0.0872, s1.loss_cls: 0.0651, s1.acc: 94.9941, s1.loss_bbox: 0.0875, s2.loss_cls: 0.0309, s2.acc: 95.2715, s2.loss_bbox: 0.0546, loss: 0.4908
2022-11-12 20:45:18,408 - mmdet - INFO - Epoch [8][900/962]	lr: 2.000e-02, eta: 0:58:21, time: 0.898, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0103, loss_rpn_bbox: 0.0108, s0.loss_cls: 0.1145, s0.acc: 95.6895, s0.loss_bbox: 0.0804, s1.loss_cls: 0.0523, s1.acc: 95.8867, s1.loss_bbox: 0.0816, s2.loss_cls: 0.0249, s2.acc: 96.0820, s2.loss_bbox: 0.0494, loss: 0.4242
2022-11-12 20:46:03,402 - mmdet - INFO - Epoch [8][950/962]	lr: 2.000e-02, eta: 0:57:36, time: 0.900, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0105, loss_rpn_bbox: 0.0108, s0.loss_cls: 0.1242, s0.acc: 95.0078, s0.loss_bbox: 0.0872, s1.loss_cls: 0.0568, s1.acc: 95.5087, s1.loss_bbox: 0.0843, s2.loss_cls: 0.0271, s2.acc: 95.7217, s2.loss_bbox: 0.0515, loss: 0.4524
2022-11-12 20:46:14,441 - mmdet - INFO - Saving checkpoint at 8 epochs
[>>] 200/200, 5.1 task/s, elapsed: 39s, ETA:     0s2022-11-12 20:46:59,557 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.69s).
Accumulating evaluation results...
DONE (t=0.10s).
2022-11-12 20:47:00,365 - mmdet - INFO -
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.120
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.298
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.082
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.120
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.217
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.217
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.217
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.217

2022-11-12 20:47:00,371 - mmdet - INFO - Exp name: customized_model_road.py
2022-11-12 20:47:00,371 - mmdet - INFO - Epoch(val) [8][200]	bbox_mAP: 0.1200, bbox_mAP_50: 0.2980, bbox_mAP_75: 0.0820, bbox_mAP_s: -1.0000, bbox_mAP_m: -1.0000, bbox_mAP_l: 0.1200, bbox_mAP_copypaste: 0.120 0.298 0.082 -1.000 -1.000 0.120
2022-11-12 20:47:47,988 - mmdet - INFO - Epoch [9][50/962]	lr: 2.000e-03, eta: 0:56:37, time: 0.952, data_time: 0.059, memory: 5062, loss_rpn_cls: 0.0077, loss_rpn_bbox: 0.0079, s0.loss_cls: 0.1050, s0.acc: 95.8730, s0.loss_bbox: 0.0735, s1.loss_cls: 0.0458, s1.acc: 96.4219, s1.loss_bbox: 0.0765, s2.loss_cls: 0.0226, s2.acc: 96.3398, s2.loss_bbox: 0.0486, loss: 0.3876
2022-11-12 20:48:33,048 - mmdet - INFO - Epoch [9][100/962]	lr: 2.000e-03, eta: 0:55:52, time: 0.901, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0084, loss_rpn_bbox: 0.0074, s0.loss_cls: 0.0945, s0.acc: 96.3633, s0.loss_bbox: 0.0655, s1.loss_cls: 0.0385, s1.acc: 97.0030, s1.loss_bbox: 0.0612, s2.loss_cls: 0.0184, s2.acc: 97.2090, s2.loss_bbox: 0.0386, loss: 0.3325
2022-11-12 20:49:18,103 - mmdet - INFO - Epoch [9][150/962]	lr: 2.000e-03, eta: 0:55:08, time: 0.901, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0073, loss_rpn_bbox: 0.0071, s0.loss_cls: 0.0944, s0.acc: 96.5176, s0.loss_bbox: 0.0659, s1.loss_cls: 0.0416, s1.acc: 96.9316, s1.loss_bbox: 0.0637, s2.loss_cls: 0.0201, s2.acc: 96.9922, s2.loss_bbox: 0.0416, loss: 0.3418
2022-11-12 20:50:03,173 - mmdet - INFO - Epoch [9][200/962]	lr: 2.000e-03, eta: 0:54:23, time: 0.901, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0061, loss_rpn_bbox: 0.0077, s0.loss_cls: 0.0911, s0.acc: 96.5059, s0.loss_bbox: 0.0657, s1.loss_cls: 0.0405, s1.acc: 96.7891, s1.loss_bbox: 0.0663, s2.loss_cls: 0.0192, s2.acc: 97.0508, s2.loss_bbox: 0.0436, loss: 0.3401
2022-11-12 20:50:48,160 - mmdet - INFO - Epoch [9][250/962]	lr: 2.000e-03, eta: 0:53:39, time: 0.900, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0072, loss_rpn_bbox: 0.0067, s0.loss_cls: 0.0822, s0.acc: 96.9023, s0.loss_bbox: 0.0592, s1.loss_cls: 0.0355, s1.acc: 97.2314, s1.loss_bbox: 0.0595, s2.loss_cls: 0.0176, s2.acc: 97.3034, s2.loss_bbox: 0.0401, loss: 0.3079
2022-11-12 20:51:33,131 - mmdet - INFO - Epoch [9][300/962]	lr: 2.000e-03, eta: 0:52:54, time: 0.899, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0071, loss_rpn_bbox: 0.0075, s0.loss_cls: 0.0897, s0.acc: 96.5527, s0.loss_bbox: 0.0645, s1.loss_cls: 0.0391, s1.acc: 97.0431, s1.loss_bbox: 0.0642, s2.loss_cls: 0.0196, s2.acc: 97.0192, s2.loss_bbox: 0.0436, loss: 0.3351
2022-11-12 20:52:18,054 - mmdet - INFO - Epoch [9][350/962]	lr: 2.000e-03, eta: 0:52:09, time: 0.898, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0068, loss_rpn_bbox: 0.0093, s0.loss_cls: 0.0959, s0.acc: 96.3262, s0.loss_bbox: 0.0676, s1.loss_cls: 0.0424, s1.acc: 96.7910, s1.loss_bbox: 0.0682, s2.loss_cls: 0.0209, s2.acc: 96.7773, s2.loss_bbox: 0.0426, loss: 0.3537
2022-11-12 20:53:03,292 - mmdet - INFO - Epoch [9][400/962]	lr: 2.000e-03, eta: 0:51:25, time: 0.905, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0073, loss_rpn_bbox: 0.0060, s0.loss_cls: 0.0821, s0.acc: 96.8770, s0.loss_bbox: 0.0558, s1.loss_cls: 0.0357, s1.acc: 97.2422, s1.loss_bbox: 0.0546, s2.loss_cls: 0.0173, s2.acc: 97.2988, s2.loss_bbox: 0.0351, loss: 0.2938
2022-11-12 20:53:48,415 - mmdet - INFO - Epoch [9][450/962]	lr: 2.000e-03, eta: 0:50:40, time: 0.902, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0057, loss_rpn_bbox: 0.0059, s0.loss_cls: 0.0794, s0.acc: 96.9629, s0.loss_bbox: 0.0552, s1.loss_cls: 0.0336, s1.acc: 97.4270, s1.loss_bbox: 0.0582, s2.loss_cls: 0.0179, s2.acc: 97.1144, s2.loss_bbox: 0.0396, loss: 0.2957
2022-11-12 20:54:33,489 - mmdet - INFO - Epoch [9][500/962]	lr: 2.000e-03, eta: 0:49:56, time: 0.901, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0062, loss_rpn_bbox: 0.0078, s0.loss_cls: 0.0911, s0.acc: 96.5371, s0.loss_bbox: 0.0645, s1.loss_cls: 0.0393, s1.acc: 97.0531, s1.loss_bbox: 0.0627, s2.loss_cls: 0.0197, s2.acc: 96.9104, s2.loss_bbox: 0.0426, loss: 0.3338
2022-11-12 20:55:18,629 - mmdet - INFO - Epoch [9][550/962]	lr: 2.000e-03, eta: 0:49:11, time: 0.903, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0079, loss_rpn_bbox: 0.0079, s0.loss_cls: 0.0934, s0.acc: 96.3047, s0.loss_bbox: 0.0682, s1.loss_cls: 0.0404, s1.acc: 96.7930, s1.loss_bbox: 0.0713, s2.loss_cls: 0.0200, s2.acc: 96.8984, s2.loss_bbox: 0.0457, loss: 0.3547
2022-11-12 20:56:03,611 - mmdet - INFO - Epoch [9][600/962]	lr: 2.000e-03, eta: 0:48:26, time: 0.900, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0063, loss_rpn_bbox: 0.0076, s0.loss_cls: 0.0881, s0.acc: 96.6309, s0.loss_bbox: 0.0614, s1.loss_cls: 0.0381, s1.acc: 97.0929, s1.loss_bbox: 0.0628, s2.loss_cls: 0.0183, s2.acc: 97.2978, s2.loss_bbox: 0.0419, loss: 0.3245
2022-11-12 20:56:48,497 - mmdet - INFO - Epoch [9][650/962]	lr: 2.000e-03, eta: 0:47:42, time: 0.898, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0057, loss_rpn_bbox: 0.0071, s0.loss_cls: 0.0897, s0.acc: 96.4551, s0.loss_bbox: 0.0585, s1.loss_cls: 0.0403, s1.acc: 96.8367, s1.loss_bbox: 0.0580, s2.loss_cls: 0.0197, s2.acc: 96.8152, s2.loss_bbox: 0.0403, loss: 0.3192
2022-11-12 20:57:33,582 - mmdet - INFO - Epoch [9][700/962]	lr: 2.000e-03, eta: 0:46:57, time: 0.902, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0086, loss_rpn_bbox: 0.0075, s0.loss_cls: 0.0980, s0.acc: 96.2617, s0.loss_bbox: 0.0701, s1.loss_cls: 0.0414, s1.acc: 96.7581, s1.loss_bbox: 0.0687, s2.loss_cls: 0.0205, s2.acc: 96.7588, s2.loss_bbox: 0.0433, loss: 0.3579
2022-11-12 20:58:18,702 - mmdet - INFO - Epoch [9][750/962]	lr: 2.000e-03, eta: 0:46:12, time: 0.902, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0062, loss_rpn_bbox: 0.0062, s0.loss_cls: 0.0804, s0.acc: 96.9844, s0.loss_bbox: 0.0547, s1.loss_cls: 0.0332, s1.acc: 97.5720, s1.loss_bbox: 0.0527, s2.loss_cls: 0.0165, s2.acc: 97.4013, s2.loss_bbox: 0.0343, loss: 0.2842
2022-11-12 20:59:03,719 - mmdet - INFO - Epoch [9][800/962]	lr: 2.000e-03, eta: 0:45:28, time: 0.900, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0070, loss_rpn_bbox: 0.0075, s0.loss_cls: 0.0887, s0.acc: 96.6621, s0.loss_bbox: 0.0590, s1.loss_cls: 0.0411, s1.acc: 97.0109, s1.loss_bbox: 0.0577, s2.loss_cls: 0.0190, s2.acc: 97.2413, s2.loss_bbox: 0.0376, loss: 0.3177
2022-11-12 20:59:48,829 - mmdet - INFO - Epoch [9][850/962]	lr: 2.000e-03, eta: 0:44:43, time: 0.902, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0060, loss_rpn_bbox: 0.0062, s0.loss_cls: 0.0889, s0.acc: 96.6367, s0.loss_bbox: 0.0595, s1.loss_cls: 0.0388, s1.acc: 97.0625, s1.loss_bbox: 0.0609, s2.loss_cls: 0.0191, s2.acc: 97.0840, s2.loss_bbox: 0.0388, loss: 0.3182
2022-11-12 21:00:34,018 - mmdet - INFO - Epoch [9][900/962]	lr: 2.000e-03, eta: 0:43:58, time: 0.904, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0083, loss_rpn_bbox: 0.0064, s0.loss_cls: 0.0904, s0.acc: 96.5430, s0.loss_bbox: 0.0616, s1.loss_cls: 0.0405, s1.acc: 96.8708, s1.loss_bbox: 0.0612, s2.loss_cls: 0.0198, s2.acc: 96.7963, s2.loss_bbox: 0.0400, loss: 0.3281
2022-11-12 21:01:19,124 - mmdet - INFO - Epoch [9][950/962]	lr: 2.000e-03, eta: 0:43:14, time: 0.902, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0086, loss_rpn_bbox: 0.0075, s0.loss_cls: 0.0864, s0.acc: 96.7168, s0.loss_bbox: 0.0621, s1.loss_cls: 0.0393, s1.acc: 97.0131, s1.loss_bbox: 0.0606, s2.loss_cls: 0.0198, s2.acc: 97.1536, s2.loss_bbox: 0.0405, loss: 0.3249
2022-11-12 21:01:29,926 - mmdet - INFO - Saving checkpoint at 9 epochs
[>>] 200/200, 5.1 task/s, elapsed: 39s, ETA:     0s2022-11-12 21:02:15,133 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.25s).
Accumulating evaluation results...
DONE (t=0.04s).
2022-11-12 21:02:15,437 - mmdet - INFO -
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.136
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.310
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.109
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.136
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.237
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.237
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.237
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.237

2022-11-12 21:02:15,441 - mmdet - INFO - Exp name: customized_model_road.py
2022-11-12 21:02:15,441 - mmdet - INFO - Epoch(val) [9][200]	bbox_mAP: 0.1360, bbox_mAP_50: 0.3100, bbox_mAP_75: 0.1090, bbox_mAP_s: -1.0000, bbox_mAP_m: -1.0000, bbox_mAP_l: 0.1360, bbox_mAP_copypaste: 0.136 0.310 0.109 -1.000 -1.000 0.136
2022-11-12 21:03:03,257 - mmdet - INFO - Epoch [10][50/962]	lr: 2.000e-03, eta: 0:42:16, time: 0.956, data_time: 0.060, memory: 5062, loss_rpn_cls: 0.0047, loss_rpn_bbox: 0.0071, s0.loss_cls: 0.0830, s0.acc: 96.7070, s0.loss_bbox: 0.0630, s1.loss_cls: 0.0338, s1.acc: 97.2989, s1.loss_bbox: 0.0603, s2.loss_cls: 0.0169, s2.acc: 97.3105, s2.loss_bbox: 0.0410, loss: 0.3097
2022-11-12 21:03:48,762 - mmdet - INFO - Epoch [10][100/962]	lr: 2.000e-03, eta: 0:41:31, time: 0.910, data_time: 0.010, memory: 5062, loss_rpn_cls: 0.0060, loss_rpn_bbox: 0.0058, s0.loss_cls: 0.0800, s0.acc: 96.9199, s0.loss_bbox: 0.0568, s1.loss_cls: 0.0342, s1.acc: 97.2959, s1.loss_bbox: 0.0546, s2.loss_cls: 0.0160, s2.acc: 97.5260, s2.loss_bbox: 0.0338, loss: 0.2873
2022-11-12 21:04:33,816 - mmdet - INFO - Epoch [10][150/962]	lr: 2.000e-03, eta: 0:40:47, time: 0.901, data_time: 0.010, memory: 5062, loss_rpn_cls: 0.0061, loss_rpn_bbox: 0.0060, s0.loss_cls: 0.0782, s0.acc: 97.0000, s0.loss_bbox: 0.0577, s1.loss_cls: 0.0346, s1.acc: 97.3834, s1.loss_bbox: 0.0571, s2.loss_cls: 0.0169, s2.acc: 97.4858, s2.loss_bbox: 0.0387, loss: 0.2954
2022-11-12 21:05:18,965 - mmdet - INFO - Epoch [10][200/962]	lr: 2.000e-03, eta: 0:40:02, time: 0.903, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0070, loss_rpn_bbox: 0.0076, s0.loss_cls: 0.0917, s0.acc: 96.4648, s0.loss_bbox: 0.0623, s1.loss_cls: 0.0395, s1.acc: 96.9041, s1.loss_bbox: 0.0623, s2.loss_cls: 0.0192, s2.acc: 96.9068, s2.loss_bbox: 0.0409, loss: 0.3305
2022-11-12 21:06:03,839 - mmdet - INFO - Epoch [10][250/962]	lr: 2.000e-03, eta: 0:39:17, time: 0.897, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0067, loss_rpn_bbox: 0.0062, s0.loss_cls: 0.0775, s0.acc: 96.9609, s0.loss_bbox: 0.0565, s1.loss_cls: 0.0331, s1.acc: 97.4407, s1.loss_bbox: 0.0545, s2.loss_cls: 0.0166, s2.acc: 97.4999, s2.loss_bbox: 0.0365, loss: 0.2874
2022-11-12 21:06:48,725 - mmdet - INFO - Epoch [10][300/962]	lr: 2.000e-03, eta: 0:38:33, time: 0.898, data_time: 0.008, memory: 5062, loss_rpn_cls: 0.0072, loss_rpn_bbox: 0.0059, s0.loss_cls: 0.0810, s0.acc: 96.7988, s0.loss_bbox: 0.0518, s1.loss_cls: 0.0357, s1.acc: 97.2744, s1.loss_bbox: 0.0516, s2.loss_cls: 0.0177, s2.acc: 97.2725, s2.loss_bbox: 0.0346, loss: 0.2855
2022-11-12 21:07:33,772 - mmdet - INFO - Epoch [10][350/962]	lr: 2.000e-03, eta: 0:37:48, time: 0.901, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0057, loss_rpn_bbox: 0.0074, s0.loss_cls: 0.0824, s0.acc: 96.7559, s0.loss_bbox: 0.0617, s1.loss_cls: 0.0364, s1.acc: 97.2228, s1.loss_bbox: 0.0662, s2.loss_cls: 0.0187, s2.acc: 97.1096, s2.loss_bbox: 0.0446, loss: 0.3231
2022-11-12 21:08:18,821 - mmdet - INFO - Epoch [10][400/962]	lr: 2.000e-03, eta: 0:37:04, time: 0.901, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0053, loss_rpn_bbox: 0.0070, s0.loss_cls: 0.0871, s0.acc: 96.6465, s0.loss_bbox: 0.0625, s1.loss_cls: 0.0365, s1.acc: 97.2355, s1.loss_bbox: 0.0570, s2.loss_cls: 0.0177, s2.acc: 97.2416, s2.loss_bbox: 0.0370, loss: 0.3100
2022-11-12 21:09:03,870 - mmdet - INFO - Epoch [10][450/962]	lr: 2.000e-03, eta: 0:36:19, time: 0.901, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0069, loss_rpn_bbox: 0.0070, s0.loss_cls: 0.0756, s0.acc: 97.0293, s0.loss_bbox: 0.0548, s1.loss_cls: 0.0314, s1.acc: 97.5783, s1.loss_bbox: 0.0549, s2.loss_cls: 0.0159, s2.acc: 97.4782, s2.loss_bbox: 0.0357, loss: 0.2823
2022-11-12 21:09:48,885 - mmdet - INFO - Epoch [10][500/962]	lr: 2.000e-03, eta: 0:35:34, time: 0.900, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0051, loss_rpn_bbox: 0.0068, s0.loss_cls: 0.0849, s0.acc: 96.6777, s0.loss_bbox: 0.0616, s1.loss_cls: 0.0360, s1.acc: 97.2444, s1.loss_bbox: 0.0596, s2.loss_cls: 0.0178, s2.acc: 97.2733, s2.loss_bbox: 0.0400, loss: 0.3119
2022-11-12 21:10:33,830 - mmdet - INFO - Epoch [10][550/962]	lr: 2.000e-03, eta: 0:34:50, time: 0.899, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0077, loss_rpn_bbox: 0.0073, s0.loss_cls: 0.0870, s0.acc: 96.4746, s0.loss_bbox: 0.0617, s1.loss_cls: 0.0378, s1.acc: 96.9533, s1.loss_bbox: 0.0643, s2.loss_cls: 0.0190, s2.acc: 96.9443, s2.loss_bbox: 0.0419, loss: 0.3268
2022-11-12 21:11:18,937 - mmdet - INFO - Epoch [10][600/962]	lr: 2.000e-03, eta: 0:34:05, time: 0.902, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0066, loss_rpn_bbox: 0.0063, s0.loss_cls: 0.0719, s0.acc: 97.2402, s0.loss_bbox: 0.0459, s1.loss_cls: 0.0303, s1.acc: 97.6451, s1.loss_bbox: 0.0472, s2.loss_cls: 0.0153, s2.acc: 97.4768, s2.loss_bbox: 0.0338, loss: 0.2574
2022-11-12 21:12:03,976 - mmdet - INFO - Epoch [10][650/962]	lr: 2.000e-03, eta: 0:33:20, time: 0.901, data_time: 0.009, memory: 5062, loss_rpn_cls: 0.0065, loss_rpn_bbox: 0.0058, s0.loss_cls: 0.0732, s0.acc: 97.1699, s0.loss_bbox: 0.0513, s1.loss_cls: 0.0307, s1.acc: 97.6625, s1.loss_bbox: 0.0511, s2.loss_cls: 0.0152, s2.acc: 97.6192, s2.loss_bbox: 0.0352, loss: 0.2691
