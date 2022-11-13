/usr/local/lib/python3.7/dist-packages/mmcv/__init__.py:21: UserWarning: On January 1, 2023, MMCV will release v2.0.0, in which it will remove components related to the training process and add a data transformation module. In addition, it will rename the package names mmcv to mmcv-lite and mmcv-full to mmcv. See https://github.com/open-mmlab/mmcv/blob/master/docs/en/compatibility.md for more details.
  'On January 1, 2023, MMCV will release v2.0.0, in which it will remove '
/content/mmdetection/mmdet/utils/setup_env.py:39: UserWarning: Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
  f'Setting OMP_NUM_THREADS environment variable for each process '
/content/mmdetection/mmdet/utils/setup_env.py:49: UserWarning: Setting MKL_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
  f'Setting MKL_NUM_THREADS environment variable for each process '
2022-11-12 14:29:58,431 - mmdet - INFO - Environment info:
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

2022-11-12 14:29:58,837 - mmdet - INFO - Distributed training: False
2022-11-12 14:29:59,142 - mmdet - INFO - Config:
model = dict(
    type='CascadeRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
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
work_dir = './work_dirs/rcnn_r50_caffe_fpn_mstrain-poly_1x_COCO'
auto_resume = False
gpu_ids = [0]

2022-11-12 14:29:59,166 - mmdet - INFO - Set random seed to 1705663563, deterministic: False
2022-11-12 14:29:59,792 - mmdet - INFO - initialize ResNet with init_cfg {'type': 'Pretrained', 'checkpoint': 'torchvision://resnet50'}
2022-11-12 14:29:59,793 - mmcv - INFO - load model from: torchvision://resnet50
2022-11-12 14:29:59,793 - mmcv - INFO - load checkpoint from torchvision path: torchvision://resnet50
2022-11-12 14:29:59,988 - mmcv - WARNING - The model and loaded state dict do not match exactly

unexpected key in source state_dict: fc.weight, fc.bias

2022-11-12 14:30:00,018 - mmdet - INFO - initialize FPN with init_cfg {'type': 'Xavier', 'layer': 'Conv2d', 'distribution': 'uniform'}
2022-11-12 14:30:00,047 - mmdet - INFO - initialize RPNHead with init_cfg {'type': 'Normal', 'layer': 'Conv2d', 'std': 0.01}
2022-11-12 14:30:00,052 - mmdet - INFO - initialize Shared2FCBBoxHead with init_cfg [{'type': 'Normal', 'std': 0.01, 'override': {'name': 'fc_cls'}}, {'type': 'Normal', 'std': 0.001, 'override': {'name': 'fc_reg'}}, {'type': 'Xavier', 'distribution': 'uniform', 'override': [{'name': 'shared_fcs'}, {'name': 'cls_fcs'}, {'name': 'reg_fcs'}]}]
2022-11-12 14:30:00,166 - mmdet - INFO - initialize Shared2FCBBoxHead with init_cfg [{'type': 'Normal', 'std': 0.01, 'override': {'name': 'fc_cls'}}, {'type': 'Normal', 'std': 0.001, 'override': {'name': 'fc_reg'}}, {'type': 'Xavier', 'distribution': 'uniform', 'override': [{'name': 'shared_fcs'}, {'name': 'cls_fcs'}, {'name': 'reg_fcs'}]}]
2022-11-12 14:30:00,273 - mmdet - INFO - initialize Shared2FCBBoxHead with init_cfg [{'type': 'Normal', 'std': 0.01, 'override': {'name': 'fc_cls'}}, {'type': 'Normal', 'std': 0.001, 'override': {'name': 'fc_reg'}}, {'type': 'Xavier', 'distribution': 'uniform', 'override': [{'name': 'shared_fcs'}, {'name': 'cls_fcs'}, {'name': 'reg_fcs'}]}]
loading annotations into memory...
Done (t=0.03s)
creating index...
index created!
2022-11-12 14:30:04,663 - mmdet - INFO - Automatic scaling of learning rate (LR) has been disabled.
loading annotations into memory...
Done (t=0.01s)
creating index...
index created!
2022-11-12 14:30:04,678 - mmdet - INFO - Start running, host: root@b915b7d09177, work_dir: /content/mmdetection/work_dirs/rcnn_r50_caffe_fpn_mstrain-poly_1x_COCO
2022-11-12 14:30:04,678 - mmdet - INFO - Hooks will be executed in the following order:
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
2022-11-12 14:30:04,678 - mmdet - INFO - workflow: [('train', 1)], max: 12 epochs
2022-11-12 14:30:04,679 - mmdet - INFO - Checkpoints will be saved to /content/mmdetection/work_dirs/rcnn_r50_caffe_fpn_mstrain-poly_1x_COCO by HardDiskBackend.
/usr/local/lib/python3.7/dist-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.)
  return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
2022-11-12 14:30:37,044 - mmdet - INFO - Epoch [1][50/962]	lr: 1.978e-03, eta: 2:03:32, time: 0.645, data_time: 0.058, memory: 3483, loss_rpn_cls: 0.3415, loss_rpn_bbox: 0.0224, s0.loss_cls: 0.3770, s0.acc: 88.1094, s0.loss_bbox: 0.0769, s1.loss_cls: 0.1522, s1.acc: 91.4746, s1.loss_bbox: 0.0272, s2.loss_cls: 0.0933, s2.acc: 85.0293, s2.loss_bbox: 0.0045, loss: 1.0950
2022-11-12 14:31:08,047 - mmdet - INFO - Epoch [1][100/962]	lr: 3.976e-03, eta: 2:00:37, time: 0.620, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0471, loss_rpn_bbox: 0.0252, s0.loss_cls: 0.2384, s0.acc: 92.6504, s0.loss_bbox: 0.1699, s1.loss_cls: 0.0868, s1.acc: 95.3945, s1.loss_bbox: 0.0850, s2.loss_cls: 0.0271, s2.acc: 97.6660, s2.loss_bbox: 0.0176, loss: 0.6972
2022-11-12 14:31:39,626 - mmdet - INFO - Epoch [1][150/962]	lr: 5.974e-03, eta: 2:00:02, time: 0.632, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0317, loss_rpn_bbox: 0.0205, s0.loss_cls: 0.2176, s0.acc: 92.4258, s0.loss_bbox: 0.1501, s1.loss_cls: 0.0864, s1.acc: 94.5918, s1.loss_bbox: 0.0931, s2.loss_cls: 0.0293, s2.acc: 97.1211, s2.loss_bbox: 0.0231, loss: 0.6517
2022-11-12 14:32:13,058 - mmdet - INFO - Epoch [1][200/962]	lr: 7.972e-03, eta: 2:01:14, time: 0.669, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0365, loss_rpn_bbox: 0.0239, s0.loss_cls: 0.2332, s0.acc: 92.0312, s0.loss_bbox: 0.1582, s1.loss_cls: 0.0929, s1.acc: 94.4023, s1.loss_bbox: 0.0995, s2.loss_cls: 0.0302, s2.acc: 97.0020, s2.loss_bbox: 0.0245, loss: 0.6990
2022-11-12 14:32:47,381 - mmdet - INFO - Epoch [1][250/962]	lr: 9.970e-03, eta: 2:02:24, time: 0.686, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0310, loss_rpn_bbox: 0.0210, s0.loss_cls: 0.2233, s0.acc: 92.3203, s0.loss_bbox: 0.1663, s1.loss_cls: 0.0902, s1.acc: 94.5059, s1.loss_bbox: 0.0979, s2.loss_cls: 0.0303, s2.acc: 97.0195, s2.loss_bbox: 0.0247, loss: 0.6847
2022-11-12 14:33:20,572 - mmdet - INFO - Epoch [1][300/962]	lr: 1.197e-02, eta: 2:02:17, time: 0.664, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0250, loss_rpn_bbox: 0.0204, s0.loss_cls: 0.2380, s0.acc: 91.6035, s0.loss_bbox: 0.1645, s1.loss_cls: 0.0998, s1.acc: 93.8164, s1.loss_bbox: 0.1070, s2.loss_cls: 0.0336, s2.acc: 96.5625, s2.loss_bbox: 0.0282, loss: 0.7165
2022-11-12 14:33:54,976 - mmdet - INFO - Epoch [1][350/962]	lr: 1.397e-02, eta: 2:02:41, time: 0.688, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0326, loss_rpn_bbox: 0.0231, s0.loss_cls: 0.2210, s0.acc: 92.5371, s0.loss_bbox: 0.1484, s1.loss_cls: 0.0926, s1.acc: 94.2520, s1.loss_bbox: 0.0985, s2.loss_cls: 0.0320, s2.acc: 96.6504, s2.loss_bbox: 0.0280, loss: 0.6763
2022-11-12 14:34:29,967 - mmdet - INFO - Epoch [1][400/962]	lr: 1.596e-02, eta: 2:03:07, time: 0.700, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0275, loss_rpn_bbox: 0.0242, s0.loss_cls: 0.2187, s0.acc: 92.4375, s0.loss_bbox: 0.1566, s1.loss_cls: 0.0911, s1.acc: 94.2617, s1.loss_bbox: 0.1017, s2.loss_cls: 0.0312, s2.acc: 96.8027, s2.loss_bbox: 0.0265, loss: 0.6774
2022-11-12 14:35:04,482 - mmdet - INFO - Epoch [1][450/962]	lr: 1.796e-02, eta: 2:03:07, time: 0.690, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0239, loss_rpn_bbox: 0.0191, s0.loss_cls: 0.2153, s0.acc: 92.6426, s0.loss_bbox: 0.1466, s1.loss_cls: 0.0878, s1.acc: 94.3418, s1.loss_bbox: 0.0967, s2.loss_cls: 0.0314, s2.acc: 96.5820, s2.loss_bbox: 0.0290, loss: 0.6497
2022-11-12 14:35:38,898 - mmdet - INFO - Epoch [1][500/962]	lr: 1.996e-02, eta: 2:02:59, time: 0.688, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0198, loss_rpn_bbox: 0.0165, s0.loss_cls: 0.2097, s0.acc: 92.3945, s0.loss_bbox: 0.1487, s1.loss_cls: 0.0936, s1.acc: 93.3945, s1.loss_bbox: 0.1087, s2.loss_cls: 0.0357, s2.acc: 95.5723, s2.loss_bbox: 0.0387, loss: 0.6715
2022-11-12 14:36:12,395 - mmdet - INFO - Epoch [1][550/962]	lr: 2.000e-02, eta: 2:02:27, time: 0.670, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0235, loss_rpn_bbox: 0.0182, s0.loss_cls: 0.2261, s0.acc: 92.3867, s0.loss_bbox: 0.1469, s1.loss_cls: 0.0931, s1.acc: 93.9941, s1.loss_bbox: 0.0972, s2.loss_cls: 0.0330, s2.acc: 96.2207, s2.loss_bbox: 0.0316, loss: 0.6694
2022-11-12 14:36:46,457 - mmdet - INFO - Epoch [1][600/962]	lr: 2.000e-02, eta: 2:02:05, time: 0.681, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0208, loss_rpn_bbox: 0.0182, s0.loss_cls: 0.2086, s0.acc: 92.3047, s0.loss_bbox: 0.1452, s1.loss_cls: 0.0921, s1.acc: 93.2891, s1.loss_bbox: 0.1093, s2.loss_cls: 0.0348, s2.acc: 95.3438, s2.loss_bbox: 0.0394, loss: 0.6682
2022-11-12 14:37:20,454 - mmdet - INFO - Epoch [1][650/962]	lr: 2.000e-02, eta: 2:01:41, time: 0.680, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0206, loss_rpn_bbox: 0.0184, s0.loss_cls: 0.2202, s0.acc: 91.7832, s0.loss_bbox: 0.1528, s1.loss_cls: 0.0981, s1.acc: 92.9219, s1.loss_bbox: 0.1161, s2.loss_cls: 0.0377, s2.acc: 94.9551, s2.loss_bbox: 0.0430, loss: 0.7070
2022-11-12 14:37:53,813 - mmdet - INFO - Epoch [1][700/962]	lr: 2.000e-02, eta: 2:01:05, time: 0.667, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0284, loss_rpn_bbox: 0.0216, s0.loss_cls: 0.2344, s0.acc: 91.6797, s0.loss_bbox: 0.1539, s1.loss_cls: 0.1002, s1.acc: 92.8730, s1.loss_bbox: 0.1181, s2.loss_cls: 0.0367, s2.acc: 95.1777, s2.loss_bbox: 0.0423, loss: 0.7356
2022-11-12 14:38:28,360 - mmdet - INFO - Epoch [1][750/962]	lr: 2.000e-02, eta: 2:00:47, time: 0.691, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0250, loss_rpn_bbox: 0.0204, s0.loss_cls: 0.2065, s0.acc: 92.2031, s0.loss_bbox: 0.1412, s1.loss_cls: 0.0905, s1.acc: 93.1562, s1.loss_bbox: 0.1117, s2.loss_cls: 0.0366, s2.acc: 94.7871, s2.loss_bbox: 0.0463, loss: 0.6782
2022-11-12 14:39:01,794 - mmdet - INFO - Epoch [1][800/962]	lr: 2.000e-02, eta: 2:00:11, time: 0.669, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0203, loss_rpn_bbox: 0.0160, s0.loss_cls: 0.2047, s0.acc: 92.4902, s0.loss_bbox: 0.1327, s1.loss_cls: 0.0909, s1.acc: 93.3633, s1.loss_bbox: 0.1073, s2.loss_cls: 0.0364, s2.acc: 94.9004, s2.loss_bbox: 0.0454, loss: 0.6539
2022-11-12 14:39:35,674 - mmdet - INFO - Epoch [1][850/962]	lr: 2.000e-02, eta: 1:59:42, time: 0.678, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0227, loss_rpn_bbox: 0.0190, s0.loss_cls: 0.2056, s0.acc: 92.5352, s0.loss_bbox: 0.1373, s1.loss_cls: 0.0894, s1.acc: 93.4863, s1.loss_bbox: 0.1019, s2.loss_cls: 0.0355, s2.acc: 95.0684, s2.loss_bbox: 0.0410, loss: 0.6525
2022-11-12 14:40:09,717 - mmdet - INFO - Epoch [1][900/962]	lr: 2.000e-02, eta: 1:59:13, time: 0.681, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0227, loss_rpn_bbox: 0.0183, s0.loss_cls: 0.2273, s0.acc: 91.4297, s0.loss_bbox: 0.1527, s1.loss_cls: 0.0996, s1.acc: 92.6172, s1.loss_bbox: 0.1216, s2.loss_cls: 0.0403, s2.acc: 94.4473, s2.loss_bbox: 0.0520, loss: 0.7345
2022-11-12 14:40:43,618 - mmdet - INFO - Epoch [1][950/962]	lr: 2.000e-02, eta: 1:58:43, time: 0.678, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0200, loss_rpn_bbox: 0.0168, s0.loss_cls: 0.2004, s0.acc: 92.3379, s0.loss_bbox: 0.1386, s1.loss_cls: 0.0912, s1.acc: 93.0605, s1.loss_bbox: 0.1139, s2.loss_cls: 0.0382, s2.acc: 94.1895, s2.loss_bbox: 0.0521, loss: 0.6713
2022-11-12 14:40:51,858 - mmdet - INFO - Saving checkpoint at 1 epochs
[>>] 200/200, 6.4 task/s, elapsed: 31s, ETA:     0s2022-11-12 14:41:27,968 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.13s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.43s).
Accumulating evaluation results...
DONE (t=0.10s).
2022-11-12 14:41:28,645 - mmdet - INFO -
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.031
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.119
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.009
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.031
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.169
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.169
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.169
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.169

2022-11-12 14:41:28,652 - mmdet - INFO - Exp name: rcnn_r50_caffe_fpn_mstrain-poly_1x_COCO.py
2022-11-12 14:41:28,652 - mmdet - INFO - Epoch(val) [1][200]	bbox_mAP: 0.0310, bbox_mAP_50: 0.1190, bbox_mAP_75: 0.0090, bbox_mAP_s: -1.0000, bbox_mAP_m: -1.0000, bbox_mAP_l: 0.0310, bbox_mAP_copypaste: 0.031 0.119 0.009 -1.000 -1.000 0.031
2022-11-12 14:42:05,631 - mmdet - INFO - Epoch [2][50/962]	lr: 2.000e-02, eta: 1:57:11, time: 0.737, data_time: 0.058, memory: 3483, loss_rpn_cls: 0.0152, loss_rpn_bbox: 0.0114, s0.loss_cls: 0.1882, s0.acc: 93.0684, s0.loss_bbox: 0.1213, s1.loss_cls: 0.0868, s1.acc: 93.5801, s1.loss_bbox: 0.1068, s2.loss_cls: 0.0364, s2.acc: 94.6445, s2.loss_bbox: 0.0509, loss: 0.6170
2022-11-12 14:42:39,837 - mmdet - INFO - Epoch [2][100/962]	lr: 2.000e-02, eta: 1:56:46, time: 0.684, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0158, loss_rpn_bbox: 0.0137, s0.loss_cls: 0.1829, s0.acc: 92.8125, s0.loss_bbox: 0.1265, s1.loss_cls: 0.0856, s1.acc: 93.4219, s1.loss_bbox: 0.1136, s2.loss_cls: 0.0371, s2.acc: 94.4180, s2.loss_bbox: 0.0539, loss: 0.6291
2022-11-12 14:43:13,829 - mmdet - INFO - Epoch [2][150/962]	lr: 2.000e-02, eta: 1:56:18, time: 0.680, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0177, loss_rpn_bbox: 0.0133, s0.loss_cls: 0.1686, s0.acc: 93.9062, s0.loss_bbox: 0.1073, s1.loss_cls: 0.0793, s1.acc: 94.1426, s1.loss_bbox: 0.0972, s2.loss_cls: 0.0332, s2.acc: 95.2012, s2.loss_bbox: 0.0429, loss: 0.5595
2022-11-12 14:43:47,877 - mmdet - INFO - Epoch [2][200/962]	lr: 2.000e-02, eta: 1:55:50, time: 0.681, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0159, loss_rpn_bbox: 0.0153, s0.loss_cls: 0.1893, s0.acc: 93.0234, s0.loss_bbox: 0.1279, s1.loss_cls: 0.0862, s1.acc: 93.6016, s1.loss_bbox: 0.1131, s2.loss_cls: 0.0359, s2.acc: 94.8496, s2.loss_bbox: 0.0513, loss: 0.6349
2022-11-12 14:44:21,528 - mmdet - INFO - Epoch [2][250/962]	lr: 2.000e-02, eta: 1:55:18, time: 0.673, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0199, loss_rpn_bbox: 0.0153, s0.loss_cls: 0.1950, s0.acc: 92.5645, s0.loss_bbox: 0.1350, s1.loss_cls: 0.0891, s1.acc: 93.2480, s1.loss_bbox: 0.1123, s2.loss_cls: 0.0378, s2.acc: 94.3750, s2.loss_bbox: 0.0510, loss: 0.6553
2022-11-12 14:44:55,759 - mmdet - INFO - Epoch [2][300/962]	lr: 2.000e-02, eta: 1:54:50, time: 0.685, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0198, loss_rpn_bbox: 0.0151, s0.loss_cls: 0.1856, s0.acc: 93.1016, s0.loss_bbox: 0.1280, s1.loss_cls: 0.0842, s1.acc: 93.7266, s1.loss_bbox: 0.1092, s2.loss_cls: 0.0350, s2.acc: 94.8984, s2.loss_bbox: 0.0500, loss: 0.6268
2022-11-12 14:45:29,913 - mmdet - INFO - Epoch [2][350/962]	lr: 2.000e-02, eta: 1:54:22, time: 0.683, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0238, loss_rpn_bbox: 0.0173, s0.loss_cls: 0.1786, s0.acc: 93.3223, s0.loss_bbox: 0.1280, s1.loss_cls: 0.0795, s1.acc: 94.0977, s1.loss_bbox: 0.1036, s2.loss_cls: 0.0332, s2.acc: 95.1309, s2.loss_bbox: 0.0498, loss: 0.6138
2022-11-12 14:46:03,411 - mmdet - INFO - Epoch [2][400/962]	lr: 2.000e-02, eta: 1:53:48, time: 0.670, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0192, loss_rpn_bbox: 0.0142, s0.loss_cls: 0.1762, s0.acc: 93.3047, s0.loss_bbox: 0.1164, s1.loss_cls: 0.0786, s1.acc: 94.0723, s1.loss_bbox: 0.1001, s2.loss_cls: 0.0327, s2.acc: 95.2324, s2.loss_bbox: 0.0479, loss: 0.5853
2022-11-12 14:46:38,215 - mmdet - INFO - Epoch [2][450/962]	lr: 2.000e-02, eta: 1:53:24, time: 0.696, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0204, loss_rpn_bbox: 0.0157, s0.loss_cls: 0.1866, s0.acc: 93.0059, s0.loss_bbox: 0.1345, s1.loss_cls: 0.0849, s1.acc: 93.8359, s1.loss_bbox: 0.1188, s2.loss_cls: 0.0361, s2.acc: 94.9062, s2.loss_bbox: 0.0552, loss: 0.6522
2022-11-12 14:47:11,912 - mmdet - INFO - Epoch [2][500/962]	lr: 2.000e-02, eta: 1:52:51, time: 0.674, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0169, loss_rpn_bbox: 0.0161, s0.loss_cls: 0.1664, s0.acc: 93.8320, s0.loss_bbox: 0.1180, s1.loss_cls: 0.0801, s1.acc: 94.0859, s1.loss_bbox: 0.1025, s2.loss_cls: 0.0335, s2.acc: 95.1113, s2.loss_bbox: 0.0479, loss: 0.5814
2022-11-12 14:47:45,821 - mmdet - INFO - Epoch [2][550/962]	lr: 2.000e-02, eta: 1:52:19, time: 0.678, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0170, loss_rpn_bbox: 0.0154, s0.loss_cls: 0.1725, s0.acc: 93.3652, s0.loss_bbox: 0.1218, s1.loss_cls: 0.0822, s1.acc: 93.7246, s1.loss_bbox: 0.1081, s2.loss_cls: 0.0368, s2.acc: 94.5039, s2.loss_bbox: 0.0550, loss: 0.6088
2022-11-12 14:48:19,774 - mmdet - INFO - Epoch [2][600/962]	lr: 2.000e-02, eta: 1:51:48, time: 0.679, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0166, loss_rpn_bbox: 0.0122, s0.loss_cls: 0.1894, s0.acc: 92.9180, s0.loss_bbox: 0.1292, s1.loss_cls: 0.0887, s1.acc: 93.3652, s1.loss_bbox: 0.1120, s2.loss_cls: 0.0373, s2.acc: 94.6426, s2.loss_bbox: 0.0535, loss: 0.6390
2022-11-12 14:48:53,340 - mmdet - INFO - Epoch [2][650/962]	lr: 2.000e-02, eta: 1:51:14, time: 0.671, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0182, loss_rpn_bbox: 0.0158, s0.loss_cls: 0.1864, s0.acc: 93.0898, s0.loss_bbox: 0.1291, s1.loss_cls: 0.0848, s1.acc: 93.7676, s1.loss_bbox: 0.1079, s2.loss_cls: 0.0356, s2.acc: 94.9590, s2.loss_bbox: 0.0509, loss: 0.6287
2022-11-12 14:49:27,506 - mmdet - INFO - Epoch [2][700/962]	lr: 2.000e-02, eta: 1:50:44, time: 0.683, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0148, loss_rpn_bbox: 0.0117, s0.loss_cls: 0.1788, s0.acc: 93.2363, s0.loss_bbox: 0.1322, s1.loss_cls: 0.0814, s1.acc: 93.8008, s1.loss_bbox: 0.1134, s2.loss_cls: 0.0360, s2.acc: 94.6797, s2.loss_bbox: 0.0561, loss: 0.6245
2022-11-12 14:50:01,682 - mmdet - INFO - Epoch [2][750/962]	lr: 2.000e-02, eta: 1:50:14, time: 0.684, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0135, loss_rpn_bbox: 0.0138, s0.loss_cls: 0.1683, s0.acc: 93.5039, s0.loss_bbox: 0.1170, s1.loss_cls: 0.0770, s1.acc: 94.0938, s1.loss_bbox: 0.1053, s2.loss_cls: 0.0328, s2.acc: 95.0117, s2.loss_bbox: 0.0507, loss: 0.5785
2022-11-12 14:50:35,963 - mmdet - INFO - Epoch [2][800/962]	lr: 2.000e-02, eta: 1:49:44, time: 0.686, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0095, loss_rpn_bbox: 0.0108, s0.loss_cls: 0.1554, s0.acc: 93.9980, s0.loss_bbox: 0.1108, s1.loss_cls: 0.0752, s1.acc: 94.1758, s1.loss_bbox: 0.1030, s2.loss_cls: 0.0341, s2.acc: 94.7734, s2.loss_bbox: 0.0540, loss: 0.5528
2022-11-12 14:51:10,053 - mmdet - INFO - Epoch [2][850/962]	lr: 2.000e-02, eta: 1:49:12, time: 0.682, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0130, loss_rpn_bbox: 0.0123, s0.loss_cls: 0.1654, s0.acc: 93.7031, s0.loss_bbox: 0.1224, s1.loss_cls: 0.0751, s1.acc: 94.3418, s1.loss_bbox: 0.1125, s2.loss_cls: 0.0323, s2.acc: 95.2090, s2.loss_bbox: 0.0560, loss: 0.5890
2022-11-12 14:51:44,269 - mmdet - INFO - Epoch [2][900/962]	lr: 2.000e-02, eta: 1:48:41, time: 0.684, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0171, loss_rpn_bbox: 0.0135, s0.loss_cls: 0.1965, s0.acc: 92.5059, s0.loss_bbox: 0.1356, s1.loss_cls: 0.0917, s1.acc: 93.3398, s1.loss_bbox: 0.1169, s2.loss_cls: 0.0395, s2.acc: 94.4531, s2.loss_bbox: 0.0564, loss: 0.6673
2022-11-12 14:52:18,015 - mmdet - INFO - Epoch [2][950/962]	lr: 2.000e-02, eta: 1:48:08, time: 0.675, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0148, loss_rpn_bbox: 0.0133, s0.loss_cls: 0.1854, s0.acc: 93.0098, s0.loss_bbox: 0.1230, s1.loss_cls: 0.0884, s1.acc: 93.2891, s1.loss_bbox: 0.1117, s2.loss_cls: 0.0384, s2.acc: 94.1660, s2.loss_bbox: 0.0568, loss: 0.6321
2022-11-12 14:52:26,125 - mmdet - INFO - Saving checkpoint at 2 epochs
[>>] 200/200, 6.2 task/s, elapsed: 32s, ETA:     0s2022-11-12 14:53:03,147 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.37s).
Accumulating evaluation results...
DONE (t=0.06s).
2022-11-12 14:53:03,597 - mmdet - INFO -
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.059
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.198
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.020
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.059
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.191
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.191
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.191
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.191

2022-11-12 14:53:03,602 - mmdet - INFO - Exp name: rcnn_r50_caffe_fpn_mstrain-poly_1x_COCO.py
2022-11-12 14:53:03,602 - mmdet - INFO - Epoch(val) [2][200]	bbox_mAP: 0.0590, bbox_mAP_50: 0.1980, bbox_mAP_75: 0.0200, bbox_mAP_s: -1.0000, bbox_mAP_m: -1.0000, bbox_mAP_l: 0.0590, bbox_mAP_copypaste: 0.059 0.198 0.020 -1.000 -1.000 0.059
2022-11-12 14:53:40,205 - mmdet - INFO - Epoch [3][50/962]	lr: 2.000e-02, eta: 1:47:01, time: 0.729, data_time: 0.058, memory: 3483, loss_rpn_cls: 0.0158, loss_rpn_bbox: 0.0113, s0.loss_cls: 0.1637, s0.acc: 93.8711, s0.loss_bbox: 0.1153, s1.loss_cls: 0.0755, s1.acc: 94.3984, s1.loss_bbox: 0.1004, s2.loss_cls: 0.0322, s2.acc: 95.0762, s2.loss_bbox: 0.0518, loss: 0.5659
2022-11-12 14:54:13,661 - mmdet - INFO - Epoch [3][100/962]	lr: 2.000e-02, eta: 1:46:27, time: 0.669, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0119, loss_rpn_bbox: 0.0125, s0.loss_cls: 0.1670, s0.acc: 93.7305, s0.loss_bbox: 0.1159, s1.loss_cls: 0.0793, s1.acc: 93.9941, s1.loss_bbox: 0.1131, s2.loss_cls: 0.0361, s2.acc: 94.5371, s2.loss_bbox: 0.0620, loss: 0.5978
2022-11-12 14:54:48,235 - mmdet - INFO - Epoch [3][150/962]	lr: 2.000e-02, eta: 1:45:58, time: 0.691, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0147, loss_rpn_bbox: 0.0123, s0.loss_cls: 0.1823, s0.acc: 93.0176, s0.loss_bbox: 0.1310, s1.loss_cls: 0.0857, s1.acc: 93.3672, s1.loss_bbox: 0.1152, s2.loss_cls: 0.0389, s2.acc: 94.2012, s2.loss_bbox: 0.0589, loss: 0.6390
2022-11-12 14:55:22,053 - mmdet - INFO - Epoch [3][200/962]	lr: 2.000e-02, eta: 1:45:25, time: 0.676, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0139, loss_rpn_bbox: 0.0125, s0.loss_cls: 0.1714, s0.acc: 93.3906, s0.loss_bbox: 0.1251, s1.loss_cls: 0.0779, s1.acc: 94.0156, s1.loss_bbox: 0.1151, s2.loss_cls: 0.0343, s2.acc: 94.6406, s2.loss_bbox: 0.0601, loss: 0.6104
2022-11-12 14:55:55,982 - mmdet - INFO - Epoch [3][250/962]	lr: 2.000e-02, eta: 1:44:53, time: 0.679, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0151, loss_rpn_bbox: 0.0141, s0.loss_cls: 0.1639, s0.acc: 93.6738, s0.loss_bbox: 0.1173, s1.loss_cls: 0.0772, s1.acc: 94.0586, s1.loss_bbox: 0.1059, s2.loss_cls: 0.0347, s2.acc: 94.7344, s2.loss_bbox: 0.0561, loss: 0.5842
2022-11-12 14:56:30,133 - mmdet - INFO - Epoch [3][300/962]	lr: 2.000e-02, eta: 1:44:22, time: 0.683, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0136, loss_rpn_bbox: 0.0133, s0.loss_cls: 0.1641, s0.acc: 93.5996, s0.loss_bbox: 0.1133, s1.loss_cls: 0.0756, s1.acc: 94.1113, s1.loss_bbox: 0.1046, s2.loss_cls: 0.0351, s2.acc: 94.4434, s2.loss_bbox: 0.0558, loss: 0.5755
2022-11-12 14:57:03,688 - mmdet - INFO - Epoch [3][350/962]	lr: 2.000e-02, eta: 1:43:48, time: 0.671, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0131, loss_rpn_bbox: 0.0111, s0.loss_cls: 0.1507, s0.acc: 94.3066, s0.loss_bbox: 0.1042, s1.loss_cls: 0.0681, s1.acc: 94.7984, s1.loss_bbox: 0.0954, s2.loss_cls: 0.0302, s2.acc: 95.3104, s2.loss_bbox: 0.0498, loss: 0.5227
2022-11-12 14:57:37,597 - mmdet - INFO - Epoch [3][400/962]	lr: 2.000e-02, eta: 1:43:16, time: 0.678, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0179, loss_rpn_bbox: 0.0118, s0.loss_cls: 0.1580, s0.acc: 93.7324, s0.loss_bbox: 0.1139, s1.loss_cls: 0.0761, s1.acc: 94.1035, s1.loss_bbox: 0.1064, s2.loss_cls: 0.0338, s2.acc: 94.8203, s2.loss_bbox: 0.0573, loss: 0.5752
2022-11-12 14:58:11,572 - mmdet - INFO - Epoch [3][450/962]	lr: 2.000e-02, eta: 1:42:44, time: 0.679, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0130, loss_rpn_bbox: 0.0111, s0.loss_cls: 0.1615, s0.acc: 93.7188, s0.loss_bbox: 0.1105, s1.loss_cls: 0.0773, s1.acc: 94.1992, s1.loss_bbox: 0.1034, s2.loss_cls: 0.0355, s2.acc: 94.6895, s2.loss_bbox: 0.0549, loss: 0.5673
2022-11-12 14:58:45,558 - mmdet - INFO - Epoch [3][500/962]	lr: 2.000e-02, eta: 1:42:12, time: 0.680, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0127, loss_rpn_bbox: 0.0105, s0.loss_cls: 0.1621, s0.acc: 94.0508, s0.loss_bbox: 0.1126, s1.loss_cls: 0.0766, s1.acc: 94.5547, s1.loss_bbox: 0.1021, s2.loss_cls: 0.0333, s2.acc: 95.3535, s2.loss_bbox: 0.0520, loss: 0.5620
2022-11-12 14:59:19,738 - mmdet - INFO - Epoch [3][550/962]	lr: 2.000e-02, eta: 1:41:40, time: 0.684, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0132, loss_rpn_bbox: 0.0133, s0.loss_cls: 0.1655, s0.acc: 93.4629, s0.loss_bbox: 0.1218, s1.loss_cls: 0.0774, s1.acc: 94.2461, s1.loss_bbox: 0.1087, s2.loss_cls: 0.0341, s2.acc: 95.0625, s2.loss_bbox: 0.0546, loss: 0.5887
2022-11-12 14:59:53,587 - mmdet - INFO - Epoch [3][600/962]	lr: 2.000e-02, eta: 1:41:07, time: 0.677, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0132, loss_rpn_bbox: 0.0121, s0.loss_cls: 0.1699, s0.acc: 93.3223, s0.loss_bbox: 0.1214, s1.loss_cls: 0.0802, s1.acc: 93.6719, s1.loss_bbox: 0.1103, s2.loss_cls: 0.0368, s2.acc: 94.3457, s2.loss_bbox: 0.0563, loss: 0.6001
2022-11-12 15:00:27,478 - mmdet - INFO - Epoch [3][650/962]	lr: 2.000e-02, eta: 1:40:34, time: 0.678, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0143, loss_rpn_bbox: 0.0119, s0.loss_cls: 0.1694, s0.acc: 93.5566, s0.loss_bbox: 0.1250, s1.loss_cls: 0.0770, s1.acc: 94.1719, s1.loss_bbox: 0.1101, s2.loss_cls: 0.0339, s2.acc: 95.0488, s2.loss_bbox: 0.0544, loss: 0.5962
2022-11-12 15:01:01,555 - mmdet - INFO - Epoch [3][700/962]	lr: 2.000e-02, eta: 1:40:02, time: 0.682, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0125, loss_rpn_bbox: 0.0095, s0.loss_cls: 0.1400, s0.acc: 94.4277, s0.loss_bbox: 0.0988, s1.loss_cls: 0.0652, s1.acc: 94.8125, s1.loss_bbox: 0.0975, s2.loss_cls: 0.0295, s2.acc: 95.5059, s2.loss_bbox: 0.0541, loss: 0.5073
2022-11-12 15:01:35,196 - mmdet - INFO - Epoch [3][750/962]	lr: 2.000e-02, eta: 1:39:29, time: 0.673, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0167, loss_rpn_bbox: 0.0144, s0.loss_cls: 0.1561, s0.acc: 93.9766, s0.loss_bbox: 0.1149, s1.loss_cls: 0.0708, s1.acc: 94.5273, s1.loss_bbox: 0.1083, s2.loss_cls: 0.0330, s2.acc: 94.9316, s2.loss_bbox: 0.0561, loss: 0.5704
2022-11-12 15:02:09,431 - mmdet - INFO - Epoch [3][800/962]	lr: 2.000e-02, eta: 1:38:57, time: 0.685, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0127, loss_rpn_bbox: 0.0125, s0.loss_cls: 0.1738, s0.acc: 93.2305, s0.loss_bbox: 0.1261, s1.loss_cls: 0.0772, s1.acc: 94.1641, s1.loss_bbox: 0.1134, s2.loss_cls: 0.0337, s2.acc: 94.8828, s2.loss_bbox: 0.0604, loss: 0.6099
2022-11-12 15:02:44,241 - mmdet - INFO - Epoch [3][850/962]	lr: 2.000e-02, eta: 1:38:27, time: 0.696, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0136, loss_rpn_bbox: 0.0145, s0.loss_cls: 0.1644, s0.acc: 93.8184, s0.loss_bbox: 0.1169, s1.loss_cls: 0.0746, s1.acc: 94.3965, s1.loss_bbox: 0.1082, s2.loss_cls: 0.0333, s2.acc: 95.0352, s2.loss_bbox: 0.0575, loss: 0.5830
2022-11-12 15:03:18,218 - mmdet - INFO - Epoch [3][900/962]	lr: 2.000e-02, eta: 1:37:54, time: 0.680, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0217, loss_rpn_bbox: 0.0125, s0.loss_cls: 0.1588, s0.acc: 93.8789, s0.loss_bbox: 0.1097, s1.loss_cls: 0.0740, s1.acc: 94.4668, s1.loss_bbox: 0.1003, s2.loss_cls: 0.0317, s2.acc: 95.3633, s2.loss_bbox: 0.0507, loss: 0.5595
2022-11-12 15:03:53,120 - mmdet - INFO - Epoch [3][950/962]	lr: 2.000e-02, eta: 1:37:24, time: 0.698, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0140, loss_rpn_bbox: 0.0123, s0.loss_cls: 0.1642, s0.acc: 93.6973, s0.loss_bbox: 0.1098, s1.loss_cls: 0.0801, s1.acc: 93.8340, s1.loss_bbox: 0.0995, s2.loss_cls: 0.0359, s2.acc: 94.4180, s2.loss_bbox: 0.0538, loss: 0.5695
2022-11-12 15:04:01,358 - mmdet - INFO - Saving checkpoint at 3 epochs
[>>] 200/200, 6.2 task/s, elapsed: 32s, ETA:     0s2022-11-12 15:04:38,500 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.61s).
Accumulating evaluation results...
DONE (t=0.10s).
2022-11-12 15:04:39,249 - mmdet - INFO -
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.076
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.216
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.038
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.076
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.207
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.207
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.207
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.207

2022-11-12 15:04:39,256 - mmdet - INFO - Exp name: rcnn_r50_caffe_fpn_mstrain-poly_1x_COCO.py
2022-11-12 15:04:39,256 - mmdet - INFO - Epoch(val) [3][200]	bbox_mAP: 0.0760, bbox_mAP_50: 0.2160, bbox_mAP_75: 0.0380, bbox_mAP_s: -1.0000, bbox_mAP_m: -1.0000, bbox_mAP_l: 0.0760, bbox_mAP_copypaste: 0.076 0.216 0.038 -1.000 -1.000 0.076
2022-11-12 15:05:15,808 - mmdet - INFO - Epoch [4][50/962]	lr: 2.000e-02, eta: 1:36:27, time: 0.728, data_time: 0.058, memory: 3483, loss_rpn_cls: 0.0131, loss_rpn_bbox: 0.0121, s0.loss_cls: 0.1586, s0.acc: 94.1758, s0.loss_bbox: 0.1085, s1.loss_cls: 0.0724, s1.acc: 94.8379, s1.loss_bbox: 0.0978, s2.loss_cls: 0.0318, s2.acc: 95.5117, s2.loss_bbox: 0.0518, loss: 0.5462
2022-11-12 15:05:50,415 - mmdet - INFO - Epoch [4][100/962]	lr: 2.000e-02, eta: 1:35:56, time: 0.692, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0103, loss_rpn_bbox: 0.0126, s0.loss_cls: 0.1618, s0.acc: 93.7891, s0.loss_bbox: 0.1184, s1.loss_cls: 0.0736, s1.acc: 94.1953, s1.loss_bbox: 0.1098, s2.loss_cls: 0.0326, s2.acc: 94.9590, s2.loss_bbox: 0.0583, loss: 0.5773
2022-11-12 15:06:24,797 - mmdet - INFO - Epoch [4][150/962]	lr: 2.000e-02, eta: 1:35:24, time: 0.688, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0172, loss_rpn_bbox: 0.0125, s0.loss_cls: 0.1635, s0.acc: 93.7656, s0.loss_bbox: 0.1155, s1.loss_cls: 0.0755, s1.acc: 94.1519, s1.loss_bbox: 0.1092, s2.loss_cls: 0.0335, s2.acc: 94.9176, s2.loss_bbox: 0.0581, loss: 0.5850
2022-11-12 15:06:59,218 - mmdet - INFO - Epoch [4][200/962]	lr: 2.000e-02, eta: 1:34:53, time: 0.688, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0119, loss_rpn_bbox: 0.0096, s0.loss_cls: 0.1407, s0.acc: 94.6777, s0.loss_bbox: 0.0992, s1.loss_cls: 0.0640, s1.acc: 95.2383, s1.loss_bbox: 0.0942, s2.loss_cls: 0.0299, s2.acc: 95.3145, s2.loss_bbox: 0.0530, loss: 0.5024
2022-11-12 15:07:33,737 - mmdet - INFO - Epoch [4][250/962]	lr: 2.000e-02, eta: 1:34:21, time: 0.690, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0135, loss_rpn_bbox: 0.0148, s0.loss_cls: 0.1603, s0.acc: 93.8301, s0.loss_bbox: 0.1127, s1.loss_cls: 0.0738, s1.acc: 94.4707, s1.loss_bbox: 0.1030, s2.loss_cls: 0.0331, s2.acc: 94.9727, s2.loss_bbox: 0.0563, loss: 0.5676
2022-11-12 15:08:07,992 - mmdet - INFO - Epoch [4][300/962]	lr: 2.000e-02, eta: 1:33:49, time: 0.685, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0130, loss_rpn_bbox: 0.0112, s0.loss_cls: 0.1477, s0.acc: 94.4531, s0.loss_bbox: 0.0951, s1.loss_cls: 0.0683, s1.acc: 94.9004, s1.loss_bbox: 0.0896, s2.loss_cls: 0.0316, s2.acc: 95.2207, s2.loss_bbox: 0.0533, loss: 0.5099
2022-11-12 15:08:42,610 - mmdet - INFO - Epoch [4][350/962]	lr: 2.000e-02, eta: 1:33:18, time: 0.692, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0103, loss_rpn_bbox: 0.0105, s0.loss_cls: 0.1598, s0.acc: 93.8262, s0.loss_bbox: 0.1116, s1.loss_cls: 0.0751, s1.acc: 94.4512, s1.loss_bbox: 0.1012, s2.loss_cls: 0.0341, s2.acc: 94.7734, s2.loss_bbox: 0.0563, loss: 0.5589
2022-11-12 15:09:17,141 - mmdet - INFO - Epoch [4][400/962]	lr: 2.000e-02, eta: 1:32:46, time: 0.691, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0107, loss_rpn_bbox: 0.0099, s0.loss_cls: 0.1384, s0.acc: 94.7090, s0.loss_bbox: 0.0913, s1.loss_cls: 0.0625, s1.acc: 95.3535, s1.loss_bbox: 0.0869, s2.loss_cls: 0.0288, s2.acc: 95.6680, s2.loss_bbox: 0.0495, loss: 0.4779
2022-11-12 15:09:51,165 - mmdet - INFO - Epoch [4][450/962]	lr: 2.000e-02, eta: 1:32:13, time: 0.680, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0092, loss_rpn_bbox: 0.0088, s0.loss_cls: 0.1371, s0.acc: 94.6230, s0.loss_bbox: 0.0953, s1.loss_cls: 0.0635, s1.acc: 95.3262, s1.loss_bbox: 0.0881, s2.loss_cls: 0.0289, s2.acc: 95.8379, s2.loss_bbox: 0.0512, loss: 0.4821
2022-11-12 15:10:25,709 - mmdet - INFO - Epoch [4][500/962]	lr: 2.000e-02, eta: 1:31:42, time: 0.691, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0118, loss_rpn_bbox: 0.0131, s0.loss_cls: 0.1539, s0.acc: 93.9277, s0.loss_bbox: 0.1106, s1.loss_cls: 0.0733, s1.acc: 94.2031, s1.loss_bbox: 0.1074, s2.loss_cls: 0.0354, s2.acc: 94.1953, s2.loss_bbox: 0.0588, loss: 0.5643
2022-11-12 15:11:00,392 - mmdet - INFO - Epoch [4][550/962]	lr: 2.000e-02, eta: 1:31:10, time: 0.694, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0168, loss_rpn_bbox: 0.0108, s0.loss_cls: 0.1526, s0.acc: 94.1484, s0.loss_bbox: 0.1132, s1.loss_cls: 0.0707, s1.acc: 94.5605, s1.loss_bbox: 0.1054, s2.loss_cls: 0.0324, s2.acc: 95.0723, s2.loss_bbox: 0.0543, loss: 0.5563
2022-11-12 15:11:34,032 - mmdet - INFO - Epoch [4][600/962]	lr: 2.000e-02, eta: 1:30:36, time: 0.673, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0146, loss_rpn_bbox: 0.0118, s0.loss_cls: 0.1593, s0.acc: 94.0996, s0.loss_bbox: 0.1129, s1.loss_cls: 0.0734, s1.acc: 94.7266, s1.loss_bbox: 0.0996, s2.loss_cls: 0.0323, s2.acc: 95.2441, s2.loss_bbox: 0.0557, loss: 0.5597
2022-11-12 15:12:08,065 - mmdet - INFO - Epoch [4][650/962]	lr: 2.000e-02, eta: 1:30:03, time: 0.681, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0157, loss_rpn_bbox: 0.0120, s0.loss_cls: 0.1633, s0.acc: 93.7539, s0.loss_bbox: 0.1144, s1.loss_cls: 0.0760, s1.acc: 94.0621, s1.loss_bbox: 0.1029, s2.loss_cls: 0.0347, s2.acc: 94.7398, s2.loss_bbox: 0.0554, loss: 0.5744
2022-11-12 15:12:41,597 - mmdet - INFO - Epoch [4][700/962]	lr: 2.000e-02, eta: 1:29:29, time: 0.671, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0142, loss_rpn_bbox: 0.0116, s0.loss_cls: 0.1562, s0.acc: 94.2129, s0.loss_bbox: 0.1133, s1.loss_cls: 0.0703, s1.acc: 94.7930, s1.loss_bbox: 0.1018, s2.loss_cls: 0.0327, s2.acc: 95.1348, s2.loss_bbox: 0.0559, loss: 0.5560
2022-11-12 15:13:15,753 - mmdet - INFO - Epoch [4][750/962]	lr: 2.000e-02, eta: 1:28:56, time: 0.683, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0106, loss_rpn_bbox: 0.0113, s0.loss_cls: 0.1478, s0.acc: 94.2383, s0.loss_bbox: 0.1037, s1.loss_cls: 0.0704, s1.acc: 94.6580, s1.loss_bbox: 0.0990, s2.loss_cls: 0.0323, s2.acc: 95.1133, s2.loss_bbox: 0.0537, loss: 0.5288
2022-11-12 15:13:49,788 - mmdet - INFO - Epoch [4][800/962]	lr: 2.000e-02, eta: 1:28:23, time: 0.681, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0124, loss_rpn_bbox: 0.0118, s0.loss_cls: 0.1588, s0.acc: 94.0176, s0.loss_bbox: 0.1114, s1.loss_cls: 0.0737, s1.acc: 94.3262, s1.loss_bbox: 0.1029, s2.loss_cls: 0.0333, s2.acc: 94.9902, s2.loss_bbox: 0.0581, loss: 0.5623
2022-11-12 15:14:23,435 - mmdet - INFO - Epoch [4][850/962]	lr: 2.000e-02, eta: 1:27:49, time: 0.673, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0131, loss_rpn_bbox: 0.0098, s0.loss_cls: 0.1514, s0.acc: 94.3047, s0.loss_bbox: 0.1102, s1.loss_cls: 0.0681, s1.acc: 94.7750, s1.loss_bbox: 0.1004, s2.loss_cls: 0.0311, s2.acc: 95.2420, s2.loss_bbox: 0.0580, loss: 0.5423
2022-11-12 15:14:57,893 - mmdet - INFO - Epoch [4][900/962]	lr: 2.000e-02, eta: 1:27:17, time: 0.689, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0100, loss_rpn_bbox: 0.0102, s0.loss_cls: 0.1323, s0.acc: 95.0703, s0.loss_bbox: 0.0865, s1.loss_cls: 0.0591, s1.acc: 95.5137, s1.loss_bbox: 0.0833, s2.loss_cls: 0.0270, s2.acc: 95.9219, s2.loss_bbox: 0.0480, loss: 0.4564
2022-11-12 15:15:32,171 - mmdet - INFO - Epoch [4][950/962]	lr: 2.000e-02, eta: 1:26:44, time: 0.686, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0120, loss_rpn_bbox: 0.0121, s0.loss_cls: 0.1649, s0.acc: 93.5762, s0.loss_bbox: 0.1234, s1.loss_cls: 0.0767, s1.acc: 93.9707, s1.loss_bbox: 0.1103, s2.loss_cls: 0.0346, s2.acc: 94.7617, s2.loss_bbox: 0.0592, loss: 0.5932
2022-11-12 15:15:40,685 - mmdet - INFO - Saving checkpoint at 4 epochs
[>>] 200/200, 6.4 task/s, elapsed: 31s, ETA:     0s2022-11-12 15:16:16,734 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.31s).
Accumulating evaluation results...
DONE (t=0.05s).
2022-11-12 15:16:17,109 - mmdet - INFO -
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.079
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.218
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.044
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.079
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.198
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.198
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.198
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.198

2022-11-12 15:16:17,113 - mmdet - INFO - Exp name: rcnn_r50_caffe_fpn_mstrain-poly_1x_COCO.py
2022-11-12 15:16:17,113 - mmdet - INFO - Epoch(val) [4][200]	bbox_mAP: 0.0790, bbox_mAP_50: 0.2180, bbox_mAP_75: 0.0440, bbox_mAP_s: -1.0000, bbox_mAP_m: -1.0000, bbox_mAP_l: 0.0790, bbox_mAP_copypaste: 0.079 0.218 0.044 -1.000 -1.000 0.079
2022-11-12 15:16:53,909 - mmdet - INFO - Epoch [5][50/962]	lr: 2.000e-02, eta: 1:25:52, time: 0.733, data_time: 0.060, memory: 3483, loss_rpn_cls: 0.0122, loss_rpn_bbox: 0.0113, s0.loss_cls: 0.1467, s0.acc: 94.5020, s0.loss_bbox: 0.1100, s1.loss_cls: 0.0635, s1.acc: 95.2207, s1.loss_bbox: 0.0973, s2.loss_cls: 0.0290, s2.acc: 95.6777, s2.loss_bbox: 0.0563, loss: 0.5263
2022-11-12 15:17:27,462 - mmdet - INFO - Epoch [5][100/962]	lr: 2.000e-02, eta: 1:25:18, time: 0.671, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0109, loss_rpn_bbox: 0.0089, s0.loss_cls: 0.1233, s0.acc: 95.3984, s0.loss_bbox: 0.0905, s1.loss_cls: 0.0544, s1.acc: 95.7891, s1.loss_bbox: 0.0841, s2.loss_cls: 0.0261, s2.acc: 95.8770, s2.loss_bbox: 0.0500, loss: 0.4481
2022-11-12 15:18:01,680 - mmdet - INFO - Epoch [5][150/962]	lr: 2.000e-02, eta: 1:24:45, time: 0.684, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0101, loss_rpn_bbox: 0.0097, s0.loss_cls: 0.1310, s0.acc: 95.0664, s0.loss_bbox: 0.0890, s1.loss_cls: 0.0614, s1.acc: 95.5268, s1.loss_bbox: 0.0857, s2.loss_cls: 0.0279, s2.acc: 95.9197, s2.loss_bbox: 0.0484, loss: 0.4633
2022-11-12 15:18:35,977 - mmdet - INFO - Epoch [5][200/962]	lr: 2.000e-02, eta: 1:24:13, time: 0.686, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0127, loss_rpn_bbox: 0.0122, s0.loss_cls: 0.1398, s0.acc: 94.5078, s0.loss_bbox: 0.1038, s1.loss_cls: 0.0626, s1.acc: 95.0547, s1.loss_bbox: 0.0957, s2.loss_cls: 0.0289, s2.acc: 95.6777, s2.loss_bbox: 0.0542, loss: 0.5100
2022-11-12 15:19:09,650 - mmdet - INFO - Epoch [5][250/962]	lr: 2.000e-02, eta: 1:23:39, time: 0.673, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0113, loss_rpn_bbox: 0.0126, s0.loss_cls: 0.1454, s0.acc: 94.2910, s0.loss_bbox: 0.1093, s1.loss_cls: 0.0654, s1.acc: 94.8704, s1.loss_bbox: 0.0999, s2.loss_cls: 0.0309, s2.acc: 95.1282, s2.loss_bbox: 0.0592, loss: 0.5340
2022-11-12 15:19:44,416 - mmdet - INFO - Epoch [5][300/962]	lr: 2.000e-02, eta: 1:23:07, time: 0.695, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0119, loss_rpn_bbox: 0.0105, s0.loss_cls: 0.1345, s0.acc: 94.6875, s0.loss_bbox: 0.0963, s1.loss_cls: 0.0618, s1.acc: 95.2070, s1.loss_bbox: 0.0911, s2.loss_cls: 0.0290, s2.acc: 95.5078, s2.loss_bbox: 0.0522, loss: 0.4872
2022-11-12 15:20:18,603 - mmdet - INFO - Epoch [5][350/962]	lr: 2.000e-02, eta: 1:22:34, time: 0.684, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0118, loss_rpn_bbox: 0.0115, s0.loss_cls: 0.1410, s0.acc: 94.6641, s0.loss_bbox: 0.0970, s1.loss_cls: 0.0642, s1.acc: 94.9941, s1.loss_bbox: 0.0911, s2.loss_cls: 0.0292, s2.acc: 95.4766, s2.loss_bbox: 0.0516, loss: 0.4974
2022-11-12 15:20:52,198 - mmdet - INFO - Epoch [5][400/962]	lr: 2.000e-02, eta: 1:22:00, time: 0.672, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0115, loss_rpn_bbox: 0.0108, s0.loss_cls: 0.1259, s0.acc: 95.2637, s0.loss_bbox: 0.0905, s1.loss_cls: 0.0579, s1.acc: 95.8047, s1.loss_bbox: 0.0829, s2.loss_cls: 0.0272, s2.acc: 96.0059, s2.loss_bbox: 0.0485, loss: 0.4552
2022-11-12 15:21:26,392 - mmdet - INFO - Epoch [5][450/962]	lr: 2.000e-02, eta: 1:21:27, time: 0.684, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0134, loss_rpn_bbox: 0.0103, s0.loss_cls: 0.1372, s0.acc: 94.7949, s0.loss_bbox: 0.0938, s1.loss_cls: 0.0620, s1.acc: 95.4350, s1.loss_bbox: 0.0912, s2.loss_cls: 0.0294, s2.acc: 95.6185, s2.loss_bbox: 0.0563, loss: 0.4936
2022-11-12 15:21:59,927 - mmdet - INFO - Epoch [5][500/962]	lr: 2.000e-02, eta: 1:20:53, time: 0.671, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0106, loss_rpn_bbox: 0.0082, s0.loss_cls: 0.1275, s0.acc: 95.0254, s0.loss_bbox: 0.0904, s1.loss_cls: 0.0575, s1.acc: 95.5430, s1.loss_bbox: 0.0898, s2.loss_cls: 0.0271, s2.acc: 95.5820, s2.loss_bbox: 0.0551, loss: 0.4663
2022-11-12 15:22:33,972 - mmdet - INFO - Epoch [5][550/962]	lr: 2.000e-02, eta: 1:20:20, time: 0.681, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0154, loss_rpn_bbox: 0.0095, s0.loss_cls: 0.1328, s0.acc: 94.8809, s0.loss_bbox: 0.0914, s1.loss_cls: 0.0599, s1.acc: 95.3540, s1.loss_bbox: 0.0858, s2.loss_cls: 0.0273, s2.acc: 95.8874, s2.loss_bbox: 0.0502, loss: 0.4721
2022-11-12 15:23:08,191 - mmdet - INFO - Epoch [5][600/962]	lr: 2.000e-02, eta: 1:19:47, time: 0.684, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0106, loss_rpn_bbox: 0.0090, s0.loss_cls: 0.1245, s0.acc: 95.1777, s0.loss_bbox: 0.0867, s1.loss_cls: 0.0563, s1.acc: 95.7266, s1.loss_bbox: 0.0848, s2.loss_cls: 0.0262, s2.acc: 96.0020, s2.loss_bbox: 0.0499, loss: 0.4479
2022-11-12 15:23:42,400 - mmdet - INFO - Epoch [5][650/962]	lr: 2.000e-02, eta: 1:19:14, time: 0.684, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0085, loss_rpn_bbox: 0.0088, s0.loss_cls: 0.1377, s0.acc: 94.5547, s0.loss_bbox: 0.0970, s1.loss_cls: 0.0623, s1.acc: 95.1328, s1.loss_bbox: 0.0920, s2.loss_cls: 0.0290, s2.acc: 95.4766, s2.loss_bbox: 0.0544, loss: 0.4896
2022-11-12 15:24:16,533 - mmdet - INFO - Epoch [5][700/962]	lr: 2.000e-02, eta: 1:18:41, time: 0.683, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0097, loss_rpn_bbox: 0.0088, s0.loss_cls: 0.1323, s0.acc: 94.9648, s0.loss_bbox: 0.0951, s1.loss_cls: 0.0586, s1.acc: 95.5469, s1.loss_bbox: 0.0912, s2.loss_cls: 0.0268, s2.acc: 95.8809, s2.loss_bbox: 0.0560, loss: 0.4786
2022-11-12 15:24:50,151 - mmdet - INFO - Epoch [5][750/962]	lr: 2.000e-02, eta: 1:18:07, time: 0.672, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0105, loss_rpn_bbox: 0.0124, s0.loss_cls: 0.1363, s0.acc: 94.6953, s0.loss_bbox: 0.0952, s1.loss_cls: 0.0646, s1.acc: 95.2539, s1.loss_bbox: 0.0876, s2.loss_cls: 0.0305, s2.acc: 95.2402, s2.loss_bbox: 0.0501, loss: 0.4872
2022-11-12 15:25:24,134 - mmdet - INFO - Epoch [5][800/962]	lr: 2.000e-02, eta: 1:17:34, time: 0.680, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0086, loss_rpn_bbox: 0.0087, s0.loss_cls: 0.1238, s0.acc: 95.2344, s0.loss_bbox: 0.0852, s1.loss_cls: 0.0566, s1.acc: 95.6172, s1.loss_bbox: 0.0827, s2.loss_cls: 0.0266, s2.acc: 95.8320, s2.loss_bbox: 0.0494, loss: 0.4416
2022-11-12 15:25:58,135 - mmdet - INFO - Epoch [5][850/962]	lr: 2.000e-02, eta: 1:17:00, time: 0.680, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0099, loss_rpn_bbox: 0.0104, s0.loss_cls: 0.1433, s0.acc: 94.5684, s0.loss_bbox: 0.0975, s1.loss_cls: 0.0691, s1.acc: 94.5684, s1.loss_bbox: 0.0928, s2.loss_cls: 0.0325, s2.acc: 95.0723, s2.loss_bbox: 0.0533, loss: 0.5088
2022-11-12 15:26:31,661 - mmdet - INFO - Epoch [5][900/962]	lr: 2.000e-02, eta: 1:16:26, time: 0.670, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0141, loss_rpn_bbox: 0.0115, s0.loss_cls: 0.1409, s0.acc: 94.8906, s0.loss_bbox: 0.0922, s1.loss_cls: 0.0639, s1.acc: 95.4238, s1.loss_bbox: 0.0875, s2.loss_cls: 0.0305, s2.acc: 95.4434, s2.loss_bbox: 0.0496, loss: 0.4901
2022-11-12 15:27:05,849 - mmdet - INFO - Epoch [5][950/962]	lr: 2.000e-02, eta: 1:15:53, time: 0.684, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0105, loss_rpn_bbox: 0.0102, s0.loss_cls: 0.1455, s0.acc: 94.3086, s0.loss_bbox: 0.1052, s1.loss_cls: 0.0658, s1.acc: 94.8770, s1.loss_bbox: 0.1008, s2.loss_cls: 0.0300, s2.acc: 95.3359, s2.loss_bbox: 0.0543, loss: 0.5223
2022-11-12 15:27:13,957 - mmdet - INFO - Saving checkpoint at 5 epochs
[>>] 200/200, 6.0 task/s, elapsed: 33s, ETA:     0s2022-11-12 15:27:52,132 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.34s).
Accumulating evaluation results...
DONE (t=0.06s).
2022-11-12 15:27:52,547 - mmdet - INFO -
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.109
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.270
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.080
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.109
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.216
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.216
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.216
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.216

2022-11-12 15:27:52,550 - mmdet - INFO - Exp name: rcnn_r50_caffe_fpn_mstrain-poly_1x_COCO.py
2022-11-12 15:27:52,550 - mmdet - INFO - Epoch(val) [5][200]	bbox_mAP: 0.1090, bbox_mAP_50: 0.2700, bbox_mAP_75: 0.0800, bbox_mAP_s: -1.0000, bbox_mAP_m: -1.0000, bbox_mAP_l: 0.1090, bbox_mAP_copypaste: 0.109 0.270 0.080 -1.000 -1.000 0.109
2022-11-12 15:28:28,661 - mmdet - INFO - Epoch [6][50/962]	lr: 2.000e-02, eta: 1:15:03, time: 0.720, data_time: 0.059, memory: 3483, loss_rpn_cls: 0.0087, loss_rpn_bbox: 0.0083, s0.loss_cls: 0.1175, s0.acc: 95.2812, s0.loss_bbox: 0.0826, s1.loss_cls: 0.0536, s1.acc: 95.7168, s1.loss_bbox: 0.0792, s2.loss_cls: 0.0251, s2.acc: 95.8340, s2.loss_bbox: 0.0452, loss: 0.4203
2022-11-12 15:29:02,712 - mmdet - INFO - Epoch [6][100/962]	lr: 2.000e-02, eta: 1:14:30, time: 0.681, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0084, loss_rpn_bbox: 0.0107, s0.loss_cls: 0.1255, s0.acc: 95.0645, s0.loss_bbox: 0.0987, s1.loss_cls: 0.0570, s1.acc: 95.4766, s1.loss_bbox: 0.0964, s2.loss_cls: 0.0268, s2.acc: 95.6055, s2.loss_bbox: 0.0559, loss: 0.4792
2022-11-12 15:29:36,752 - mmdet - INFO - Epoch [6][150/962]	lr: 2.000e-02, eta: 1:13:57, time: 0.681, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0100, loss_rpn_bbox: 0.0110, s0.loss_cls: 0.1200, s0.acc: 95.4883, s0.loss_bbox: 0.0861, s1.loss_cls: 0.0520, s1.acc: 96.0742, s1.loss_bbox: 0.0869, s2.loss_cls: 0.0243, s2.acc: 96.3145, s2.loss_bbox: 0.0528, loss: 0.4433
2022-11-12 15:30:10,410 - mmdet - INFO - Epoch [6][200/962]	lr: 2.000e-02, eta: 1:13:23, time: 0.673, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0088, loss_rpn_bbox: 0.0105, s0.loss_cls: 0.1259, s0.acc: 95.1484, s0.loss_bbox: 0.0923, s1.loss_cls: 0.0547, s1.acc: 95.8496, s1.loss_bbox: 0.0903, s2.loss_cls: 0.0275, s2.acc: 95.7285, s2.loss_bbox: 0.0546, loss: 0.4647
2022-11-12 15:30:44,559 - mmdet - INFO - Epoch [6][250/962]	lr: 2.000e-02, eta: 1:12:50, time: 0.683, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0122, loss_rpn_bbox: 0.0099, s0.loss_cls: 0.1149, s0.acc: 95.6211, s0.loss_bbox: 0.0806, s1.loss_cls: 0.0514, s1.acc: 96.0781, s1.loss_bbox: 0.0788, s2.loss_cls: 0.0249, s2.acc: 96.1973, s2.loss_bbox: 0.0526, loss: 0.4252
2022-11-12 15:31:18,180 - mmdet - INFO - Epoch [6][300/962]	lr: 2.000e-02, eta: 1:12:16, time: 0.672, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0087, loss_rpn_bbox: 0.0108, s0.loss_cls: 0.1162, s0.acc: 95.4629, s0.loss_bbox: 0.0827, s1.loss_cls: 0.0533, s1.acc: 95.7891, s1.loss_bbox: 0.0823, s2.loss_cls: 0.0249, s2.acc: 96.2520, s2.loss_bbox: 0.0499, loss: 0.4289
2022-11-12 15:31:52,890 - mmdet - INFO - Epoch [6][350/962]	lr: 2.000e-02, eta: 1:11:44, time: 0.694, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0089, loss_rpn_bbox: 0.0104, s0.loss_cls: 0.1257, s0.acc: 95.0801, s0.loss_bbox: 0.0908, s1.loss_cls: 0.0587, s1.acc: 95.4753, s1.loss_bbox: 0.0904, s2.loss_cls: 0.0278, s2.acc: 95.6857, s2.loss_bbox: 0.0555, loss: 0.4682
2022-11-12 15:32:27,192 - mmdet - INFO - Epoch [6][400/962]	lr: 2.000e-02, eta: 1:11:11, time: 0.686, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0092, loss_rpn_bbox: 0.0087, s0.loss_cls: 0.1241, s0.acc: 95.3301, s0.loss_bbox: 0.0880, s1.loss_cls: 0.0565, s1.acc: 95.5745, s1.loss_bbox: 0.0888, s2.loss_cls: 0.0271, s2.acc: 95.7777, s2.loss_bbox: 0.0545, loss: 0.4570
2022-11-12 15:33:00,963 - mmdet - INFO - Epoch [6][450/962]	lr: 2.000e-02, eta: 1:10:37, time: 0.675, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0116, loss_rpn_bbox: 0.0113, s0.loss_cls: 0.1390, s0.acc: 94.9336, s0.loss_bbox: 0.0942, s1.loss_cls: 0.0630, s1.acc: 95.2707, s1.loss_bbox: 0.0860, s2.loss_cls: 0.0290, s2.acc: 95.5679, s2.loss_bbox: 0.0510, loss: 0.4852
2022-11-12 15:33:35,204 - mmdet - INFO - Epoch [6][500/962]	lr: 2.000e-02, eta: 1:10:04, time: 0.685, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0118, loss_rpn_bbox: 0.0100, s0.loss_cls: 0.1378, s0.acc: 94.7910, s0.loss_bbox: 0.0975, s1.loss_cls: 0.0621, s1.acc: 95.0962, s1.loss_bbox: 0.0910, s2.loss_cls: 0.0294, s2.acc: 95.4803, s2.loss_bbox: 0.0541, loss: 0.4938
2022-11-12 15:34:08,962 - mmdet - INFO - Epoch [6][550/962]	lr: 2.000e-02, eta: 1:09:30, time: 0.675, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0124, loss_rpn_bbox: 0.0118, s0.loss_cls: 0.1347, s0.acc: 94.8516, s0.loss_bbox: 0.0925, s1.loss_cls: 0.0629, s1.acc: 95.0291, s1.loss_bbox: 0.0919, s2.loss_cls: 0.0300, s2.acc: 95.2458, s2.loss_bbox: 0.0543, loss: 0.4905
2022-11-12 15:34:42,873 - mmdet - INFO - Epoch [6][600/962]	lr: 2.000e-02, eta: 1:08:57, time: 0.678, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0079, loss_rpn_bbox: 0.0075, s0.loss_cls: 0.1014, s0.acc: 96.0391, s0.loss_bbox: 0.0732, s1.loss_cls: 0.0445, s1.acc: 96.5352, s1.loss_bbox: 0.0750, s2.loss_cls: 0.0214, s2.acc: 96.5527, s2.loss_bbox: 0.0472, loss: 0.3781
2022-11-12 15:35:16,998 - mmdet - INFO - Epoch [6][650/962]	lr: 2.000e-02, eta: 1:08:24, time: 0.683, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0112, loss_rpn_bbox: 0.0105, s0.loss_cls: 0.1331, s0.acc: 94.9062, s0.loss_bbox: 0.0944, s1.loss_cls: 0.0611, s1.acc: 95.3125, s1.loss_bbox: 0.0931, s2.loss_cls: 0.0283, s2.acc: 95.7363, s2.loss_bbox: 0.0536, loss: 0.4853
2022-11-12 15:35:50,614 - mmdet - INFO - Epoch [6][700/962]	lr: 2.000e-02, eta: 1:07:50, time: 0.672, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0101, loss_rpn_bbox: 0.0107, s0.loss_cls: 0.1160, s0.acc: 95.4238, s0.loss_bbox: 0.0902, s1.loss_cls: 0.0545, s1.acc: 95.6325, s1.loss_bbox: 0.0869, s2.loss_cls: 0.0270, s2.acc: 95.6442, s2.loss_bbox: 0.0518, loss: 0.4472
2022-11-12 15:36:25,103 - mmdet - INFO - Epoch [6][750/962]	lr: 2.000e-02, eta: 1:07:17, time: 0.690, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0096, loss_rpn_bbox: 0.0098, s0.loss_cls: 0.1148, s0.acc: 95.5137, s0.loss_bbox: 0.0818, s1.loss_cls: 0.0530, s1.acc: 95.9136, s1.loss_bbox: 0.0758, s2.loss_cls: 0.0247, s2.acc: 96.2301, s2.loss_bbox: 0.0466, loss: 0.4161
2022-11-12 15:36:59,102 - mmdet - INFO - Epoch [6][800/962]	lr: 2.000e-02, eta: 1:06:43, time: 0.680, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0102, loss_rpn_bbox: 0.0095, s0.loss_cls: 0.1089, s0.acc: 95.7598, s0.loss_bbox: 0.0772, s1.loss_cls: 0.0499, s1.acc: 96.1836, s1.loss_bbox: 0.0814, s2.loss_cls: 0.0242, s2.acc: 96.2168, s2.loss_bbox: 0.0505, loss: 0.4117
2022-11-12 15:37:32,822 - mmdet - INFO - Epoch [6][850/962]	lr: 2.000e-02, eta: 1:06:10, time: 0.674, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0116, loss_rpn_bbox: 0.0104, s0.loss_cls: 0.1251, s0.acc: 95.2227, s0.loss_bbox: 0.0939, s1.loss_cls: 0.0555, s1.acc: 95.8486, s1.loss_bbox: 0.0885, s2.loss_cls: 0.0259, s2.acc: 96.1165, s2.loss_bbox: 0.0524, loss: 0.4632
2022-11-12 15:38:06,948 - mmdet - INFO - Epoch [6][900/962]	lr: 2.000e-02, eta: 1:05:36, time: 0.683, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0096, loss_rpn_bbox: 0.0113, s0.loss_cls: 0.1227, s0.acc: 95.3223, s0.loss_bbox: 0.0902, s1.loss_cls: 0.0558, s1.acc: 95.5779, s1.loss_bbox: 0.0924, s2.loss_cls: 0.0272, s2.acc: 95.8051, s2.loss_bbox: 0.0551, loss: 0.4644
2022-11-12 15:38:40,573 - mmdet - INFO - Epoch [6][950/962]	lr: 2.000e-02, eta: 1:05:02, time: 0.672, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0122, loss_rpn_bbox: 0.0097, s0.loss_cls: 0.1441, s0.acc: 94.6680, s0.loss_bbox: 0.0908, s1.loss_cls: 0.0669, s1.acc: 95.0800, s1.loss_bbox: 0.0851, s2.loss_cls: 0.0305, s2.acc: 95.5253, s2.loss_bbox: 0.0516, loss: 0.4909
2022-11-12 15:38:48,737 - mmdet - INFO - Saving checkpoint at 6 epochs
[>>] 200/200, 6.4 task/s, elapsed: 31s, ETA:     0s2022-11-12 15:39:25,733 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.27s).
Accumulating evaluation results...
DONE (t=0.06s).
2022-11-12 15:39:26,077 - mmdet - INFO -
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.107
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.268
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.090
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.107
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.208
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.208
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.208
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.208

2022-11-12 15:39:26,081 - mmdet - INFO - Exp name: rcnn_r50_caffe_fpn_mstrain-poly_1x_COCO.py
2022-11-12 15:39:26,081 - mmdet - INFO - Epoch(val) [6][200]	bbox_mAP: 0.1070, bbox_mAP_50: 0.2680, bbox_mAP_75: 0.0900, bbox_mAP_s: -1.0000, bbox_mAP_m: -1.0000, bbox_mAP_l: 0.1070, bbox_mAP_copypaste: 0.107 0.268 0.090 -1.000 -1.000 0.107
2022-11-12 15:40:02,711 - mmdet - INFO - Epoch [7][50/962]	lr: 2.000e-02, eta: 1:04:15, time: 0.730, data_time: 0.060, memory: 3483, loss_rpn_cls: 0.0096, loss_rpn_bbox: 0.0079, s0.loss_cls: 0.1195, s0.acc: 95.3867, s0.loss_bbox: 0.0830, s1.loss_cls: 0.0560, s1.acc: 95.5195, s1.loss_bbox: 0.0845, s2.loss_cls: 0.0278, s2.acc: 95.5684, s2.loss_bbox: 0.0531, loss: 0.4414
2022-11-12 15:40:36,880 - mmdet - INFO - Epoch [7][100/962]	lr: 2.000e-02, eta: 1:03:42, time: 0.683, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0101, loss_rpn_bbox: 0.0104, s0.loss_cls: 0.1348, s0.acc: 94.7246, s0.loss_bbox: 0.0899, s1.loss_cls: 0.0659, s1.acc: 94.8672, s1.loss_bbox: 0.0920, s2.loss_cls: 0.0313, s2.acc: 95.2012, s2.loss_bbox: 0.0532, loss: 0.4876
2022-11-12 15:41:10,951 - mmdet - INFO - Epoch [7][150/962]	lr: 2.000e-02, eta: 1:03:09, time: 0.681, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0081, loss_rpn_bbox: 0.0073, s0.loss_cls: 0.1037, s0.acc: 95.9707, s0.loss_bbox: 0.0727, s1.loss_cls: 0.0510, s1.acc: 95.9018, s1.loss_bbox: 0.0797, s2.loss_cls: 0.0252, s2.acc: 95.8512, s2.loss_bbox: 0.0501, loss: 0.3979
2022-11-12 15:41:44,915 - mmdet - INFO - Epoch [7][200/962]	lr: 2.000e-02, eta: 1:02:35, time: 0.679, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0110, loss_rpn_bbox: 0.0105, s0.loss_cls: 0.1146, s0.acc: 95.5059, s0.loss_bbox: 0.0869, s1.loss_cls: 0.0504, s1.acc: 96.1387, s1.loss_bbox: 0.0850, s2.loss_cls: 0.0243, s2.acc: 96.1797, s2.loss_bbox: 0.0514, loss: 0.4341
2022-11-12 15:42:18,465 - mmdet - INFO - Epoch [7][250/962]	lr: 2.000e-02, eta: 1:02:01, time: 0.671, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0096, loss_rpn_bbox: 0.0095, s0.loss_cls: 0.1119, s0.acc: 95.5723, s0.loss_bbox: 0.0828, s1.loss_cls: 0.0517, s1.acc: 95.7656, s1.loss_bbox: 0.0813, s2.loss_cls: 0.0254, s2.acc: 95.9688, s2.loss_bbox: 0.0527, loss: 0.4249
2022-11-12 15:42:52,515 - mmdet - INFO - Epoch [7][300/962]	lr: 2.000e-02, eta: 1:01:28, time: 0.681, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0081, loss_rpn_bbox: 0.0082, s0.loss_cls: 0.1083, s0.acc: 95.8789, s0.loss_bbox: 0.0776, s1.loss_cls: 0.0487, s1.acc: 96.3687, s1.loss_bbox: 0.0788, s2.loss_cls: 0.0230, s2.acc: 96.4017, s2.loss_bbox: 0.0485, loss: 0.4012
2022-11-12 15:43:26,081 - mmdet - INFO - Epoch [7][350/962]	lr: 2.000e-02, eta: 1:00:54, time: 0.671, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0105, loss_rpn_bbox: 0.0103, s0.loss_cls: 0.1220, s0.acc: 95.3184, s0.loss_bbox: 0.0862, s1.loss_cls: 0.0547, s1.acc: 95.6512, s1.loss_bbox: 0.0866, s2.loss_cls: 0.0245, s2.acc: 96.1926, s2.loss_bbox: 0.0490, loss: 0.4439
2022-11-12 15:44:00,154 - mmdet - INFO - Epoch [7][400/962]	lr: 2.000e-02, eta: 1:00:21, time: 0.681, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0097, loss_rpn_bbox: 0.0101, s0.loss_cls: 0.1197, s0.acc: 95.4277, s0.loss_bbox: 0.0858, s1.loss_cls: 0.0533, s1.acc: 95.8975, s1.loss_bbox: 0.0847, s2.loss_cls: 0.0247, s2.acc: 96.2002, s2.loss_bbox: 0.0501, loss: 0.4381
2022-11-12 15:44:34,640 - mmdet - INFO - Epoch [7][450/962]	lr: 2.000e-02, eta: 0:59:48, time: 0.690, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0099, loss_rpn_bbox: 0.0103, s0.loss_cls: 0.1276, s0.acc: 94.8828, s0.loss_bbox: 0.0900, s1.loss_cls: 0.0616, s1.acc: 95.2560, s1.loss_bbox: 0.0883, s2.loss_cls: 0.0296, s2.acc: 95.3991, s2.loss_bbox: 0.0542, loss: 0.4714
2022-11-12 15:45:08,125 - mmdet - INFO - Epoch [7][500/962]	lr: 2.000e-02, eta: 0:59:14, time: 0.670, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0098, loss_rpn_bbox: 0.0091, s0.loss_cls: 0.1194, s0.acc: 95.3965, s0.loss_bbox: 0.0815, s1.loss_cls: 0.0542, s1.acc: 95.7476, s1.loss_bbox: 0.0777, s2.loss_cls: 0.0260, s2.acc: 95.8064, s2.loss_bbox: 0.0497, loss: 0.4272
2022-11-12 15:45:42,161 - mmdet - INFO - Epoch [7][550/962]	lr: 2.000e-02, eta: 0:58:40, time: 0.681, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0072, loss_rpn_bbox: 0.0065, s0.loss_cls: 0.1017, s0.acc: 96.1191, s0.loss_bbox: 0.0724, s1.loss_cls: 0.0475, s1.acc: 96.3645, s1.loss_bbox: 0.0710, s2.loss_cls: 0.0231, s2.acc: 96.5326, s2.loss_bbox: 0.0425, loss: 0.3719
2022-11-12 15:46:15,665 - mmdet - INFO - Epoch [7][600/962]	lr: 2.000e-02, eta: 0:58:07, time: 0.670, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0111, loss_rpn_bbox: 0.0094, s0.loss_cls: 0.1181, s0.acc: 95.4707, s0.loss_bbox: 0.0851, s1.loss_cls: 0.0546, s1.acc: 95.7207, s1.loss_bbox: 0.0859, s2.loss_cls: 0.0261, s2.acc: 96.0879, s2.loss_bbox: 0.0514, loss: 0.4416
2022-11-12 15:46:49,737 - mmdet - INFO - Epoch [7][650/962]	lr: 2.000e-02, eta: 0:57:33, time: 0.681, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0094, loss_rpn_bbox: 0.0087, s0.loss_cls: 0.1192, s0.acc: 95.2598, s0.loss_bbox: 0.0813, s1.loss_cls: 0.0550, s1.acc: 95.6519, s1.loss_bbox: 0.0817, s2.loss_cls: 0.0257, s2.acc: 96.0599, s2.loss_bbox: 0.0504, loss: 0.4314
2022-11-12 15:47:23,695 - mmdet - INFO - Epoch [7][700/962]	lr: 2.000e-02, eta: 0:57:00, time: 0.679, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0082, loss_rpn_bbox: 0.0091, s0.loss_cls: 0.1145, s0.acc: 95.6211, s0.loss_bbox: 0.0854, s1.loss_cls: 0.0519, s1.acc: 95.9661, s1.loss_bbox: 0.0783, s2.loss_cls: 0.0250, s2.acc: 96.1243, s2.loss_bbox: 0.0498, loss: 0.4221
2022-11-12 15:47:56,911 - mmdet - INFO - Epoch [7][750/962]	lr: 2.000e-02, eta: 0:56:25, time: 0.664, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0093, loss_rpn_bbox: 0.0115, s0.loss_cls: 0.1094, s0.acc: 95.7227, s0.loss_bbox: 0.0834, s1.loss_cls: 0.0497, s1.acc: 96.0466, s1.loss_bbox: 0.0838, s2.loss_cls: 0.0228, s2.acc: 96.3476, s2.loss_bbox: 0.0515, loss: 0.4213
2022-11-12 15:48:30,916 - mmdet - INFO - Epoch [7][800/962]	lr: 2.000e-02, eta: 0:55:52, time: 0.680, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0118, loss_rpn_bbox: 0.0083, s0.loss_cls: 0.0988, s0.acc: 96.2930, s0.loss_bbox: 0.0741, s1.loss_cls: 0.0436, s1.acc: 96.7012, s1.loss_bbox: 0.0780, s2.loss_cls: 0.0224, s2.acc: 96.6074, s2.loss_bbox: 0.0485, loss: 0.3855
2022-11-12 15:49:05,021 - mmdet - INFO - Epoch [7][850/962]	lr: 2.000e-02, eta: 0:55:19, time: 0.682, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0088, loss_rpn_bbox: 0.0105, s0.loss_cls: 0.1168, s0.acc: 95.5430, s0.loss_bbox: 0.0795, s1.loss_cls: 0.0514, s1.acc: 95.9430, s1.loss_bbox: 0.0756, s2.loss_cls: 0.0243, s2.acc: 96.2399, s2.loss_bbox: 0.0453, loss: 0.4121
2022-11-12 15:49:39,194 - mmdet - INFO - Epoch [7][900/962]	lr: 2.000e-02, eta: 0:54:45, time: 0.683, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0096, loss_rpn_bbox: 0.0089, s0.loss_cls: 0.1233, s0.acc: 95.4258, s0.loss_bbox: 0.0834, s1.loss_cls: 0.0560, s1.acc: 95.8633, s1.loss_bbox: 0.0821, s2.loss_cls: 0.0262, s2.acc: 96.0234, s2.loss_bbox: 0.0496, loss: 0.4392
2022-11-12 15:50:13,290 - mmdet - INFO - Epoch [7][950/962]	lr: 2.000e-02, eta: 0:54:12, time: 0.682, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0084, loss_rpn_bbox: 0.0103, s0.loss_cls: 0.1201, s0.acc: 95.4297, s0.loss_bbox: 0.0844, s1.loss_cls: 0.0521, s1.acc: 96.1230, s1.loss_bbox: 0.0797, s2.loss_cls: 0.0243, s2.acc: 96.2871, s2.loss_bbox: 0.0518, loss: 0.4310
2022-11-12 15:50:21,381 - mmdet - INFO - Saving checkpoint at 7 epochs
[>>] 200/200, 6.4 task/s, elapsed: 31s, ETA:     0s2022-11-12 15:50:57,209 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.47s).
Accumulating evaluation results...
DONE (t=0.05s).
2022-11-12 15:50:57,749 - mmdet - INFO -
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.111
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.283
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.070
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.111
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.220
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.220
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.220
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.220

2022-11-12 15:50:57,752 - mmdet - INFO - Exp name: rcnn_r50_caffe_fpn_mstrain-poly_1x_COCO.py
2022-11-12 15:50:57,752 - mmdet - INFO - Epoch(val) [7][200]	bbox_mAP: 0.1110, bbox_mAP_50: 0.2830, bbox_mAP_75: 0.0700, bbox_mAP_s: -1.0000, bbox_mAP_m: -1.0000, bbox_mAP_l: 0.1110, bbox_mAP_copypaste: 0.111 0.283 0.070 -1.000 -1.000 0.111
2022-11-12 15:51:34,311 - mmdet - INFO - Epoch [8][50/962]	lr: 2.000e-02, eta: 0:53:26, time: 0.729, data_time: 0.059, memory: 3483, loss_rpn_cls: 0.0105, loss_rpn_bbox: 0.0102, s0.loss_cls: 0.1114, s0.acc: 95.6895, s0.loss_bbox: 0.0753, s1.loss_cls: 0.0522, s1.acc: 95.9948, s1.loss_bbox: 0.0754, s2.loss_cls: 0.0241, s2.acc: 96.3031, s2.loss_bbox: 0.0511, loss: 0.4102
2022-11-12 15:52:08,323 - mmdet - INFO - Epoch [8][100/962]	lr: 2.000e-02, eta: 0:52:53, time: 0.680, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0094, loss_rpn_bbox: 0.0089, s0.loss_cls: 0.0967, s0.acc: 96.1250, s0.loss_bbox: 0.0724, s1.loss_cls: 0.0419, s1.acc: 96.7227, s1.loss_bbox: 0.0711, s2.loss_cls: 0.0204, s2.acc: 96.9297, s2.loss_bbox: 0.0435, loss: 0.3643
2022-11-12 15:52:42,354 - mmdet - INFO - Epoch [8][150/962]	lr: 2.000e-02, eta: 0:52:19, time: 0.681, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0094, loss_rpn_bbox: 0.0096, s0.loss_cls: 0.1109, s0.acc: 95.6602, s0.loss_bbox: 0.0774, s1.loss_cls: 0.0496, s1.acc: 96.2324, s1.loss_bbox: 0.0754, s2.loss_cls: 0.0226, s2.acc: 96.4121, s2.loss_bbox: 0.0452, loss: 0.4001
2022-11-12 15:53:16,284 - mmdet - INFO - Epoch [8][200/962]	lr: 2.000e-02, eta: 0:51:46, time: 0.679, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0088, loss_rpn_bbox: 0.0092, s0.loss_cls: 0.1076, s0.acc: 95.9551, s0.loss_bbox: 0.0773, s1.loss_cls: 0.0490, s1.acc: 96.1372, s1.loss_bbox: 0.0801, s2.loss_cls: 0.0240, s2.acc: 96.1469, s2.loss_bbox: 0.0514, loss: 0.4074
2022-11-12 15:53:49,773 - mmdet - INFO - Epoch [8][250/962]	lr: 2.000e-02, eta: 0:51:12, time: 0.670, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0068, loss_rpn_bbox: 0.0081, s0.loss_cls: 0.1056, s0.acc: 95.8066, s0.loss_bbox: 0.0795, s1.loss_cls: 0.0493, s1.acc: 96.0049, s1.loss_bbox: 0.0747, s2.loss_cls: 0.0237, s2.acc: 96.1970, s2.loss_bbox: 0.0456, loss: 0.3933
2022-11-12 15:54:23,723 - mmdet - INFO - Epoch [8][300/962]	lr: 2.000e-02, eta: 0:50:38, time: 0.679, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0121, loss_rpn_bbox: 0.0103, s0.loss_cls: 0.1060, s0.acc: 95.8555, s0.loss_bbox: 0.0769, s1.loss_cls: 0.0450, s1.acc: 96.4659, s1.loss_bbox: 0.0711, s2.loss_cls: 0.0214, s2.acc: 96.6145, s2.loss_bbox: 0.0454, loss: 0.3883
2022-11-12 15:54:57,854 - mmdet - INFO - Epoch [8][350/962]	lr: 2.000e-02, eta: 0:50:05, time: 0.683, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0079, loss_rpn_bbox: 0.0117, s0.loss_cls: 0.1162, s0.acc: 95.5488, s0.loss_bbox: 0.0852, s1.loss_cls: 0.0524, s1.acc: 95.9665, s1.loss_bbox: 0.0793, s2.loss_cls: 0.0250, s2.acc: 96.2479, s2.loss_bbox: 0.0496, loss: 0.4273
2022-11-12 15:55:31,396 - mmdet - INFO - Epoch [8][400/962]	lr: 2.000e-02, eta: 0:49:31, time: 0.671, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0124, loss_rpn_bbox: 0.0107, s0.loss_cls: 0.1295, s0.acc: 95.0840, s0.loss_bbox: 0.0943, s1.loss_cls: 0.0597, s1.acc: 95.3520, s1.loss_bbox: 0.0945, s2.loss_cls: 0.0285, s2.acc: 95.6325, s2.loss_bbox: 0.0562, loss: 0.4856
2022-11-12 15:56:05,397 - mmdet - INFO - Epoch [8][450/962]	lr: 2.000e-02, eta: 0:48:58, time: 0.680, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0101, loss_rpn_bbox: 0.0085, s0.loss_cls: 0.0968, s0.acc: 96.2480, s0.loss_bbox: 0.0722, s1.loss_cls: 0.0442, s1.acc: 96.5863, s1.loss_bbox: 0.0684, s2.loss_cls: 0.0216, s2.acc: 96.5423, s2.loss_bbox: 0.0436, loss: 0.3655
2022-11-12 15:56:39,069 - mmdet - INFO - Epoch [8][500/962]	lr: 2.000e-02, eta: 0:48:24, time: 0.673, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0065, loss_rpn_bbox: 0.0065, s0.loss_cls: 0.1112, s0.acc: 95.8086, s0.loss_bbox: 0.0732, s1.loss_cls: 0.0496, s1.acc: 96.3409, s1.loss_bbox: 0.0762, s2.loss_cls: 0.0238, s2.acc: 96.4033, s2.loss_bbox: 0.0504, loss: 0.3974
2022-11-12 15:57:13,619 - mmdet - INFO - Epoch [8][550/962]	lr: 2.000e-02, eta: 0:47:51, time: 0.691, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0086, loss_rpn_bbox: 0.0089, s0.loss_cls: 0.1167, s0.acc: 95.3887, s0.loss_bbox: 0.0901, s1.loss_cls: 0.0518, s1.acc: 95.8262, s1.loss_bbox: 0.0890, s2.loss_cls: 0.0259, s2.acc: 95.8984, s2.loss_bbox: 0.0568, loss: 0.4477
2022-11-12 15:57:47,630 - mmdet - INFO - Epoch [8][600/962]	lr: 2.000e-02, eta: 0:47:17, time: 0.680, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0093, loss_rpn_bbox: 0.0074, s0.loss_cls: 0.1045, s0.acc: 96.1250, s0.loss_bbox: 0.0706, s1.loss_cls: 0.0458, s1.acc: 96.5754, s1.loss_bbox: 0.0719, s2.loss_cls: 0.0222, s2.acc: 96.6377, s2.loss_bbox: 0.0448, loss: 0.3765
2022-11-12 15:58:21,250 - mmdet - INFO - Epoch [8][650/962]	lr: 2.000e-02, eta: 0:46:43, time: 0.672, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0073, loss_rpn_bbox: 0.0081, s0.loss_cls: 0.1019, s0.acc: 96.0547, s0.loss_bbox: 0.0702, s1.loss_cls: 0.0442, s1.acc: 96.4531, s1.loss_bbox: 0.0754, s2.loss_cls: 0.0216, s2.acc: 96.6016, s2.loss_bbox: 0.0484, loss: 0.3771
2022-11-12 15:58:55,201 - mmdet - INFO - Epoch [8][700/962]	lr: 2.000e-02, eta: 0:46:10, time: 0.679, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0095, loss_rpn_bbox: 0.0095, s0.loss_cls: 0.1114, s0.acc: 95.8438, s0.loss_bbox: 0.0804, s1.loss_cls: 0.0483, s1.acc: 96.4141, s1.loss_bbox: 0.0775, s2.loss_cls: 0.0230, s2.acc: 96.7305, s2.loss_bbox: 0.0487, loss: 0.4083
2022-11-12 15:59:29,054 - mmdet - INFO - Epoch [8][750/962]	lr: 2.000e-02, eta: 0:45:36, time: 0.677, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0084, loss_rpn_bbox: 0.0090, s0.loss_cls: 0.1217, s0.acc: 95.2656, s0.loss_bbox: 0.0840, s1.loss_cls: 0.0544, s1.acc: 95.8223, s1.loss_bbox: 0.0815, s2.loss_cls: 0.0263, s2.acc: 95.9375, s2.loss_bbox: 0.0498, loss: 0.4351
2022-11-12 16:00:02,700 - mmdet - INFO - Epoch [8][800/962]	lr: 2.000e-02, eta: 0:45:03, time: 0.673, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0091, loss_rpn_bbox: 0.0093, s0.loss_cls: 0.1091, s0.acc: 95.7773, s0.loss_bbox: 0.0799, s1.loss_cls: 0.0496, s1.acc: 96.2109, s1.loss_bbox: 0.0819, s2.loss_cls: 0.0248, s2.acc: 96.2383, s2.loss_bbox: 0.0535, loss: 0.4172
2022-11-12 16:00:36,521 - mmdet - INFO - Epoch [8][850/962]	lr: 2.000e-02, eta: 0:44:29, time: 0.676, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0096, loss_rpn_bbox: 0.0096, s0.loss_cls: 0.1091, s0.acc: 95.8008, s0.loss_bbox: 0.0808, s1.loss_cls: 0.0509, s1.acc: 96.0215, s1.loss_bbox: 0.0781, s2.loss_cls: 0.0252, s2.acc: 96.1680, s2.loss_bbox: 0.0472, loss: 0.4106
2022-11-12 16:01:10,534 - mmdet - INFO - Epoch [8][900/962]	lr: 2.000e-02, eta: 0:43:55, time: 0.680, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0073, loss_rpn_bbox: 0.0092, s0.loss_cls: 0.1089, s0.acc: 95.8301, s0.loss_bbox: 0.0741, s1.loss_cls: 0.0472, s1.acc: 96.2221, s1.loss_bbox: 0.0722, s2.loss_cls: 0.0229, s2.acc: 96.4134, s2.loss_bbox: 0.0461, loss: 0.3879
2022-11-12 16:01:44,568 - mmdet - INFO - Epoch [8][950/962]	lr: 2.000e-02, eta: 0:43:22, time: 0.681, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0086, loss_rpn_bbox: 0.0089, s0.loss_cls: 0.0968, s0.acc: 96.2930, s0.loss_bbox: 0.0702, s1.loss_cls: 0.0415, s1.acc: 96.7559, s1.loss_bbox: 0.0609, s2.loss_cls: 0.0196, s2.acc: 96.9102, s2.loss_bbox: 0.0387, loss: 0.3453
2022-11-12 16:01:52,627 - mmdet - INFO - Saving checkpoint at 8 epochs
[>>] 200/200, 6.2 task/s, elapsed: 32s, ETA:     0s2022-11-12 16:02:29,399 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.30s).
Accumulating evaluation results...
DONE (t=0.05s).
2022-11-12 16:02:29,760 - mmdet - INFO -
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.120
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.268
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.099
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.120
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.227
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.227
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.227
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.227

2022-11-12 16:02:29,763 - mmdet - INFO - Exp name: rcnn_r50_caffe_fpn_mstrain-poly_1x_COCO.py
2022-11-12 16:02:29,763 - mmdet - INFO - Epoch(val) [8][200]	bbox_mAP: 0.1200, bbox_mAP_50: 0.2680, bbox_mAP_75: 0.0990, bbox_mAP_s: -1.0000, bbox_mAP_m: -1.0000, bbox_mAP_l: 0.1200, bbox_mAP_copypaste: 0.120 0.268 0.099 -1.000 -1.000 0.120
2022-11-12 16:03:05,802 - mmdet - INFO - Epoch [9][50/962]	lr: 2.000e-03, eta: 0:42:37, time: 0.718, data_time: 0.058, memory: 3483, loss_rpn_cls: 0.0074, loss_rpn_bbox: 0.0068, s0.loss_cls: 0.0952, s0.acc: 96.2559, s0.loss_bbox: 0.0678, s1.loss_cls: 0.0414, s1.acc: 96.7689, s1.loss_bbox: 0.0660, s2.loss_cls: 0.0205, s2.acc: 96.7572, s2.loss_bbox: 0.0428, loss: 0.3478
2022-11-12 16:03:39,772 - mmdet - INFO - Epoch [9][100/962]	lr: 2.000e-03, eta: 0:42:04, time: 0.679, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0082, loss_rpn_bbox: 0.0082, s0.loss_cls: 0.0939, s0.acc: 96.2402, s0.loss_bbox: 0.0675, s1.loss_cls: 0.0409, s1.acc: 96.8523, s1.loss_bbox: 0.0681, s2.loss_cls: 0.0203, s2.acc: 96.8640, s2.loss_bbox: 0.0440, loss: 0.3511
2022-11-12 16:04:13,741 - mmdet - INFO - Epoch [9][150/962]	lr: 2.000e-03, eta: 0:41:30, time: 0.679, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0071, loss_rpn_bbox: 0.0070, s0.loss_cls: 0.0919, s0.acc: 96.3906, s0.loss_bbox: 0.0693, s1.loss_cls: 0.0388, s1.acc: 97.0963, s1.loss_bbox: 0.0674, s2.loss_cls: 0.0193, s2.acc: 97.2523, s2.loss_bbox: 0.0448, loss: 0.3457
2022-11-12 16:04:47,099 - mmdet - INFO - Epoch [9][200/962]	lr: 2.000e-03, eta: 0:40:56, time: 0.667, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0069, loss_rpn_bbox: 0.0060, s0.loss_cls: 0.0794, s0.acc: 97.0332, s0.loss_bbox: 0.0551, s1.loss_cls: 0.0346, s1.acc: 97.3936, s1.loss_bbox: 0.0537, s2.loss_cls: 0.0168, s2.acc: 97.3834, s2.loss_bbox: 0.0351, loss: 0.2876
2022-11-12 16:05:21,682 - mmdet - INFO - Epoch [9][250/962]	lr: 2.000e-03, eta: 0:40:23, time: 0.692, data_time: 0.013, memory: 3483, loss_rpn_cls: 0.0056, loss_rpn_bbox: 0.0072, s0.loss_cls: 0.0851, s0.acc: 96.5684, s0.loss_bbox: 0.0603, s1.loss_cls: 0.0368, s1.acc: 97.0807, s1.loss_bbox: 0.0611, s2.loss_cls: 0.0187, s2.acc: 97.1157, s2.loss_bbox: 0.0410, loss: 0.3158
2022-11-12 16:05:55,185 - mmdet - INFO - Epoch [9][300/962]	lr: 2.000e-03, eta: 0:39:49, time: 0.670, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0055, loss_rpn_bbox: 0.0064, s0.loss_cls: 0.0819, s0.acc: 96.8223, s0.loss_bbox: 0.0570, s1.loss_cls: 0.0356, s1.acc: 97.2981, s1.loss_bbox: 0.0563, s2.loss_cls: 0.0177, s2.acc: 97.1904, s2.loss_bbox: 0.0369, loss: 0.2972
2022-11-12 16:06:29,091 - mmdet - INFO - Epoch [9][350/962]	lr: 2.000e-03, eta: 0:39:16, time: 0.678, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0058, loss_rpn_bbox: 0.0062, s0.loss_cls: 0.0853, s0.acc: 96.7188, s0.loss_bbox: 0.0621, s1.loss_cls: 0.0359, s1.acc: 97.3060, s1.loss_bbox: 0.0598, s2.loss_cls: 0.0181, s2.acc: 97.1727, s2.loss_bbox: 0.0396, loss: 0.3127
2022-11-12 16:07:02,966 - mmdet - INFO - Epoch [9][400/962]	lr: 2.000e-03, eta: 0:38:42, time: 0.677, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0056, loss_rpn_bbox: 0.0068, s0.loss_cls: 0.0738, s0.acc: 97.1582, s0.loss_bbox: 0.0515, s1.loss_cls: 0.0315, s1.acc: 97.5569, s1.loss_bbox: 0.0524, s2.loss_cls: 0.0156, s2.acc: 97.4435, s2.loss_bbox: 0.0365, loss: 0.2738
2022-11-12 16:07:36,483 - mmdet - INFO - Epoch [9][450/962]	lr: 2.000e-03, eta: 0:38:08, time: 0.670, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0071, loss_rpn_bbox: 0.0072, s0.loss_cls: 0.0866, s0.acc: 96.6758, s0.loss_bbox: 0.0630, s1.loss_cls: 0.0376, s1.acc: 97.1389, s1.loss_bbox: 0.0631, s2.loss_cls: 0.0190, s2.acc: 97.1215, s2.loss_bbox: 0.0410, loss: 0.3246
2022-11-12 16:08:10,499 - mmdet - INFO - Epoch [9][500/962]	lr: 2.000e-03, eta: 0:37:35, time: 0.680, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0050, loss_rpn_bbox: 0.0054, s0.loss_cls: 0.0692, s0.acc: 97.2949, s0.loss_bbox: 0.0427, s1.loss_cls: 0.0309, s1.acc: 97.6582, s1.loss_bbox: 0.0436, s2.loss_cls: 0.0156, s2.acc: 97.7222, s2.loss_bbox: 0.0322, loss: 0.2446
2022-11-12 16:08:44,121 - mmdet - INFO - Epoch [9][550/962]	lr: 2.000e-03, eta: 0:37:01, time: 0.672, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0051, loss_rpn_bbox: 0.0060, s0.loss_cls: 0.0841, s0.acc: 96.7676, s0.loss_bbox: 0.0579, s1.loss_cls: 0.0370, s1.acc: 97.1536, s1.loss_bbox: 0.0588, s2.loss_cls: 0.0188, s2.acc: 97.0925, s2.loss_bbox: 0.0400, loss: 0.3077
2022-11-12 16:09:18,543 - mmdet - INFO - Epoch [9][600/962]	lr: 2.000e-03, eta: 0:36:28, time: 0.688, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0060, loss_rpn_bbox: 0.0057, s0.loss_cls: 0.0790, s0.acc: 96.9141, s0.loss_bbox: 0.0538, s1.loss_cls: 0.0346, s1.acc: 97.2950, s1.loss_bbox: 0.0552, s2.loss_cls: 0.0167, s2.acc: 97.3777, s2.loss_bbox: 0.0376, loss: 0.2887
2022-11-12 16:09:52,583 - mmdet - INFO - Epoch [9][650/962]	lr: 2.000e-03, eta: 0:35:54, time: 0.681, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0057, loss_rpn_bbox: 0.0083, s0.loss_cls: 0.0782, s0.acc: 96.9121, s0.loss_bbox: 0.0573, s1.loss_cls: 0.0345, s1.acc: 97.2891, s1.loss_bbox: 0.0610, s2.loss_cls: 0.0174, s2.acc: 97.3555, s2.loss_bbox: 0.0406, loss: 0.3031
2022-11-12 16:10:26,150 - mmdet - INFO - Epoch [9][700/962]	lr: 2.000e-03, eta: 0:35:20, time: 0.671, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0090, loss_rpn_bbox: 0.0081, s0.loss_cls: 0.0912, s0.acc: 96.4902, s0.loss_bbox: 0.0618, s1.loss_cls: 0.0390, s1.acc: 96.9990, s1.loss_bbox: 0.0613, s2.loss_cls: 0.0192, s2.acc: 96.9813, s2.loss_bbox: 0.0423, loss: 0.3319
2022-11-12 16:11:00,119 - mmdet - INFO - Epoch [9][750/962]	lr: 2.000e-03, eta: 0:34:47, time: 0.679, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0065, loss_rpn_bbox: 0.0059, s0.loss_cls: 0.0683, s0.acc: 97.4180, s0.loss_bbox: 0.0500, s1.loss_cls: 0.0286, s1.acc: 97.8633, s1.loss_bbox: 0.0507, s2.loss_cls: 0.0138, s2.acc: 97.8164, s2.loss_bbox: 0.0336, loss: 0.2574
2022-11-12 16:11:33,948 - mmdet - INFO - Epoch [9][800/962]	lr: 2.000e-03, eta: 0:34:13, time: 0.677, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0068, loss_rpn_bbox: 0.0067, s0.loss_cls: 0.0847, s0.acc: 96.6797, s0.loss_bbox: 0.0623, s1.loss_cls: 0.0385, s1.acc: 96.9258, s1.loss_bbox: 0.0617, s2.loss_cls: 0.0193, s2.acc: 96.8690, s2.loss_bbox: 0.0409, loss: 0.3208
2022-11-12 16:12:07,604 - mmdet - INFO - Epoch [9][850/962]	lr: 2.000e-03, eta: 0:33:39, time: 0.673, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0066, loss_rpn_bbox: 0.0066, s0.loss_cls: 0.0810, s0.acc: 96.8730, s0.loss_bbox: 0.0601, s1.loss_cls: 0.0341, s1.acc: 97.3146, s1.loss_bbox: 0.0600, s2.loss_cls: 0.0171, s2.acc: 97.3315, s2.loss_bbox: 0.0418, loss: 0.3072
2022-11-12 16:12:41,608 - mmdet - INFO - Epoch [9][900/962]	lr: 2.000e-03, eta: 0:33:06, time: 0.680, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0065, loss_rpn_bbox: 0.0065, s0.loss_cls: 0.0783, s0.acc: 97.0566, s0.loss_bbox: 0.0554, s1.loss_cls: 0.0322, s1.acc: 97.5858, s1.loss_bbox: 0.0527, s2.loss_cls: 0.0167, s2.acc: 97.3413, s2.loss_bbox: 0.0360, loss: 0.2843
2022-11-12 16:13:15,154 - mmdet - INFO - Epoch [9][950/962]	lr: 2.000e-03, eta: 0:32:32, time: 0.671, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0062, loss_rpn_bbox: 0.0065, s0.loss_cls: 0.0849, s0.acc: 96.6289, s0.loss_bbox: 0.0654, s1.loss_cls: 0.0368, s1.acc: 97.0755, s1.loss_bbox: 0.0633, s2.loss_cls: 0.0191, s2.acc: 97.0674, s2.loss_bbox: 0.0379, loss: 0.3201
2022-11-12 16:13:23,759 - mmdet - INFO - Saving checkpoint at 9 epochs
[>>] 200/200, 6.4 task/s, elapsed: 31s, ETA:     0s2022-11-12 16:14:00,241 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.24s).
Accumulating evaluation results...
DONE (t=0.04s).
2022-11-12 16:14:00,529 - mmdet - INFO -
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.139
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.309
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.110
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.139
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.241
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.241
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.241
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.241

2022-11-12 16:14:00,531 - mmdet - INFO - Exp name: rcnn_r50_caffe_fpn_mstrain-poly_1x_COCO.py
2022-11-12 16:14:00,532 - mmdet - INFO - Epoch(val) [9][200]	bbox_mAP: 0.1390, bbox_mAP_50: 0.3090, bbox_mAP_75: 0.1100, bbox_mAP_s: -1.0000, bbox_mAP_m: -1.0000, bbox_mAP_l: 0.1390, bbox_mAP_copypaste: 0.139 0.309 0.110 -1.000 -1.000 0.139
2022-11-12 16:14:37,064 - mmdet - INFO - Epoch [10][50/962]	lr: 2.000e-03, eta: 0:31:49, time: 0.728, data_time: 0.060, memory: 3483, loss_rpn_cls: 0.0055, loss_rpn_bbox: 0.0072, s0.loss_cls: 0.0795, s0.acc: 96.8867, s0.loss_bbox: 0.0596, s1.loss_cls: 0.0326, s1.acc: 97.4616, s1.loss_bbox: 0.0567, s2.loss_cls: 0.0165, s2.acc: 97.3632, s2.loss_bbox: 0.0385, loss: 0.2961
2022-11-12 16:15:10,570 - mmdet - INFO - Epoch [10][100/962]	lr: 2.000e-03, eta: 0:31:15, time: 0.670, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0059, loss_rpn_bbox: 0.0067, s0.loss_cls: 0.0754, s0.acc: 97.0625, s0.loss_bbox: 0.0588, s1.loss_cls: 0.0327, s1.acc: 97.4395, s1.loss_bbox: 0.0585, s2.loss_cls: 0.0162, s2.acc: 97.4055, s2.loss_bbox: 0.0375, loss: 0.2917
2022-11-12 16:15:44,489 - mmdet - INFO - Epoch [10][150/962]	lr: 2.000e-03, eta: 0:30:41, time: 0.678, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0057, loss_rpn_bbox: 0.0053, s0.loss_cls: 0.0663, s0.acc: 97.5059, s0.loss_bbox: 0.0448, s1.loss_cls: 0.0275, s1.acc: 97.9556, s1.loss_bbox: 0.0437, s2.loss_cls: 0.0133, s2.acc: 97.9409, s2.loss_bbox: 0.0297, loss: 0.2364
2022-11-12 16:16:18,471 - mmdet - INFO - Epoch [10][200/962]	lr: 2.000e-03, eta: 0:30:08, time: 0.680, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0053, loss_rpn_bbox: 0.0057, s0.loss_cls: 0.0767, s0.acc: 96.8828, s0.loss_bbox: 0.0523, s1.loss_cls: 0.0320, s1.acc: 97.4399, s1.loss_bbox: 0.0515, s2.loss_cls: 0.0159, s2.acc: 97.5015, s2.loss_bbox: 0.0349, loss: 0.2743
2022-11-12 16:16:52,008 - mmdet - INFO - Epoch [10][250/962]	lr: 2.000e-03, eta: 0:29:34, time: 0.671, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0063, loss_rpn_bbox: 0.0082, s0.loss_cls: 0.0792, s0.acc: 96.9141, s0.loss_bbox: 0.0616, s1.loss_cls: 0.0325, s1.acc: 97.3892, s1.loss_bbox: 0.0597, s2.loss_cls: 0.0175, s2.acc: 97.2014, s2.loss_bbox: 0.0392, loss: 0.3042
2022-11-12 16:17:26,212 - mmdet - INFO - Epoch [10][300/962]	lr: 2.000e-03, eta: 0:29:01, time: 0.684, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0060, loss_rpn_bbox: 0.0073, s0.loss_cls: 0.0803, s0.acc: 96.7383, s0.loss_bbox: 0.0587, s1.loss_cls: 0.0336, s1.acc: 97.2083, s1.loss_bbox: 0.0584, s2.loss_cls: 0.0169, s2.acc: 97.3279, s2.loss_bbox: 0.0404, loss: 0.3016
2022-11-12 16:17:59,939 - mmdet - INFO - Epoch [10][350/962]	lr: 2.000e-03, eta: 0:28:27, time: 0.675, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0052, loss_rpn_bbox: 0.0056, s0.loss_cls: 0.0709, s0.acc: 97.2324, s0.loss_bbox: 0.0556, s1.loss_cls: 0.0304, s1.acc: 97.5716, s1.loss_bbox: 0.0555, s2.loss_cls: 0.0161, s2.acc: 97.4306, s2.loss_bbox: 0.0358, loss: 0.2751
2022-11-12 16:18:33,929 - mmdet - INFO - Epoch [10][400/962]	lr: 2.000e-03, eta: 0:27:53, time: 0.680, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0064, loss_rpn_bbox: 0.0068, s0.loss_cls: 0.0748, s0.acc: 97.0605, s0.loss_bbox: 0.0547, s1.loss_cls: 0.0319, s1.acc: 97.4469, s1.loss_bbox: 0.0547, s2.loss_cls: 0.0161, s2.acc: 97.4694, s2.loss_bbox: 0.0369, loss: 0.2824
2022-11-12 16:19:07,912 - mmdet - INFO - Epoch [10][450/962]	lr: 2.000e-03, eta: 0:27:20, time: 0.680, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0060, loss_rpn_bbox: 0.0077, s0.loss_cls: 0.0737, s0.acc: 97.1172, s0.loss_bbox: 0.0523, s1.loss_cls: 0.0323, s1.acc: 97.5556, s1.loss_bbox: 0.0533, s2.loss_cls: 0.0160, s2.acc: 97.4627, s2.loss_bbox: 0.0351, loss: 0.2764
2022-11-12 16:19:41,424 - mmdet - INFO - Epoch [10][500/962]	lr: 2.000e-03, eta: 0:26:46, time: 0.670, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0059, loss_rpn_bbox: 0.0049, s0.loss_cls: 0.0674, s0.acc: 97.3340, s0.loss_bbox: 0.0495, s1.loss_cls: 0.0267, s1.acc: 97.9004, s1.loss_bbox: 0.0497, s2.loss_cls: 0.0129, s2.acc: 98.0175, s2.loss_bbox: 0.0337, loss: 0.2508
2022-11-12 16:20:15,355 - mmdet - INFO - Epoch [10][550/962]	lr: 2.000e-03, eta: 0:26:12, time: 0.679, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0041, loss_rpn_bbox: 0.0055, s0.loss_cls: 0.0717, s0.acc: 97.0820, s0.loss_bbox: 0.0522, s1.loss_cls: 0.0288, s1.acc: 97.6944, s1.loss_bbox: 0.0478, s2.loss_cls: 0.0154, s2.acc: 97.5814, s2.loss_bbox: 0.0334, loss: 0.2590
2022-11-12 16:20:48,879 - mmdet - INFO - Epoch [10][600/962]	lr: 2.000e-03, eta: 0:25:39, time: 0.670, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0062, loss_rpn_bbox: 0.0064, s0.loss_cls: 0.0746, s0.acc: 97.1660, s0.loss_bbox: 0.0529, s1.loss_cls: 0.0323, s1.acc: 97.3924, s1.loss_bbox: 0.0523, s2.loss_cls: 0.0162, s2.acc: 97.4017, s2.loss_bbox: 0.0368, loss: 0.2777
2022-11-12 16:21:22,819 - mmdet - INFO - Epoch [10][650/962]	lr: 2.000e-03, eta: 0:25:05, time: 0.679, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0055, loss_rpn_bbox: 0.0053, s0.loss_cls: 0.0707, s0.acc: 97.1855, s0.loss_bbox: 0.0498, s1.loss_cls: 0.0303, s1.acc: 97.5723, s1.loss_bbox: 0.0478, s2.loss_cls: 0.0160, s2.acc: 97.3086, s2.loss_bbox: 0.0315, loss: 0.2567
2022-11-12 16:21:57,289 - mmdet - INFO - Epoch [10][700/962]	lr: 2.000e-03, eta: 0:24:32, time: 0.689, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0054, loss_rpn_bbox: 0.0058, s0.loss_cls: 0.0610, s0.acc: 97.5820, s0.loss_bbox: 0.0437, s1.loss_cls: 0.0242, s1.acc: 98.1015, s1.loss_bbox: 0.0430, s2.loss_cls: 0.0123, s2.acc: 97.9680, s2.loss_bbox: 0.0293, loss: 0.2246
2022-11-12 16:22:30,829 - mmdet - INFO - Epoch [10][750/962]	lr: 2.000e-03, eta: 0:23:58, time: 0.671, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0056, loss_rpn_bbox: 0.0071, s0.loss_cls: 0.0776, s0.acc: 96.8672, s0.loss_bbox: 0.0545, s1.loss_cls: 0.0328, s1.acc: 97.4059, s1.loss_bbox: 0.0525, s2.loss_cls: 0.0160, s2.acc: 97.4525, s2.loss_bbox: 0.0362, loss: 0.2823
2022-11-12 16:23:04,730 - mmdet - INFO - Epoch [10][800/962]	lr: 2.000e-03, eta: 0:23:24, time: 0.678, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0053, loss_rpn_bbox: 0.0067, s0.loss_cls: 0.0793, s0.acc: 96.7832, s0.loss_bbox: 0.0577, s1.loss_cls: 0.0329, s1.acc: 97.3727, s1.loss_bbox: 0.0545, s2.loss_cls: 0.0155, s2.acc: 97.4822, s2.loss_bbox: 0.0355, loss: 0.2873
2022-11-12 16:23:38,264 - mmdet - INFO - Epoch [10][850/962]	lr: 2.000e-03, eta: 0:22:51, time: 0.671, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0048, loss_rpn_bbox: 0.0050, s0.loss_cls: 0.0730, s0.acc: 97.0664, s0.loss_bbox: 0.0547, s1.loss_cls: 0.0307, s1.acc: 97.5512, s1.loss_bbox: 0.0515, s2.loss_cls: 0.0150, s2.acc: 97.5839, s2.loss_bbox: 0.0353, loss: 0.2700
2022-11-12 16:24:12,246 - mmdet - INFO - Epoch [10][900/962]	lr: 2.000e-03, eta: 0:22:17, time: 0.680, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0056, loss_rpn_bbox: 0.0052, s0.loss_cls: 0.0630, s0.acc: 97.5430, s0.loss_bbox: 0.0418, s1.loss_cls: 0.0259, s1.acc: 98.1053, s1.loss_bbox: 0.0421, s2.loss_cls: 0.0120, s2.acc: 98.2947, s2.loss_bbox: 0.0294, loss: 0.2248
2022-11-12 16:24:46,174 - mmdet - INFO - Epoch [10][950/962]	lr: 2.000e-03, eta: 0:21:43, time: 0.679, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0059, loss_rpn_bbox: 0.0058, s0.loss_cls: 0.0846, s0.acc: 96.7012, s0.loss_bbox: 0.0638, s1.loss_cls: 0.0368, s1.acc: 97.1473, s1.loss_bbox: 0.0638, s2.loss_cls: 0.0184, s2.acc: 97.1879, s2.loss_bbox: 0.0415, loss: 0.3207
2022-11-12 16:24:54,222 - mmdet - INFO - Saving checkpoint at 10 epochs
[>>] 200/200, 6.4 task/s, elapsed: 31s, ETA:     0s2022-11-12 16:25:29,807 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.27s).
Accumulating evaluation results...
DONE (t=0.04s).
2022-11-12 16:25:30,128 - mmdet - INFO -
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.144
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.312
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.119
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.144
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.246
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.246
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.246
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.246

2022-11-12 16:25:30,130 - mmdet - INFO - Exp name: rcnn_r50_caffe_fpn_mstrain-poly_1x_COCO.py
2022-11-12 16:25:30,131 - mmdet - INFO - Epoch(val) [10][200]	bbox_mAP: 0.1440, bbox_mAP_50: 0.3120, bbox_mAP_75: 0.1190, bbox_mAP_s: -1.0000, bbox_mAP_m: -1.0000, bbox_mAP_l: 0.1440, bbox_mAP_copypaste: 0.144 0.312 0.119 -1.000 -1.000 0.144
2022-11-12 16:26:07,736 - mmdet - INFO - Epoch [11][50/962]	lr: 2.000e-03, eta: 0:21:01, time: 0.749, data_time: 0.065, memory: 3483, loss_rpn_cls: 0.0061, loss_rpn_bbox: 0.0065, s0.loss_cls: 0.0739, s0.acc: 96.9941, s0.loss_bbox: 0.0540, s1.loss_cls: 0.0316, s1.acc: 97.3514, s1.loss_bbox: 0.0518, s2.loss_cls: 0.0161, s2.acc: 97.4718, s2.loss_bbox: 0.0350, loss: 0.2749
2022-11-12 16:26:41,605 - mmdet - INFO - Epoch [11][100/962]	lr: 2.000e-03, eta: 0:20:27, time: 0.677, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0054, loss_rpn_bbox: 0.0056, s0.loss_cls: 0.0727, s0.acc: 97.1855, s0.loss_bbox: 0.0501, s1.loss_cls: 0.0307, s1.acc: 97.6341, s1.loss_bbox: 0.0498, s2.loss_cls: 0.0148, s2.acc: 97.6493, s2.loss_bbox: 0.0343, loss: 0.2633
2022-11-12 16:27:14,954 - mmdet - INFO - Epoch [11][150/962]	lr: 2.000e-03, eta: 0:19:54, time: 0.667, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0051, loss_rpn_bbox: 0.0056, s0.loss_cls: 0.0681, s0.acc: 97.3105, s0.loss_bbox: 0.0530, s1.loss_cls: 0.0285, s1.acc: 97.7252, s1.loss_bbox: 0.0502, s2.loss_cls: 0.0139, s2.acc: 97.9207, s2.loss_bbox: 0.0338, loss: 0.2582
2022-11-12 16:27:48,915 - mmdet - INFO - Epoch [11][200/962]	lr: 2.000e-03, eta: 0:19:20, time: 0.679, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0033, loss_rpn_bbox: 0.0051, s0.loss_cls: 0.0639, s0.acc: 97.4688, s0.loss_bbox: 0.0430, s1.loss_cls: 0.0272, s1.acc: 97.8594, s1.loss_bbox: 0.0422, s2.loss_cls: 0.0135, s2.acc: 97.7754, s2.loss_bbox: 0.0285, loss: 0.2269
2022-11-12 16:28:22,420 - mmdet - INFO - Epoch [11][250/962]	lr: 2.000e-03, eta: 0:18:46, time: 0.670, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0060, loss_rpn_bbox: 0.0065, s0.loss_cls: 0.0687, s0.acc: 97.2754, s0.loss_bbox: 0.0529, s1.loss_cls: 0.0272, s1.acc: 97.8904, s1.loss_bbox: 0.0490, s2.loss_cls: 0.0135, s2.acc: 97.8512, s2.loss_bbox: 0.0333, loss: 0.2573
2022-11-12 16:28:56,325 - mmdet - INFO - Epoch [11][300/962]	lr: 2.000e-03, eta: 0:18:13, time: 0.678, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0050, loss_rpn_bbox: 0.0063, s0.loss_cls: 0.0680, s0.acc: 97.3047, s0.loss_bbox: 0.0482, s1.loss_cls: 0.0268, s1.acc: 97.9419, s1.loss_bbox: 0.0476, s2.loss_cls: 0.0137, s2.acc: 97.9147, s2.loss_bbox: 0.0343, loss: 0.2499
2022-11-12 16:29:30,196 - mmdet - INFO - Epoch [11][350/962]	lr: 2.000e-03, eta: 0:17:39, time: 0.677, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0055, loss_rpn_bbox: 0.0053, s0.loss_cls: 0.0668, s0.acc: 97.3633, s0.loss_bbox: 0.0517, s1.loss_cls: 0.0284, s1.acc: 97.7495, s1.loss_bbox: 0.0519, s2.loss_cls: 0.0147, s2.acc: 97.6065, s2.loss_bbox: 0.0361, loss: 0.2604
2022-11-12 16:30:04,118 - mmdet - INFO - Epoch [11][400/962]	lr: 2.000e-03, eta: 0:17:05, time: 0.678, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0061, loss_rpn_bbox: 0.0058, s0.loss_cls: 0.0746, s0.acc: 97.0234, s0.loss_bbox: 0.0528, s1.loss_cls: 0.0306, s1.acc: 97.6222, s1.loss_bbox: 0.0532, s2.loss_cls: 0.0152, s2.acc: 97.6432, s2.loss_bbox: 0.0362, loss: 0.2746
2022-11-12 16:30:37,946 - mmdet - INFO - Epoch [11][450/962]	lr: 2.000e-03, eta: 0:16:32, time: 0.677, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0056, loss_rpn_bbox: 0.0053, s0.loss_cls: 0.0635, s0.acc: 97.5918, s0.loss_bbox: 0.0454, s1.loss_cls: 0.0259, s1.acc: 97.9947, s1.loss_bbox: 0.0437, s2.loss_cls: 0.0125, s2.acc: 98.0980, s2.loss_bbox: 0.0310, loss: 0.2329
2022-11-12 16:31:11,347 - mmdet - INFO - Epoch [11][500/962]	lr: 2.000e-03, eta: 0:15:58, time: 0.668, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0046, loss_rpn_bbox: 0.0049, s0.loss_cls: 0.0681, s0.acc: 97.3652, s0.loss_bbox: 0.0492, s1.loss_cls: 0.0284, s1.acc: 97.7638, s1.loss_bbox: 0.0460, s2.loss_cls: 0.0139, s2.acc: 97.9029, s2.loss_bbox: 0.0312, loss: 0.2462
2022-11-12 16:31:45,342 - mmdet - INFO - Epoch [11][550/962]	lr: 2.000e-03, eta: 0:15:24, time: 0.680, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0049, loss_rpn_bbox: 0.0062, s0.loss_cls: 0.0613, s0.acc: 97.4785, s0.loss_bbox: 0.0460, s1.loss_cls: 0.0266, s1.acc: 97.8186, s1.loss_bbox: 0.0452, s2.loss_cls: 0.0133, s2.acc: 97.7432, s2.loss_bbox: 0.0301, loss: 0.2336
2022-11-12 16:32:19,227 - mmdet - INFO - Epoch [11][600/962]	lr: 2.000e-03, eta: 0:14:51, time: 0.678, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0064, loss_rpn_bbox: 0.0065, s0.loss_cls: 0.0684, s0.acc: 97.3301, s0.loss_bbox: 0.0517, s1.loss_cls: 0.0267, s1.acc: 97.9292, s1.loss_bbox: 0.0507, s2.loss_cls: 0.0133, s2.acc: 97.9677, s2.loss_bbox: 0.0343, loss: 0.2581
2022-11-12 16:32:52,635 - mmdet - INFO - Epoch [11][650/962]	lr: 2.000e-03, eta: 0:14:17, time: 0.668, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0058, loss_rpn_bbox: 0.0065, s0.loss_cls: 0.0753, s0.acc: 97.0273, s0.loss_bbox: 0.0537, s1.loss_cls: 0.0314, s1.acc: 97.6047, s1.loss_bbox: 0.0524, s2.loss_cls: 0.0156, s2.acc: 97.5286, s2.loss_bbox: 0.0355, loss: 0.2762
2022-11-12 16:33:26,603 - mmdet - INFO - Epoch [11][700/962]	lr: 2.000e-03, eta: 0:13:44, time: 0.679, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0034, loss_rpn_bbox: 0.0060, s0.loss_cls: 0.0670, s0.acc: 97.2891, s0.loss_bbox: 0.0494, s1.loss_cls: 0.0279, s1.acc: 97.7224, s1.loss_bbox: 0.0498, s2.loss_cls: 0.0140, s2.acc: 97.7257, s2.loss_bbox: 0.0342, loss: 0.2517
2022-11-12 16:34:00,392 - mmdet - INFO - Epoch [11][750/962]	lr: 2.000e-03, eta: 0:13:10, time: 0.676, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0044, loss_rpn_bbox: 0.0054, s0.loss_cls: 0.0594, s0.acc: 97.6582, s0.loss_bbox: 0.0450, s1.loss_cls: 0.0240, s1.acc: 98.1168, s1.loss_bbox: 0.0423, s2.loss_cls: 0.0122, s2.acc: 98.0876, s2.loss_bbox: 0.0286, loss: 0.2212
2022-11-12 16:34:34,384 - mmdet - INFO - Epoch [11][800/962]	lr: 2.000e-03, eta: 0:12:36, time: 0.680, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0057, loss_rpn_bbox: 0.0065, s0.loss_cls: 0.0697, s0.acc: 97.2949, s0.loss_bbox: 0.0568, s1.loss_cls: 0.0297, s1.acc: 97.7122, s1.loss_bbox: 0.0550, s2.loss_cls: 0.0147, s2.acc: 97.6241, s2.loss_bbox: 0.0362, loss: 0.2744
2022-11-12 16:35:08,224 - mmdet - INFO - Epoch [11][850/962]	lr: 2.000e-03, eta: 0:12:03, time: 0.677, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0052, loss_rpn_bbox: 0.0060, s0.loss_cls: 0.0771, s0.acc: 96.9844, s0.loss_bbox: 0.0581, s1.loss_cls: 0.0309, s1.acc: 97.5522, s1.loss_bbox: 0.0552, s2.loss_cls: 0.0154, s2.acc: 97.6122, s2.loss_bbox: 0.0363, loss: 0.2842
2022-11-12 16:35:41,689 - mmdet - INFO - Epoch [11][900/962]	lr: 2.000e-03, eta: 0:11:29, time: 0.669, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0055, loss_rpn_bbox: 0.0065, s0.loss_cls: 0.0748, s0.acc: 96.9297, s0.loss_bbox: 0.0521, s1.loss_cls: 0.0298, s1.acc: 97.6669, s1.loss_bbox: 0.0522, s2.loss_cls: 0.0154, s2.acc: 97.5168, s2.loss_bbox: 0.0362, loss: 0.2726
2022-11-12 16:36:15,771 - mmdet - INFO - Epoch [11][950/962]	lr: 2.000e-03, eta: 0:10:55, time: 0.682, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0057, loss_rpn_bbox: 0.0065, s0.loss_cls: 0.0684, s0.acc: 97.2773, s0.loss_bbox: 0.0494, s1.loss_cls: 0.0270, s1.acc: 97.7656, s1.loss_bbox: 0.0488, s2.loss_cls: 0.0138, s2.acc: 97.7773, s2.loss_bbox: 0.0335, loss: 0.2530
2022-11-12 16:36:23,858 - mmdet - INFO - Saving checkpoint at 11 epochs
[>>] 200/200, 6.2 task/s, elapsed: 32s, ETA:     0s2022-11-12 16:37:00,752 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.23s).
Accumulating evaluation results...
DONE (t=0.05s).
2022-11-12 16:37:01,043 - mmdet - INFO -
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.134
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.304
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.112
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.134
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.239
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.239
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.239
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.239

2022-11-12 16:37:01,046 - mmdet - INFO - Exp name: rcnn_r50_caffe_fpn_mstrain-poly_1x_COCO.py
2022-11-12 16:37:01,046 - mmdet - INFO - Epoch(val) [11][200]	bbox_mAP: 0.1340, bbox_mAP_50: 0.3040, bbox_mAP_75: 0.1120, bbox_mAP_s: -1.0000, bbox_mAP_m: -1.0000, bbox_mAP_l: 0.1340, bbox_mAP_copypaste: 0.134 0.304 0.112 -1.000 -1.000 0.134
2022-11-12 16:37:37,213 - mmdet - INFO - Epoch [12][50/962]	lr: 2.000e-04, eta: 0:10:13, time: 0.721, data_time: 0.059, memory: 3483, loss_rpn_cls: 0.0067, loss_rpn_bbox: 0.0071, s0.loss_cls: 0.0747, s0.acc: 97.0977, s0.loss_bbox: 0.0519, s1.loss_cls: 0.0298, s1.acc: 97.7103, s1.loss_bbox: 0.0508, s2.loss_cls: 0.0151, s2.acc: 97.5791, s2.loss_bbox: 0.0329, loss: 0.2691
2022-11-12 16:38:11,773 - mmdet - INFO - Epoch [12][100/962]	lr: 2.000e-04, eta: 0:09:39, time: 0.691, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0041, loss_rpn_bbox: 0.0077, s0.loss_cls: 0.0680, s0.acc: 97.2188, s0.loss_bbox: 0.0470, s1.loss_cls: 0.0286, s1.acc: 97.6953, s1.loss_bbox: 0.0466, s2.loss_cls: 0.0145, s2.acc: 97.5699, s2.loss_bbox: 0.0315, loss: 0.2481
2022-11-12 16:38:45,682 - mmdet - INFO - Epoch [12][150/962]	lr: 2.000e-04, eta: 0:09:06, time: 0.678, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0051, loss_rpn_bbox: 0.0049, s0.loss_cls: 0.0619, s0.acc: 97.5078, s0.loss_bbox: 0.0480, s1.loss_cls: 0.0262, s1.acc: 97.9569, s1.loss_bbox: 0.0462, s2.loss_cls: 0.0135, s2.acc: 97.6876, s2.loss_bbox: 0.0309, loss: 0.2368
2022-11-12 16:39:19,166 - mmdet - INFO - Epoch [12][200/962]	lr: 2.000e-04, eta: 0:08:32, time: 0.670, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0053, loss_rpn_bbox: 0.0063, s0.loss_cls: 0.0711, s0.acc: 97.2305, s0.loss_bbox: 0.0607, s1.loss_cls: 0.0292, s1.acc: 97.7170, s1.loss_bbox: 0.0564, s2.loss_cls: 0.0141, s2.acc: 97.8018, s2.loss_bbox: 0.0365, loss: 0.2796
2022-11-12 16:39:53,003 - mmdet - INFO - Epoch [12][250/962]	lr: 2.000e-04, eta: 0:07:59, time: 0.677, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0046, loss_rpn_bbox: 0.0044, s0.loss_cls: 0.0601, s0.acc: 97.6582, s0.loss_bbox: 0.0417, s1.loss_cls: 0.0253, s1.acc: 97.9294, s1.loss_bbox: 0.0411, s2.loss_cls: 0.0124, s2.acc: 98.0291, s2.loss_bbox: 0.0286, loss: 0.2181
2022-11-12 16:40:26,501 - mmdet - INFO - Epoch [12][300/962]	lr: 2.000e-04, eta: 0:07:25, time: 0.670, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0050, loss_rpn_bbox: 0.0062, s0.loss_cls: 0.0631, s0.acc: 97.4707, s0.loss_bbox: 0.0476, s1.loss_cls: 0.0253, s1.acc: 97.9569, s1.loss_bbox: 0.0484, s2.loss_cls: 0.0125, s2.acc: 97.8826, s2.loss_bbox: 0.0330, loss: 0.2412
2022-11-12 16:41:00,533 - mmdet - INFO - Epoch [12][350/962]	lr: 2.000e-04, eta: 0:06:51, time: 0.681, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0061, loss_rpn_bbox: 0.0062, s0.loss_cls: 0.0622, s0.acc: 97.6035, s0.loss_bbox: 0.0482, s1.loss_cls: 0.0264, s1.acc: 97.9117, s1.loss_bbox: 0.0449, s2.loss_cls: 0.0123, s2.acc: 98.0541, s2.loss_bbox: 0.0304, loss: 0.2367
2022-11-12 16:41:34,552 - mmdet - INFO - Epoch [12][400/962]	lr: 2.000e-04, eta: 0:06:18, time: 0.680, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0055, loss_rpn_bbox: 0.0055, s0.loss_cls: 0.0749, s0.acc: 97.0547, s0.loss_bbox: 0.0538, s1.loss_cls: 0.0326, s1.acc: 97.4771, s1.loss_bbox: 0.0522, s2.loss_cls: 0.0167, s2.acc: 97.3189, s2.loss_bbox: 0.0332, loss: 0.2742
2022-11-12 16:42:08,420 - mmdet - INFO - Epoch [12][450/962]	lr: 2.000e-04, eta: 0:05:44, time: 0.677, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0046, loss_rpn_bbox: 0.0054, s0.loss_cls: 0.0706, s0.acc: 97.1602, s0.loss_bbox: 0.0515, s1.loss_cls: 0.0275, s1.acc: 97.8061, s1.loss_bbox: 0.0469, s2.loss_cls: 0.0138, s2.acc: 97.8565, s2.loss_bbox: 0.0315, loss: 0.2518
2022-11-12 16:42:42,394 - mmdet - INFO - Epoch [12][500/962]	lr: 2.000e-04, eta: 0:05:10, time: 0.679, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0049, loss_rpn_bbox: 0.0056, s0.loss_cls: 0.0613, s0.acc: 97.6426, s0.loss_bbox: 0.0462, s1.loss_cls: 0.0258, s1.acc: 97.9836, s1.loss_bbox: 0.0442, s2.loss_cls: 0.0130, s2.acc: 97.8991, s2.loss_bbox: 0.0300, loss: 0.2310
2022-11-12 16:43:15,857 - mmdet - INFO - Epoch [12][550/962]	lr: 2.000e-04, eta: 0:04:37, time: 0.669, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0037, loss_rpn_bbox: 0.0048, s0.loss_cls: 0.0620, s0.acc: 97.5352, s0.loss_bbox: 0.0438, s1.loss_cls: 0.0251, s1.acc: 98.0692, s1.loss_bbox: 0.0428, s2.loss_cls: 0.0124, s2.acc: 97.9825, s2.loss_bbox: 0.0290, loss: 0.2236
2022-11-12 16:43:49,853 - mmdet - INFO - Epoch [12][600/962]	lr: 2.000e-04, eta: 0:04:03, time: 0.680, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0042, loss_rpn_bbox: 0.0050, s0.loss_cls: 0.0615, s0.acc: 97.5410, s0.loss_bbox: 0.0450, s1.loss_cls: 0.0256, s1.acc: 98.0031, s1.loss_bbox: 0.0439, s2.loss_cls: 0.0128, s2.acc: 97.9494, s2.loss_bbox: 0.0299, loss: 0.2279
2022-11-12 16:44:23,766 - mmdet - INFO - Epoch [12][650/962]	lr: 2.000e-04, eta: 0:03:29, time: 0.678, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0058, loss_rpn_bbox: 0.0055, s0.loss_cls: 0.0666, s0.acc: 97.3496, s0.loss_bbox: 0.0476, s1.loss_cls: 0.0266, s1.acc: 97.9308, s1.loss_bbox: 0.0481, s2.loss_cls: 0.0131, s2.acc: 97.9968, s2.loss_bbox: 0.0344, loss: 0.2478
2022-11-12 16:44:57,117 - mmdet - INFO - Epoch [12][700/962]	lr: 2.000e-04, eta: 0:02:56, time: 0.667, data_time: 0.010, memory: 3483, loss_rpn_cls: 0.0045, loss_rpn_bbox: 0.0049, s0.loss_cls: 0.0673, s0.acc: 97.3340, s0.loss_bbox: 0.0490, s1.loss_cls: 0.0268, s1.acc: 97.8314, s1.loss_bbox: 0.0461, s2.loss_cls: 0.0132, s2.acc: 97.8722, s2.loss_bbox: 0.0314, loss: 0.2433
2022-11-12 16:45:31,052 - mmdet - INFO - Epoch [12][750/962]	lr: 2.000e-04, eta: 0:02:22, time: 0.679, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0045, loss_rpn_bbox: 0.0062, s0.loss_cls: 0.0673, s0.acc: 97.3145, s0.loss_bbox: 0.0471, s1.loss_cls: 0.0297, s1.acc: 97.6991, s1.loss_bbox: 0.0443, s2.loss_cls: 0.0152, s2.acc: 97.7114, s2.loss_bbox: 0.0288, loss: 0.2431
2022-11-12 16:46:04,594 - mmdet - INFO - Epoch [12][800/962]	lr: 2.000e-04, eta: 0:01:49, time: 0.671, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0054, loss_rpn_bbox: 0.0046, s0.loss_cls: 0.0543, s0.acc: 97.8262, s0.loss_bbox: 0.0374, s1.loss_cls: 0.0214, s1.acc: 98.3498, s1.loss_bbox: 0.0354, s2.loss_cls: 0.0112, s2.acc: 98.2656, s2.loss_bbox: 0.0242, loss: 0.1939
2022-11-12 16:46:38,862 - mmdet - INFO - Epoch [12][850/962]	lr: 2.000e-04, eta: 0:01:15, time: 0.685, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0050, loss_rpn_bbox: 0.0061, s0.loss_cls: 0.0616, s0.acc: 97.5625, s0.loss_bbox: 0.0450, s1.loss_cls: 0.0246, s1.acc: 98.0132, s1.loss_bbox: 0.0441, s2.loss_cls: 0.0124, s2.acc: 98.0070, s2.loss_bbox: 0.0303, loss: 0.2290
2022-11-12 16:47:12,661 - mmdet - INFO - Epoch [12][900/962]	lr: 2.000e-04, eta: 0:00:41, time: 0.676, data_time: 0.012, memory: 3483, loss_rpn_cls: 0.0055, loss_rpn_bbox: 0.0054, s0.loss_cls: 0.0552, s0.acc: 97.8203, s0.loss_bbox: 0.0394, s1.loss_cls: 0.0223, s1.acc: 98.2646, s1.loss_bbox: 0.0380, s2.loss_cls: 0.0116, s2.acc: 98.2758, s2.loss_bbox: 0.0256, loss: 0.2029
2022-11-12 16:47:46,149 - mmdet - INFO - Epoch [12][950/962]	lr: 2.000e-04, eta: 0:00:08, time: 0.670, data_time: 0.011, memory: 3483, loss_rpn_cls: 0.0071, loss_rpn_bbox: 0.0074, s0.loss_cls: 0.0737, s0.acc: 97.0840, s0.loss_bbox: 0.0603, s1.loss_cls: 0.0306, s1.acc: 97.5252, s1.loss_bbox: 0.0543, s2.loss_cls: 0.0145, s2.acc: 97.7965, s2.loss_bbox: 0.0360, loss: 0.2839
2022-11-12 16:47:54,319 - mmdet - INFO - Saving checkpoint at 12 epochs
[>>] 200/200, 6.3 task/s, elapsed: 32s, ETA:     0s2022-11-12 16:48:31,325 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.67s).
Accumulating evaluation results...
DONE (t=0.04s).
2022-11-12 16:48:32,045 - mmdet - INFO -
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.136
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.312
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.103
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.136
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.242
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.242
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.242
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.242

2022-11-12 16:48:32,048 - mmdet - INFO - Exp name: rcnn_r50_caffe_fpn_mstrain-poly_1x_COCO.py
2022-11-12 16:48:32,048 - mmdet - INFO - Epoch(val) [12][200]	bbox_mAP: 0.1360, bbox_mAP_50: 0.3120, bbox_mAP_75: 0.1030, bbox_mAP_s: -1.0000, bbox_mAP_m: -1.0000, bbox_mAP_l: 0.1360, bbox_mAP_copypaste: 0.136 0.312 0.103 -1.000 -1.000 0.136