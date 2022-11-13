!python tools/test.py \
    configs/COCO_Road/rcnn_r50_caffe_fpn_mstrain-poly_1x_COCO.py \
    work_dirs/rcnn_r50_caffe_fpn_mstrain-poly_1x_COCO/latest.pth \
    --eval bbox segm \
    --show-dir results/rcnn_r50_caffe_fpn_mstrain-poly_1x_COCO