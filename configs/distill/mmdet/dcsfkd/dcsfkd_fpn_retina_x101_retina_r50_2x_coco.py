_base_ = ['./dcsfkd_fpn_faster-rcnn_r101_faster-rcnn_r50_2x_coco.py']

teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_1x_coco/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth'  # noqa: E501

model = dict(
    architecture=dict(
        cfg_path='mmdet::retinanet/retinanet_r50_fpn_2x_coco.py'),
    teacher=dict(
        cfg_path='mmdet::retinanet/retinanet_x101-64x4d_fpn_1x_coco.py'),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        distill_losses=dict(
            loss_pkd_fpn0=dict(type='DCSFKDLoss', loss_weight=2),
            loss_pkd_fpn1=dict(type='DCSFKDLoss', loss_weight=2),
            loss_pkd_fpn2=dict(type='DCSFKDLoss', loss_weight=2),
            loss_pkd_fpn3=dict(type='DCSFKDLoss', loss_weight=2))))

# optimizer
optim_wrapper = dict(optimizer=dict(lr=0.01))
