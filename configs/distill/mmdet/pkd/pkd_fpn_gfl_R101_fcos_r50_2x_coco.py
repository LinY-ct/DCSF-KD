_base_ = ['./pkd_fpn_retina_x101_retina_r50_2x_coco.py']

teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco_20200630_102002-134b07df.pth'  # noqa: E501

model = dict(
    architecture=dict(
        cfg_path='mmdet::fcos/fcos_r50-caffe_fpn_gn-head_ms-640-800-2x_coco.py'),
    teacher=dict(
        cfg_path=  # noqa: E251
        'mmdet::gfl/gfl_r101_fpn_ms-2x_coco.py'),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
            type='ConfigurableDistiller',
            student_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
            teacher_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
            distill_losses=dict(
                loss_pkd_fpn0=dict(type='PKDLoss', loss_weight=10),
                loss_pkd_fpn1=dict(type='PKDLoss', loss_weight=10),
                loss_pkd_fpn2=dict(type='PKDLoss', loss_weight=10),
                loss_pkd_fpn3=dict(type='PKDLoss', loss_weight=10)),
            loss_forward_mappings=dict(
                loss_pkd_fpn0=dict(
                    preds_S=dict(from_student=True, recorder='fpn', data_idx=0),
                    preds_T=dict(from_student=False, recorder='fpn', data_idx=0)),
                loss_pkd_fpn1=dict(
                    preds_S=dict(from_student=True, recorder='fpn', data_idx=1),
                    preds_T=dict(from_student=False, recorder='fpn', data_idx=1)),
                loss_pkd_fpn2=dict(
                    preds_S=dict(from_student=True, recorder='fpn', data_idx=2),
                    preds_T=dict(from_student=False, recorder='fpn', data_idx=2)),
                loss_pkd_fpn3=dict(
                    preds_S=dict(from_student=True, recorder='fpn', data_idx=3),
                    preds_T=dict(from_student=False, recorder='fpn',
                                 data_idx=3))))
)

# # training schedule for 1x
# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
#
# # learning rate
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=12,
#         by_epoch=True,
#         milestones=[8, 11],
#         gamma=0.1)
# ]

