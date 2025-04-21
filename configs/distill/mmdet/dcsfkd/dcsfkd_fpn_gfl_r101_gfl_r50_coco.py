_base_ = ['./dcsfkd_fpn_retina_x101_retina_r50_2x_coco.py']

teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco_20200630_102002-134b07df.pth'  # noqa: E501

model = dict(
    architecture=dict(
        cfg_path='mmdet::gfl/gfl_r50_fpn_1x_coco.py'),
    teacher=dict(
        cfg_path=  # noqa: E251
        'mmdet::gfl/gfl_r101_fpn_ms-2x_coco.py'),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
            type='ConfigurableDistiller',
            student_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
            teacher_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
            distill_losses=dict(
                loss_acd_fpn0=dict(type='DCSFKDLoss', loss_weight=2),
                loss_acd_fpn1=dict(type='DCSFKDLoss', loss_weight=2),
                loss_acd_fpn2=dict(type='DCSFKDLoss', loss_weight=2),
                loss_acd_fpn3=dict(type='DCSFKDLoss', loss_weight=2)),
            loss_forward_mappings=dict(
                loss_acd_fpn0=dict(
                    preds_S=dict(from_student=True, recorder='fpn', data_idx=0),
                    preds_T=dict(from_student=False, recorder='fpn', data_idx=0)),
                loss_acd_fpn1=dict(
                    preds_S=dict(from_student=True, recorder='fpn', data_idx=1),
                    preds_T=dict(from_student=False, recorder='fpn', data_idx=1)),
                loss_acd_fpn2=dict(
                    preds_S=dict(from_student=True, recorder='fpn', data_idx=2),
                    preds_T=dict(from_student=False, recorder='fpn', data_idx=2)),
                loss_acd_fpn3=dict(
                    preds_S=dict(from_student=True, recorder='fpn', data_idx=3),
                    preds_T=dict(from_student=False, recorder='fpn',
                                 data_idx=3))))
)


