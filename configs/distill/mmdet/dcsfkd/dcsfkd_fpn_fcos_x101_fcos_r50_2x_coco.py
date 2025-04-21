_base_ = ['./dcsfkkd_fpn_retina_x101_retina_r50_2x_coco.py']

teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco-511424d6.pth'  # noqa

model = dict(
    architecture=dict(
        cfg_path='mmdet::fcos/fcos_r50-caffe_fpn_gn-head_ms-640-800-2x_coco.py'),
    teacher=dict(
        cfg_path=  # noqa: E251
        'mmdet::fcos/fcos_r101-caffe_fpn_gn-head_ms-640-800-2x_coco.py'),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
            type='ConfigurableDistiller',
            student_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
            teacher_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
            distill_losses=dict(
                loss_pkd_fpn0=dict(type='DCSFKDLoss', loss_weight=2),
                loss_pkd_fpn1=dict(type='DCSFKDLoss', loss_weight=2),
                loss_pkd_fpn2=dict(type='DCSFKDLoss', loss_weight=2),
                loss_pkd_fpn3=dict(type='DCSFKDLoss', loss_weight=2)),
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


