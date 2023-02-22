_base_ = [
    '../_base_/datasets/semantickitti.py', '../_base_/models/cylinder3d.py',
    '../_base_/schedules/cyclic-20e.py', '../_base_/default_runtime.py'
]

load_from = 'checkpoints/converted_model.pth'
