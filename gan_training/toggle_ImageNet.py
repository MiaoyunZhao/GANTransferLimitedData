
import torch
import torch.utils.data
import torch.utils.data.distributed
import torchvision


def toggle_grad_D(model, requires_grad, fix_layer=0):
    if fix_layer==0:
        for p in model.parameters():
            p.requires_grad_(requires_grad)
    else:
        if fix_layer <= 7:
            for name, param in model.fc.named_parameters():
                param.requires_grad_(requires_grad)
        if fix_layer <= 6:
            for name, param in model.resnet_5_0.named_parameters():
                param.requires_grad_(requires_grad)
            for name, param in model.resnet_5_1.named_parameters():
                param.requires_grad_(requires_grad)
        if fix_layer <= 5:
            for name, param in model.resnet_4_0.named_parameters():
                param.requires_grad_(requires_grad)
            for name, param in model.resnet_4_1.named_parameters():
                param.requires_grad_(requires_grad)
        if fix_layer <= 4:
            for name, param in model.resnet_3_0.named_parameters():
                param.requires_grad_(requires_grad)
            for name, param in model.resnet_3_1.named_parameters():
                param.requires_grad_(requires_grad)
        if fix_layer <= 3:
            for name, param in model.resnet_2_0.named_parameters():
                param.requires_grad_(requires_grad)
            for name, param in model.resnet_2_1.named_parameters():
                param.requires_grad_(requires_grad)
        if fix_layer <= 2:
            for name, param in model.resnet_1_0.named_parameters():
                param.requires_grad_(requires_grad)
            for name, param in model.resnet_1_1.named_parameters():
                param.requires_grad_(requires_grad)
        if fix_layer <= 1:
            for name, param in model.resnet_0_0.named_parameters():
                param.requires_grad_(requires_grad)
            for name, param in model.resnet_0_1.named_parameters():
                param.requires_grad_(requires_grad)


def toggle_grad_G(model, requires_grad, fix_layer=0):
    if fix_layer == 0:
        for p in model.parameters():
            p.requires_grad_(requires_grad)
    else:
        if fix_layer >= -1:
            for name, param in model.resnet_5_0.named_parameters():
                param.requires_grad_(requires_grad)
            for name, param in model.resnet_5_1.named_parameters():
                param.requires_grad_(requires_grad)
        if fix_layer >= -2:
            for name, param in model.resnet_4_0.named_parameters():
                param.requires_grad_(requires_grad)
            for name, param in model.resnet_4_1.named_parameters():
                param.requires_grad_(requires_grad)
        if fix_layer >= -3:
            for name, param in model.resnet_3_0.named_parameters():
                param.requires_grad_(requires_grad)
            for name, param in model.resnet_3_1.named_parameters():
                param.requires_grad_(requires_grad)
        if fix_layer >= -4:
            for name, param in model.resnet_2_0.named_parameters():
                param.requires_grad_(requires_grad)
            for name, param in model.resnet_2_1.named_parameters():
                param.requires_grad_(requires_grad)
        if fix_layer >= -5:
            for name, param in model.resnet_1_0.named_parameters():
                param.requires_grad_(requires_grad)
            for name, param in model.resnet_1_1.named_parameters():
                param.requires_grad_(requires_grad)
        if fix_layer >= -6:
            for name, param in model.resnet_0_0.named_parameters():
                param.requires_grad_(requires_grad)
            for name, param in model.resnet_0_1.named_parameters():
                param.requires_grad_(requires_grad)
        if fix_layer >= -7:
            for name, param in model.fc.named_parameters():
                param.requires_grad_(requires_grad)


# def model_equal_part_G(model, dict_all, fix_layer=0):
#     model_dict = model.state_dict()
#     if fix_layer == -1:
#         dict_fix = {k: v for k, v in dict_all.items() if
#                 k in model_dict and k.find('embedding') == -1 and k.find('fc') == -1
#                     and k.find('resnet_0_0') == -1 and k.find('resnet_0_1') == -1
#                     and k.find('resnet_1_0') == -1 and k.find('resnet_1_1') == -1
#                     and k.find('resnet_2_0') == -1 and k.find('resnet_2_1') == -1
#                     and k.find('resnet_3_0') == -1 and k.find('resnet_3_1') == -1
#                     and k.find('resnet_4_0') == -1 and k.find('resnet_4_1') == -1
#                     and k.find('resnet_5_0') == -1 and k.find('resnet_5_1') == -1
#                     }
#     elif fix_layer == -2:
#         dict_fix = {k: v for k, v in dict_all.items() if
#                 k in model_dict and k.find('embedding') == -1 and k.find('fc') == -1
#                     and k.find('resnet_0_0') == -1 and k.find('resnet_0_1') == -1
#                     and k.find('resnet_1_0') == -1 and k.find('resnet_1_1') == -1
#                     and k.find('resnet_2_0') == -1 and k.find('resnet_2_1') == -1
#                     and k.find('resnet_3_0') == -1 and k.find('resnet_3_1') == -1
#                     and k.find('resnet_4_0') == -1 and k.find('resnet_4_1') == -1
#                     }
#     elif fix_layer == -3:
#         dict_fix = {k: v for k, v in dict_all.items() if
#                 k in model_dict and k.find('embedding') == -1 and k.find('fc') == -1
#                     and k.find('resnet_0_0') == -1 and k.find('resnet_0_1') == -1
#                     and k.find('resnet_1_0') == -1 and k.find('resnet_1_1') == -1
#                     and k.find('resnet_2_0') == -1 and k.find('resnet_2_1') == -1
#                     and k.find('resnet_3_0') == -1 and k.find('resnet_3_1') == -1
#                     }
#     elif fix_layer == -4:
#         dict_fix = {k: v for k, v in dict_all.items() if
#                 k in model_dict and k.find('embedding') == -1 and k.find('fc') == -1
#                     and k.find('resnet_0_0') == -1 and k.find('resnet_0_1') == -1
#                     and k.find('resnet_1_0') == -1 and k.find('resnet_1_1') == -1
#                     and k.find('resnet_2_0') == -1 and k.find('resnet_2_1') == -1
#                     }
#     elif fix_layer == -5:
#         dict_fix = {k: v for k, v in dict_all.items() if
#                 k in model_dict and k.find('embedding') == -1 and k.find('fc') == -1
#                     and k.find('resnet_0_0') == -1 and k.find('resnet_0_1') == -1
#                     and k.find('resnet_1_0') == -1 and k.find('resnet_1_1') == -1
#                     }
#     elif fix_layer == -6:
#         dict_fix = {k: v for k, v in dict_all.items() if
#                 k in model_dict and k.find('embedding') == -1 and k.find('fc') == -1
#                     and k.find('resnet_0_0') == -1 and k.find('resnet_0_1') == -1
#                     }
#
#     model_dict.update(dict_fix)
#     model.load_state_dict(model_dict)
#     return model


def model_equal_part_G(model, dict_all, fix_layer=0):
    model_dict = model.state_dict()
    dict_fix = {k: v for k, v in dict_all.items() if
                k in model_dict and k.find('embedding') == -1 and k.find('fc') == -1
                }
    model_dict.update(dict_fix)
    model.load_state_dict(model_dict)
    return model


# def model_equal_part_D(model, dict_all, fix_layer=0):
#     model_dict = model.state_dict()
#     if fix_layer == 4:
#         dict_fix = {k: v for k, v in dict_all.items() if
#                 k in model_dict and k.find('embedding') == -1 and k.find('fc') == -1
#                     and k.find('resnet_5_0') == -1 and k.find('resnet_5_1') == -1
#                     and k.find('resnet_4_0') == -1 and k.find('resnet_4_1') == -1
#                     and k.find('resnet_3_0') == -1 and k.find('resnet_3_1') == -1
#                     }
#     elif fix_layer == 3:
#         dict_fix = {k: v for k, v in dict_all.items() if
#                 k in model_dict and k.find('embedding') == -1 and k.find('fc') == -1
#                     and k.find('resnet_5_0') == -1 and k.find('resnet_5_1') == -1
#                     and k.find('resnet_4_0') == -1 and k.find('resnet_4_1') == -1
#                     and k.find('resnet_3_0') == -1 and k.find('resnet_3_1') == -1
#                     and k.find('resnet_2_0') == -1 and k.find('resnet_2_1') == -1
#                     }
#     elif fix_layer == 2:
#         dict_fix = {k: v for k, v in dict_all.items() if
#                 k in model_dict and k.find('embedding') == -1 and k.find('fc') == -1
#                     and k.find('resnet_5_0') == -1 and k.find('resnet_5_1') == -1
#                     and k.find('resnet_4_0') == -1 and k.find('resnet_4_1') == -1
#                     and k.find('resnet_3_0') == -1 and k.find('resnet_3_1') == -1
#                     and k.find('resnet_2_0') == -1 and k.find('resnet_2_1') == -1
#                     and k.find('resnet_1_0') == -1 and k.find('resnet_1_1') == -1
#                     }
#     elif fix_layer == 1:
#         dict_fix = {k: v for k, v in dict_all.items() if
#                 k in model_dict and k.find('embedding') == -1 and k.find('fc') == -1
#                     and k.find('resnet_5_0') == -1 and k.find('resnet_5_1') == -1
#                     and k.find('resnet_4_0') == -1 and k.find('resnet_4_1') == -1
#                     and k.find('resnet_3_0') == -1 and k.find('resnet_3_1') == -1
#                     and k.find('resnet_2_0') == -1 and k.find('resnet_2_1') == -1
#                     and k.find('resnet_1_0') == -1 and k.find('resnet_1_1') == -1
#                     and k.find('resnet_0_0') == -1 and k.find('resnet_0_1') == -1
#                     }
#
#     model_dict.update(dict_fix)
#     model.load_state_dict(model_dict)
#     return model


def model_equal_part_D(model, dict_all, fix_layer=0):
    model_dict = model.state_dict()
    dict_fix = {k: v for k, v in dict_all.items() if
                k in model_dict and k.find('embedding') == -1 and k.find('fc') == -1
                }
    model_dict.update(dict_fix)
    model.load_state_dict(model_dict)
    return model