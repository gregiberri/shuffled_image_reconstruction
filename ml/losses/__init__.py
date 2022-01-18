# -*- coding: utf-8 -*-
# @Time    : 2021/11/11
# @Author  : Albert Gregus
# @Email   : g.albert95@gmail.com

import torch.nn as torch_losses
from ml.losses import weighted_losses


def get_loss(config):
    """
    Get the loss function according to the loss config name and parameters.

    :param config: config containing the loss name as config.name and the parameters as config.params
    :return: the loss function
    """
    if hasattr(torch_losses, str(config.name)):
        function = getattr(torch_losses, config.name)
        return function(**config.params.dict())
    elif hasattr(weighted_losses, str(config.name)):
        function = getattr(weighted_losses, config.name)
        return function(**config.params.dict())
    else:
        raise ValueError(f'Wrong loss name: {config.name}')
