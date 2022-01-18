import torch
from torch import nn
import torchvision.models as torch_models
from ml.models import mnist_resnet


def get_model(config, **kwargs):
    """
    Select the model according to the model config name and its parameters

    :param config: config containing the model name as config.name and the parameters as config.params
    :return: model
    """
    if hasattr(torch_models, config.name):
        function = getattr(torch_models, config.name)
        model = function(**config.params.dict())
        change_model_head(model, config)
        model = add_model_bottom(model, config)
        return model
    elif hasattr(mnist_resnet, config.name):
        function = getattr(mnist_resnet, config.name)
        model = function(**config.params.dict(), **kwargs)
        return model
    else:
        raise ValueError(f'Wrong model name in model configs: {config.name}')


def add_model_bottom(model, config):
    """
    Change the head (the last layer is different for different models in torchvision)
    """
    input_features = list(model.children())[0][0].in_channels
    if config.input_channels != input_features:
        return torch.nn.Sequential(torch.nn.Conv2d(config.input_channels, input_features,
                                                   kernel_size=3, stride=1, padding=1),
                                   model)
    else:
        raise ValueError('Can not find model last layer.')

def change_model_head(model, config):
    """
    Change the head (the last layer is different for different models in torchvision)
    """
    if hasattr(model, 'fc'):
        head = nn.Linear(model.fc.in_features, 1, bias=True)
        model.fc = head
    elif hasattr(model, 'classifier'):
        head = nn.Linear(model.classifier[-1].in_features, config.output_channels, bias=True)
        model.classifier[-1] = head
    else:
        raise ValueError('Can not find model last layer.')
