from warnings import warn

from torchvision import transforms

from data.transforms.random_choice_rotation import RandomChoiceRotate


def make_transform_composition(transformation_list, split):
    """
    Just casually iterating through the transformations from the transform list, load them and make the composition.
    :param transformation_list: list containing lists of the function names and parameters:
    [['function1', [function1_params], ['function2', [function2_params]. ...]

    :param split: in which split we are: `train`, `val` or `test`
    :return: transform
    """
    compose_list = []
    if split == 'val' or split == 'test':
        transformation_list = transformation_list[-2:]

    for item in transformation_list:
        if hasattr(transforms, item[0]):
            function = getattr(transforms, item[0])
            compose_list.append(function(*item[1]))
        elif item[0] == 'RandomChoiceRotation':
            compose_list.append(RandomChoiceRotate(*item[1]))
        else:
            warn(f'Skipping unknown transform: {item[0]}.')
    return transforms.Compose(compose_list)

