import torchvision.transforms.functional as TF
import random
import torch


class RandomChoiceRotate(torch.nn.Module):
    """
    Rotate the image with one of the given rotation degrees.

    Args:
        degree_choices (list): degrees of rotation to choose from
    """

    def __init__(self, degree_choices) -> None:
        super().__init__()
        self.degree_choices = degree_choices

    def forward(self, image):
        angle = random.choice(self.degree_choices)
        return TF.rotate(image, angle)
