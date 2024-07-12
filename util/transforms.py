import torch

class RandomZeroTransform:
    """Transform to zero-out an image with a given probability."""

    def __init__(self, probability=0.15):
        assert 0 <= probability <= 1, 'Probability must be between 0 and 1'
        self.probability = probability

    def __call__(self, y):
        if torch.rand(1) < self.probability:
            return torch.zeros_like(y)
        return y