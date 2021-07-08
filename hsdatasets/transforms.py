import torch

class ToTensor(object):
    """ Convert tuples of hyperspectral data and labels to tensors."""

    def __call__(self, sample):
        patch, label = sample
        patch = torch.from_numpy(patch.astype(float))
        label = torch.from_numpy(label)

        return (patch.type(torch.float32), label.type(torch.long))
