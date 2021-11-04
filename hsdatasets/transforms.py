import torch

class ToTensor(object):
    """ Convert tuples of hyperspectral data and labels to tensors."""

    def __call__(self, sample):
        patch, label = sample
        patch = torch.from_numpy(patch.astype(float))
        label = torch.from_numpy(label)

        return (patch.type(torch.float32), label.type(torch.long))

class InsertEmptyChannelDim(object):
    """ Insert Empty Channel dimension to apply 3D-Convolutions to hyperspectral images tensors."""

    def __call__(self, sample):
        patch, label = sample
        patch = torch.unsqueeze(patch, 0)

        return (patch, label)

class PermuteData(object):
    """ Permutes sample-data dimensions as defined in `new_order`. """
    def __init__(self, new_order):
        self.new_order = new_order

    def __call__(self, sample):
        patch, label = sample
        patch = patch.permute(self.new_order)

        return (patch, label)

class ReplaceLabel(object):
    """ Replace label `orig_lbl` with `new_lbl`."""
    def __init__(self, orig_lbl, new_lbl):
        self.orig_lbl = orig_lbl
        self.new_lbl = new_lbl

    def __call__(self, sample):
        patch, label = sample
        label[label == self.orig_lbl] = self.new_lbl

        return (patch, label)
