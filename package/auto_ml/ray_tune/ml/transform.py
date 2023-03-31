import torch


class ResizeChannelTo3(torch.nn.Module):
    def __init__(self, size=3):
        super().__init__()
        self.size = size

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        shape = list(img.shape)
        shape[0] = 3
        data = torch.zeros(shape)
        data[0, :, :] = img
        data[1, :, :] = img
        data[2, :, :] = img
        return data
