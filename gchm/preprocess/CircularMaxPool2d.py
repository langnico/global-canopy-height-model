import torch
import torch.nn as nn


class CircularMaxPool2d(nn.Module):
    """Note: The default settings of nan_value and kernel values are only valid for positive input data."""
    
    def __init__(self, radius, batch=1, channels=1, nan_value=-99):
        super(CircularMaxPool2d, self).__init__()
        self.radius = radius
        self.batch = batch
        self.channels = channels
        self.kernel = nn.Parameter(self.create_kernel(), requires_grad=False)
        self.nan_value = nan_value
        
    def create_kernel(self):
        kernel = torch.zeros(1, 1, 1, 1, 2*self.radius+1, 2*self.radius+1)
        y, x = torch.meshgrid(torch.linspace(-self.radius, self.radius, 2*self.radius+1),
                            torch.linspace(-self.radius, self.radius, 2*self.radius+1))
        mask = x**2 + y**2 <= self.radius**2
        kernel[:, :, :, :, ~mask] = -float('inf')
        return kernel

    def forward(self, x):
        
        # nan to num
        x[x.isnan()] = self.nan_value
                
        # Pad the input image to ensure that the output has the same size
        pad = self.radius
        x = nn.functional.pad(x, (pad, pad, pad, pad), mode='reflect')
        
        # Apply max-pooling using the kernel
        # unfold the image to obtain a tensor of size (batch, channels, height, width, patch_size, patch_size)
        patches = x.unfold(2, 2*self.radius+1, 1).unfold(3, 2*self.radius+1, 1)        
        # mask the patches with a circular mask that sets all invalid pixels to -inf
        masked = (patches + self.kernel)
        # compute the max within each patch (i.e. max along the last to dimensions)
        maxpooled = torch.amax(masked, dim=(4, 5))
        # Set nan_value to nan
        maxpooled[maxpooled == self.nan_value] = float('nan')
        
        return maxpooled


if __name__ == '__main__':

    # Example how to use this class

    # create a tensor that is assumed to have e.g. 1m GSD with shape (samples, channels, height, width)
    tensor_1m_GSD = torch.randint(low=0, high=50, size=(1, 1, 1000, 1000), dtype=torch.float32)

    # apply circular max pooling with GEDI footprint (25 diameter)
    circular_max_pool = CircularMaxPool2d(radius=12)

    tensor_1m_GSD_pooled = circular_max_pool(tensor_1m_GSD)
    print(tensor_1m_GSD_pooled.size())
