import torch
import torch.nn as nn
from diffusers import DDPMScheduler, UNet2DConditionModel

class FeatureAlignment(nn.Module):
    def __init__(self):
        super(FeatureAlignment, self).__init__()
        self.conv = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)

    def forward(self, image, pred):
        # Concatenate image and prediction along the channel dimension
        x = torch.cat([image, pred], dim=1)
        return self.conv(x)


# Define the conditional UNet model
class ConditionalUNet(nn.Module):
    def __init__(self):
        super(ConditionalUNet, self).__init__()
        # state_dict = torch.load("/home/zzl/codes/InterSegAdapter/pre_weights/diffusion_model.pt", map_location='cpu')
        # self.unet = UNet2DConditionModel.load_state_dict(state_dict)  # google/ddpm-celebahq-256
        self.unet = UNet2DConditionModel.from_pretrained("/home/zzl/codes/InterSegAdapter/pre_weights/ddpm-celebahq-256/bin")  # google/ddpm-celebahq-256
        self.align = FeatureAlignment()

    def forward(self, image, pred):
        aligned_features = self.align(image, pred)
        return self.unet(aligned_features).sample


# Create a test function
def test_conditional_unet():
    # Initialize the model
    model = ConditionalUNet()

    # Create dummy data
    image = torch.randn(1, 3, 256, 256)  # Batch size of 1, 3 channels (RGB), 256x256 resolution
    pred = torch.randn(1, 1, 256, 256)  # Batch size of 1, 1 channel (prediction mask), 256x256 resolution

    # Forward pass
    output = model(image, pred)

    print(f"Output shape: {output.shape}")


if __name__ == '__main__':
    # Run the test
    test_conditional_unet()