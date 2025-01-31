import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
import numpy as np
# from collections import OrderedDict

# Convolutional block: Conv2D -> BatchNorm -> ReLU
class convblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(convblock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x

# Decoder block: Upsampling -> Skip connection -> Channel reduction -> Convolutions
class stackDecoder(nn.Module):
    def __init__(self, big_channel, channel1, channel2, kernel_size=3, padding=1):
        super(stackDecoder, self).__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels=big_channel,
            out_channels=big_channel,
            kernel_size=2,
            stride=2
        )
        self.reduce_channels = nn.Conv2d(big_channel + channel1, channel2, kernel_size=1)
        self.block = nn.Sequential(
            convblock(channel2, channel2, kernel_size, padding),
            convblock(channel2, channel2, kernel_size, padding),
        )

    def forward(self, x, dow_tensor):
        x = self.upsample(x)  # Upsample the input
        dow_tensor = F.interpolate(dow_tensor, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, dow_tensor], dim=1)  # Concatenate skip connection
        x = self.reduce_channels(x)  # Reduce channels with 1x1 convolution
        x = self.block(x)  # Apply convolutional layers
        return x

# U-Net with a pretrained VGG16 encoder
class UnetVGG16(nn.Module):
    def __init__(self):
        super(UnetVGG16, self).__init__()
        # Load pretrained VGG16 model
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        vgg_features = vgg16.features

        # Define the encoder blocks using VGG16 feature layers
        self.down1 = nn.Sequential(*vgg_features[:5])   # Output: [64, 256, 256]
        self.down2 = nn.Sequential(*vgg_features[5:10]) # Output: [128, 128, 128]
        self.down3 = nn.Sequential(*vgg_features[10:17])# Output: [256, 64, 64]
        self.down4 = nn.Sequential(*vgg_features[17:24])# Output: [512, 32, 32]
        self.down5 = nn.Sequential(*vgg_features[24:31])# Output: [512, 16, 16]

        # Bottleneck layer
        self.center = convblock(512, 512)

        # Define the decoder blocks
        self.up5 = stackDecoder(big_channel=512, channel1=512, channel2=256)  # Output: [256, 32, 32]
        self.up4 = stackDecoder(big_channel=256, channel1=512, channel2=128)  # Output: [128, 64, 64]
        self.up3 = stackDecoder(big_channel=128, channel1=256, channel2=64)   # Output: [64, 128, 128]
        self.up2 = stackDecoder(big_channel=64, channel1=128, channel2=32)    # Output: [32, 256, 256]
        self.up1 = stackDecoder(big_channel=32, channel1=64, channel2=16)     # Output: [16, 256, 256]

        # Final convolutional layer
        self.conv = nn.Conv2d(16, 1, kernel_size=1, bias=True)

    def forward(self, x):
        # Encoder
        down1 = self.down1(x)    # [64, 256, 256]
        down2 = self.down2(down1) # [128, 128, 128]
        down3 = self.down3(down2) # [256, 64, 64]
        down4 = self.down4(down3) # [512, 32, 32]
        down5 = self.down5(down4) # [512, 16, 16]

        # Bottleneck
        out = self.center(down5)  # [512, 16, 16]

        # Decoder
        up5 = self.up5(out, down5)  # [256, 32, 32]
        up4 = self.up4(up5, down4)  # [128, 64, 64]
        up3 = self.up3(up4, down3)  # [64, 128, 128]
        up2 = self.up2(up3, down2)  # [32, 256, 256]
        up1 = self.up1(up2, down1)  # [16, 256, 256]

        # Final output
        out = self.conv(up1)        # [1, 256, 256]
        out = torch.sigmoid(out)    # Apply sigmoid for binary segmentation
        return out

# Load Model
def load_model(model_path, device):
    model = UnetVGG16().to(device)  # Define model and move to device
    model.eval()  # Set to evaluation mode

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Handle DataParallel "module." prefix issue
    if any(k.startswith("module.") for k in checkpoint.keys()):
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}

    model.load_state_dict(checkpoint, strict=True)  # Load weights

    return model

# Image Preprocessing Function
def process_image(image):
    transform = Compose([
        Resize(256, 256),
        Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255),
        ToTensorV2()
    ])
    
    # ✅ If input is a NumPy array, use it directly
    if isinstance(image, np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure RGB format
    else:
        image = cv2.imread(image)  # Load from file path if needed
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply transformations
    transformed = transform(image=image)
    image_tensor = transformed["image"].unsqueeze(0).to(torch.device("cpu"))  # Convert to tensor
    return image_tensor


# Run Model and Generate Outputs
def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor.float())
        output_mask = (output >= 0.5).float()
        return output_mask.squeeze().cpu().numpy()

# Function to Overlay Mask on MRI Image
def overlay_images(model_input, predicted_mask):
    model_input_np = model_input.squeeze().cpu().numpy()
    model_input_np = np.transpose(model_input_np, (1, 2, 0))
    model_input_np = ((model_input_np - model_input_np.min()) / (model_input_np.max() - model_input_np.min()) * 255).astype(np.uint8)

    mask = (predicted_mask * 255).astype(np.uint8)
    red_mask = np.zeros_like(model_input_np)
    red_mask[:, :, 0] = mask

    return cv2.addWeighted(model_input_np, 1, red_mask, 0.5, 0)
