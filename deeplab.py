import torch
import torch.nn as nn
from torch.nn import functional as F


class ASPP(nn.Module):

    def __init__(self, in_channels, atrous_rates):
        
        super(ASPP, self).__init__()
        self.aspp_layers = nn.ModuleList()

        for rate in atrous_rates:
            self.aspp_layers.append(
                nn.Conv2d(
                    in_channels, 256, kernel_size=3, padding=rate, dilation=rate, bias=False
                )
            )


        self.global_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )


        self.conv1x1 = nn.Conv2d(256 * (len(atrous_rates) + 1), 256, kernel_size=1, bias=False)

    def forward(self, x):
        aspp_outs = [layer(x) for layer in self.aspp_layers]
        global_pooling = self.global_pooling(x)
        global_pooling = F.interpolate(global_pooling, size=x.size()[2:], mode="bilinear", align_corners=False)
        aspp_outs.append(global_pooling)

        x = torch.cat(aspp_outs, dim=1)
        x = self.conv1x1(x)
        return x


class DeepLabV3Plus(nn.Module):
   
    def __init__(self, in_channels=3, out_channels=1, atrous_rates=[6, 12, 18, 24]):
       
        super(DeepLabV3Plus, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(512, atrous_rates)

       
        self.upsample = nn.ConvTranspose2d(256, 1, kernel_size=16, stride=16, padding=0, bias=False)
        self.output_conv = nn.Conv2d(1, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        aspp_out = self.aspp(x5)
        upsampled = self.upsample(aspp_out)
        output = self.output_conv(upsampled)

        return torch.sigmoid(output)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepLabV3Plus(in_channels=3, out_channels=1).to(device)
    input_tensor = torch.randn(1, 3, 128, 128).to(device)  # Example input
    output = model(input_tensor)
    print("Output shape:", output.shape)  
