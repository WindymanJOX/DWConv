import torch
import torch.nn as nn


class DWConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size = 3):
        super().__init__()
        self.dwconv = nn.Conv2d(in_c, in_c, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=in_c)

        num_groups = min(32, in_c)
        self.norm = nn.GroupNorm(num_groups, in_c)

        self.pwconv = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0)
        self.act = nn.GELU()
    
    def forward(self, x):
        x0 = self.dwconv(x)
        x0 = self.norm(x0)
        x1 = self.pwconv(x0)
        x1 = self.act(x1)

        return x1
    

class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, blocks = 2, kernel_size = 3):
        super().__init__()
        self.convs = nn.ModuleList()

        for _ in range(blocks):
            self.convs.append(DWConvBlock(in_c, out_c, kernel_size=kernel_size))
            in_c = out_c

        self.down_sample = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)

        x_down = self.down_sample(x)

        return x, x_down


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, blocks = 2, kernel_size = 3):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_c, in_c//2, kernel_size = 2, stride=2)
        self.convs = nn.ModuleList()

        for _ in range(blocks):
            self.convs.append(DWConvBlock(in_c, out_c, kernel_size))
            in_c = out_c

        
    def forward(self, x, skip):
        x = self.up_conv(x)
        x = torch.cat([x, skip], dim=1)
        for conv in self.convs:
            x = conv(x)
        return x


class DWEncoder(nn.Module):
    def __init__(self, in_c, base_c = 32, blocks = [2,2,4,2], channels=[32, 64, 128, 256], kernel_size = 3):
        super().__init__()

        self.stem = DWConvBlock(in_c, base_c, kernel_size=5)
        self.encoders = nn.ModuleList()

        in_c = base_c
        for block_count, channel in zip(blocks, channels):
            self.encoders.append(DownBlock(in_c, channel, block_count, kernel_size=kernel_size))
            in_c = channel
        
        self.bridge = DWConvBlock(in_c, in_c*2)

    def forward(self, x):
        skips = []

        x = self.stem(x)

        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)
        
        x = self.bridge(x)
        
        return x, skips
    

class DWDecoder(nn.Module):
    def __init__(self, out_c, blocks = [2,2,2,2], channels=[32, 64, 128, 256], kernel_size = 3, if_ds = False):
        super().__init__()
        self.if_ds = if_ds
        self.decoders = nn.ModuleList()

        channels = channels[::-1]
        in_c = channels[0]*2
        for block_count, channel in zip(blocks, channels):
            self.decoders.append(UpBlock(in_c, channel, block_count, kernel_size))
            in_c = channel
        
        self.final_conv = nn.Conv2d(channels[-1], out_c, 1)
    
    def forward(self, x, skips):
        outs = []
        for skip, decoder in zip(skips[::-1], self.decoders):
            x = decoder(x, skip)
            outs.append(x)
        x_out = self.final_conv(x)

        if self.if_ds:
            outs.append(x_out)
            return outs[::-1]
        else:
            return x_out
        
        
class DWUNet(nn.Module):
    def __init__(self, in_c, out_c, base_c = 32,
                 encoder_blocks = [2, 2, 4, 2],
                 decoder_blocks = [2, 2, 2, 2],
                 channels=[32, 64, 128, 256],
                 if_ds = False):
        super().__init__()

        self.is_ds = if_ds

        self.encoder = DWEncoder(in_c, base_c, encoder_blocks, channels)
        self.decoder = DWDecoder(out_c, decoder_blocks, channels, 3, if_ds)

    def forward(self, x):
        x, skips = self.encoder(x)
        return self.decoder(x, skips)        


# if __name__ == '__main__':
#     model = DWUNet(1, 2)
    # with open('./network_structure.txt', 'a') as f:
    #     print(model, file=f)
    # data = torch.rand((1,1,128,128))
    # output = model(data)
    # print(output.shape)