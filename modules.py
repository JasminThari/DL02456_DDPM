import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, img_dim=64, num_max_pools=3, initial_feature_maps=64, time_dim=256,
                 device="cpu"):
        # num_max_pools is the number of times we downsample the image
        # make an reasionable assumption about the size of the image and choose num_max_pools accordingly
        # as an example an 64 x 64 image would go from 64 -> 32 -> 16 -> 8 with 3 max pools
        # larger images would presumably need more max pools

        # check that the image is divisible by 2^num_max_pools
        assert img_dim % (2 ** num_max_pools) == 0, "Image size not divisible by 2^num_max_pools"

        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.img_dim = img_dim  # the width and height of the image (assuming square)
        self.num_max_pools = num_max_pools  # the number of times we downsample the image
        self.initial_feature_maps = initial_feature_maps  # was 64 but can be changed. Is doubled each time we downsample
        self.channels_into_bottleneck = initial_feature_maps * (
                    2 ** (self.num_max_pools - 1))  # the number of channels going into the bottleneck

        # Encoder
        self.encoder = self.make_encoder(c_in)

        # Bottleneck
        self.bottle = nn.Sequential(
            DoubleConv(self.channels_into_bottleneck, 2 * self.channels_into_bottleneck),
            DoubleConv(2 * self.channels_into_bottleneck, 2 * self.channels_into_bottleneck),
            DoubleConv(2 * self.channels_into_bottleneck, self.channels_into_bottleneck))

        # Decoder
        self.decoder = self.make_decoder(c_out)

    def make_encoder(self, c_in):
        encoder = nn.ModuleList()
        # add first double conv layer
        encoder.append(DoubleConv(c_in, self.initial_feature_maps))
        # add the rest of the layers except the last one
        for i in range(1, self.num_max_pools, 1):
            encoder.append(Down(self.initial_feature_maps * (2 ** (i - 1)), self.initial_feature_maps * (2 ** i)))
            encoder.append(SelfAttention(self.initial_feature_maps * (2 ** i), self.img_dim // (2 ** (i))))

        # last layer is different, here the number of channels is not doubled when we downsample
        encoder.append(Down(self.initial_feature_maps * (2 ** i), self.initial_feature_maps * (2 ** i)))
        encoder.append(SelfAttention(self.initial_feature_maps * (2 ** i), self.img_dim // (2 ** (i + 1))))

        return encoder

    def make_decoder(self, c_out):
        decoder = nn.ModuleList()
        # add the layers
        # loop backwards
        for i in range(self.num_max_pools, 1, -1):
            decoder.append(Up(self.initial_feature_maps * (2 ** i), self.initial_feature_maps * (2 ** (i - 2))))
            decoder.append(SelfAttention(self.initial_feature_maps * (2 ** (i - 2)), self.img_dim // (2 ** (i - 1))))

        # add the second last layer
        decoder.append(Up(self.initial_feature_maps * 2, self.initial_feature_maps))
        decoder.append(SelfAttention(self.initial_feature_maps, self.img_dim))
        # finally add the last layey which is a double conv layer
        decoder.append(nn.Conv2d(self.initial_feature_maps, c_out, kernel_size=1))
        return decoder

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        skips = []
        # go through the encoder
        for idx, layer in enumerate(self.encoder):
            if isinstance(layer, DoubleConv):
                x = layer(x)
                skips.append(x)
            elif isinstance(layer, Down):
                x = layer(x, t)
            elif isinstance(layer, SelfAttention):
                x = layer(x)
                if idx != len(self.encoder) - 1:
                    skips.append(x)

        # go through the bottleneck
        x = self.bottle(x)

        # go through the decoder
        for layer in self.decoder:
            if isinstance(layer, Up):
                x = layer(x, skips.pop(), t)
            elif isinstance(layer, SelfAttention):
                x = layer(x)
            elif isinstance(layer, nn.Conv2d):
                x = layer(x)

        return x


if __name__ == '__main__':
    batch_size = 2
    img_channels = 3
    img_dim = 64
    num_max_pools = 3  # 3 is fitting for images of 64x64,but for 28x28 we can only do 2.
    x_org = torch.randn(batch_size, img_channels, img_dim, img_dim)
    timesteps = torch.randint(0, 500, (batch_size,))
    # create the uNet
    net = UNet(c_in=img_channels, c_out=img_channels, img_dim=img_dim, num_max_pools=num_max_pools,device="cpu")  # remember to set both in and out channels to 1 when MNIST
    x_new = net(x_org, timesteps)


