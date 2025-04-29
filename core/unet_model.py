import torch
from IPython.core.pylabtools import figsize
from torch import nn

from torch.distributions.dirichlet import Dirichlet

from tqdm.auto import tqdm
import typing as tp

import matplotlib.pyplot as plt
import seaborn as sns


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels,
                               out_channels,
                               kernel_size=kernel_size,
                               padding=kernel_size // 2)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels,
                               out_channels,
                               kernel_size=kernel_size,
                               padding=kernel_size // 2)

        self.bn2 = nn.BatchNorm1d(out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu(x)
        x = self.bn2(x)
        return x


class DifferentKernelSizeResidualBlock(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_sizes: list[int],
                 ffn_hidden: int = 256) -> None:
        super().__init__()
        self.conv_blocks = nn.ModuleList([ConvBlock(in_channels,
                                                    out_channels,
                                                    kernel_size)
                                          for kernel_size in kernel_sizes])

        self.fcnn = nn.Sequential(
            nn.Linear(out_channels * len(kernel_sizes), ffn_hidden),  # а если не 1, а 3 поставить?
            nn.ReLU(),
            nn.Linear(ffn_hidden, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, out_channels),
        )
        self.skip_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.layernorm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_x = self.skip_conv(x)
        x = torch.cat(
            [conv_block(x) for conv_block in self.conv_blocks], dim=1)
        x = x.transpose(1, 2)
        x = self.fcnn(x)
        x = x + residual_x.transpose(1, 2)
        x = self.layernorm(x)
        return x.transpose(1, 2)


class UnetBlock(nn.Module):
    def __init__(self,
                 mode: str,
                 in_channels: int,
                 out_channels: int,
                 diff_kernel_size_block: bool = True) -> None:

        super().__init__()
        assert mode in ['downsample', 'upsample'], (
            'Choose either downsample or upsample')
        self.mode = mode
        if diff_kernel_size_block:
            self.block = DifferentKernelSizeResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_sizes=[3, 5, 7, 9])  # 7
        else:
            self.block = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3)

        self.nonparametric = \
            nn.MaxPool1d(2) if self.mode == 'downsample' else nn.Upsample(scale_factor=2)
        # print(self.nonparametric)

    def forward(self, x
                ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x_updated = self.nonparametric(x)
        # print('x_updated shape:', x_updated.shape)
        x_updated = self.block(x_updated)

        # if self.mode == 'downsample':
        #     return x_updated, x
        # else:
        return x_updated


def sinusoidal_embedding(n, d):
    wk = torch.tensor([1 / 10_000 ** (j / d) for j in range(d)])
    wk = wk.reshape(1, d).repeat(n, 1)
    t = torch.arange(n).reshape(n, 1)
    embedding = wk * t

    for i in range(0, d, 2):
        embedding[:, i] = torch.sin(embedding[:, i])
        if i + 1 < d:
            embedding[:, i + 1] = torch.cos(embedding[:, i + 1])
    return embedding


class TimeConcatUNet(nn.Module):
    def __init__(self,
                 n_steps: int = 1000,
                 time_emb_dim: int = 100,
                 hidden_dim: int = 128,
                 #  num_time_channels: int = ,
                 ) -> None:
        super().__init__()

        self.n_steps = n_steps
        self.time_emb_dim = time_emb_dim
        # self.num_time_channels = num_time_channels

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(
            self.n_steps, self.time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(
            self.n_steps, self.time_emb_dim)
        # self.time_embed.requires_grad_(False) # мы эмбединг слой уже инициализировали и менять его не будем

        # Sinusoidal embedding-processing FCNN
        self.te1 = self._make_te(self.time_emb_dim, 4)  # мб побольше всё-таки
        self.te5 = self._make_te(self.time_emb_dim, hidden_dim)

        self.downblock1 = UnetBlock('downsample', 8, hidden_dim)
        self.downblock2 = UnetBlock('downsample', hidden_dim, hidden_dim)
        self.downblock3 = UnetBlock('downsample', hidden_dim, hidden_dim)
        self.downblock4 = UnetBlock('downsample', hidden_dim, hidden_dim)

        self.upblock1 = UnetBlock('upsample', 2 * hidden_dim, hidden_dim)
        self.upblock2 = UnetBlock('upsample', hidden_dim, hidden_dim)
        self.upblock3 = UnetBlock('upsample', hidden_dim, hidden_dim)
        self.upblock4 = UnetBlock('upsample', hidden_dim, 4)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Concatenating time channels
        t = self.time_embed(t) #.view(x.shape[0], -1)

        # running UNet
        # print(x.shape, t.shape)
        time1 = self._adaptate_time(self.te1(t), x)
        # print(x.shape, time1.shape)
        z1 = self.downblock1(torch.cat([x, time1], dim=1))
        z2 = self.downblock2(z1)
        z3 = self.downblock3(z2)
        z4 = self.downblock4(z3)

        time5 = self._adaptate_time(self.te5(t), z4)
        y = self.upblock1(torch.cat([z4, time5], dim=1))
        y = self.upblock2(y + z3)
        y = self.upblock3(y + z2)
        y = self.upblock4(y + z1)
        return y

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )

    def _adaptate_time(self, time: torch.Tensor, x):
        return time.view(x.shape[0], -1, 1).repeat(
            1, 1, x.shape[-1])


class RNAFlowMatchingModel:
    def __init__(self, unet_hidden_dim: int,
                 num_time_steps: int = 200,
                 time_denominator: tp.Optional[int] = None,
                 device: str | torch.device = 'cuda:0') -> None:
        """

        :param unet_hidden_dim: hidden dimension for Unet-like model
        :param num_time_steps: number of integration steps.
        :param time_denominator: number of time step will be devided by this
         constant for obtaining Dirichlet distribution at time t. Such a division
         is required because time stamp is integer parameter that can be quite big,
         and it's direct substitution to Dirichlet likelihood will blow it up.
        """
        self.unet_hidden_dim = unet_hidden_dim
        self.num_time_steps = num_time_steps
        if time_denominator is None:
            self.time_denominator = self.num_time_steps // 10
        else:
            self.time_denominator = time_denominator
        self.device = device
        self.unet = TimeConcatUNet(hidden_dim=self.unet_hidden_dim).to(self.device)

    def draw_points_at_time_t(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        dirichlet_params_at_time_t = torch.ones_like(x).to(self.device) + x * time / self.time_denominator
        dirichlet_params_at_time_t = dirichlet_params_at_time_t.transpose(1, 2) # Dir distr across each channel, not length
        distr = Dirichlet(dirichlet_params_at_time_t)
        sequences_at_time_t = distr.sample().transpose(1, 2).to(self.device)
        return sequences_at_time_t.to(self.device)

    def predict_dirichlet_params(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        alphas = self.unet(x, time)
        return torch.abs(alphas)

    @torch.no_grad()
    def generate(self, num_samples: int = 1) -> torch.Tensor:
        x = torch.ones(num_samples, 4, 80).to(self.device)
        for t in range(0, self.num_time_steps):
            time = torch.tensor([t]).view(1, 1, 1).repeat(num_samples, 1, 1).to(self.device)
            log_alphas = self.predict_dirichlet_params(x, time)
            x = self.draw_points_at_time_t(log_alphas, time)
        return x #.argmax(dim=1)
