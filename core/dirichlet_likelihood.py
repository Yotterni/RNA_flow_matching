import torch
from torch import nn


class NegativeLogDirichletLikelihood(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """

        :param alpha: logarithm of alpha. Will NOT be exponentiated.
        :param target: smoothed degenerate distribution.
        :return:
        """
        # exponentiate alpha
        # alpha = log_alpha.exp()

        # calculating log(1 / Beta(a))
        neg_logbeta = -torch.lgamma(alpha).sum(dim=1) + torch.lgamma(alpha.sum(dim=1))
        # print(neg_logbeta.shape)

        # calculating x part. yup, here target is x (heh yoda style)
        x_part = ((alpha - 1) * target.log()).sum(dim=1)

        likelihood = neg_logbeta + x_part
        # print(likelihood)
        likelihood = likelihood.mean()
        return -likelihood