"""Data model architectures module."""

from ._rank_resnet3d import (RankResNet, rankresnet18, rankresnet34,
                             rankresnet50, rankresnet101, rankresnet152)
from ._rank_sfcn import RankSFCN
from ._resnet3d import (ResNet, resnet18, resnet34, resnet50, resnet101,
                        resnet152)
from ._sfcn import SFCN

__all__ = ['RankResNet',
           'rankresnet18',
           'rankresnet34',
           'rankresnet50',
           'rankresnet101',
           'rankresnet152',
           'RankSFCN',
           'ResNet',
           'resnet18',
           'resnet34',
           'resnet50',
           'resnet101',
           'resnet152',
           'SFCN']
