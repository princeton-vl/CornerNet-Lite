import torch
import torch.nn as nn

from .py_utils import TopPool, BottomPool, LeftPool, RightPool

from .py_utils.utils import convolution, residual, corner_pool
from .py_utils.losses import CornerNet_Saccade_Loss
from .py_utils.modules import saccade_net, saccade_module, saccade

def make_pool_layer(dim):
    return nn.Sequential()

def make_hg_layer(inp_dim, out_dim, modules):
    layers  = [residual(inp_dim, out_dim, stride=2)]
    layers += [residual(out_dim, out_dim) for _ in range(1, modules)]
    return nn.Sequential(*layers)

class model(saccade_net):
    def _pred_mod(self, dim):
        return nn.Sequential(
            convolution(3, 256, 256, with_bn=False),
            nn.Conv2d(256, dim, (1, 1))
        )

    def _merge_mod(self):
        return nn.Sequential(
            nn.Conv2d(256, 256, (1, 1), bias=False),
            nn.BatchNorm2d(256)
        )

    def __init__(self):
        stacks  = 3
        pre     = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(128, 256, stride=2)
        )
        hg_mods = nn.ModuleList([
            saccade_module(
                3, [256, 384, 384, 512], [1, 1, 1, 1],
                make_pool_layer=make_pool_layer,
                make_hg_layer=make_hg_layer
            ) for _ in range(stacks)
        ])
        cnvs    = nn.ModuleList([convolution(3, 256, 256) for _ in range(stacks)])
        inters  = nn.ModuleList([residual(256, 256) for _ in range(stacks - 1)])
        cnvs_   = nn.ModuleList([self._merge_mod() for _ in range(stacks - 1)])
        inters_ = nn.ModuleList([self._merge_mod() for _ in range(stacks - 1)])

        att_mods = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    convolution(3, 384, 256, with_bn=False),
                    nn.Conv2d(256, 1, (1, 1))
                ),
                nn.Sequential(
                    convolution(3, 384, 256, with_bn=False),
                    nn.Conv2d(256, 1, (1, 1))
                ),
                nn.Sequential(
                    convolution(3, 256, 256, with_bn=False),
                    nn.Conv2d(256, 1, (1, 1))
                )
            ]) for _ in range(stacks)
        ])
        for att_mod in att_mods:
            for att in att_mod:
                torch.nn.init.constant_(att[-1].bias, -2.19)

        hgs = saccade(pre, hg_mods, cnvs, inters, cnvs_, inters_) 

        tl_modules = nn.ModuleList([corner_pool(256, TopPool, LeftPool) for _ in range(stacks)])
        br_modules = nn.ModuleList([corner_pool(256, BottomPool, RightPool) for _ in range(stacks)])

        tl_heats = nn.ModuleList([self._pred_mod(80) for _ in range(stacks)])
        br_heats = nn.ModuleList([self._pred_mod(80) for _ in range(stacks)])
        for tl_heat, br_heat in zip(tl_heats, br_heats):
            torch.nn.init.constant_(tl_heat[-1].bias, -2.19)
            torch.nn.init.constant_(br_heat[-1].bias, -2.19)

        tl_tags = nn.ModuleList([self._pred_mod(1) for _ in range(stacks)])
        br_tags = nn.ModuleList([self._pred_mod(1) for _ in range(stacks)])

        tl_offs = nn.ModuleList([self._pred_mod(2) for _ in range(stacks)])
        br_offs = nn.ModuleList([self._pred_mod(2) for _ in range(stacks)])

        super(model, self).__init__(
            hgs, tl_modules, br_modules, tl_heats, br_heats, 
            tl_tags, br_tags, tl_offs, br_offs, att_mods
        )

        self.loss = CornerNet_Saccade_Loss(pull_weight=1e-1, push_weight=1e-1)
