import torch
import torch.nn as nn

from .utils import residual, upsample, merge, _decode

def _make_layer(inp_dim, out_dim, modules):
    layers  = [residual(inp_dim, out_dim)]
    layers += [residual(out_dim, out_dim) for _ in range(1, modules)]
    return nn.Sequential(*layers)

def _make_layer_revr(inp_dim, out_dim, modules):
    layers  = [residual(inp_dim, inp_dim) for _ in range(modules - 1)]
    layers += [residual(inp_dim, out_dim)]
    return nn.Sequential(*layers)

def _make_pool_layer(dim):
    return nn.MaxPool2d(kernel_size=2, stride=2)

def _make_unpool_layer(dim):
    return upsample(scale_factor=2)

def _make_merge_layer(dim):
    return merge()

class hg_module(nn.Module):
    def __init__(
        self, n, dims, modules, make_up_layer=_make_layer,
        make_pool_layer=_make_pool_layer, make_hg_layer=_make_layer,
        make_low_layer=_make_layer, make_hg_layer_revr=_make_layer_revr,
        make_unpool_layer=_make_unpool_layer, make_merge_layer=_make_merge_layer
    ):
        super(hg_module, self).__init__()

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.n    = n
        self.up1  = make_up_layer(curr_dim, curr_dim, curr_mod)
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(curr_dim, next_dim, curr_mod)
        self.low2 = hg_module(
            n - 1, dims[1:], modules[1:],
            make_up_layer=make_up_layer,
            make_pool_layer=make_pool_layer,
            make_hg_layer=make_hg_layer,
            make_low_layer=make_low_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer
        ) if n > 1 else make_low_layer(next_dim, next_dim, next_mod)
        self.low3 = make_hg_layer_revr(next_dim, curr_dim, curr_mod)
        self.up2  = make_unpool_layer(curr_dim)
        self.merg = make_merge_layer(curr_dim)

    def forward(self, x):
        up1  = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        merg = self.merg(up1, up2)
        return merg

class hg(nn.Module):
    def __init__(self, pre, hg_modules, cnvs, inters, cnvs_, inters_):
        super(hg, self).__init__()

        self.pre  = pre
        self.hgs  = hg_modules
        self.cnvs = cnvs

        self.inters  = inters
        self.inters_ = inters_
        self.cnvs_   = cnvs_

    def forward(self, x):
        inter = self.pre(x)

        cnvs  = []
        for ind, (hg_, cnv_) in enumerate(zip(self.hgs, self.cnvs)):
            hg  = hg_(inter)
            cnv = cnv_(hg)
            cnvs.append(cnv)

            if ind < len(self.hgs) - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = nn.functional.relu_(inter)
                inter = self.inters[ind](inter)
        return cnvs

class hg_net(nn.Module):
    def __init__(
        self, hg, tl_modules, br_modules, tl_heats, br_heats, 
        tl_tags, br_tags, tl_offs, br_offs
    ):
        super(hg_net, self).__init__()

        self._decode = _decode

        self.hg = hg

        self.tl_modules = tl_modules
        self.br_modules = br_modules

        self.tl_heats = tl_heats
        self.br_heats = br_heats

        self.tl_tags = tl_tags
        self.br_tags = br_tags
        
        self.tl_offs = tl_offs
        self.br_offs = br_offs

    def _train(self, *xs):
        image = xs[0]
        cnvs  = self.hg(image)

        tl_modules = [tl_mod_(cnv) for tl_mod_, cnv in zip(self.tl_modules, cnvs)]
        br_modules = [br_mod_(cnv) for br_mod_, cnv in zip(self.br_modules, cnvs)]
        tl_heats   = [tl_heat_(tl_mod) for tl_heat_, tl_mod in zip(self.tl_heats, tl_modules)]
        br_heats   = [br_heat_(br_mod) for br_heat_, br_mod in zip(self.br_heats, br_modules)]
        tl_tags    = [tl_tag_(tl_mod)  for tl_tag_,  tl_mod in zip(self.tl_tags,  tl_modules)]
        br_tags    = [br_tag_(br_mod)  for br_tag_,  br_mod in zip(self.br_tags,  br_modules)]
        tl_offs    = [tl_off_(tl_mod)  for tl_off_,  tl_mod in zip(self.tl_offs,  tl_modules)]
        br_offs    = [br_off_(br_mod)  for br_off_,  br_mod in zip(self.br_offs,  br_modules)]
        return [tl_heats, br_heats, tl_tags, br_tags, tl_offs, br_offs]

    def _test(self, *xs, **kwargs):
        image = xs[0]
        cnvs  = self.hg(image)

        tl_mod = self.tl_modules[-1](cnvs[-1])
        br_mod = self.br_modules[-1](cnvs[-1])

        tl_heat, br_heat = self.tl_heats[-1](tl_mod), self.br_heats[-1](br_mod)
        tl_tag,  br_tag  = self.tl_tags[-1](tl_mod),  self.br_tags[-1](br_mod)
        tl_off,  br_off  = self.tl_offs[-1](tl_mod),  self.br_offs[-1](br_mod)

        outs = [tl_heat, br_heat, tl_tag, br_tag, tl_off, br_off]
        return self._decode(*outs, **kwargs), tl_heat, br_heat, tl_tag, br_tag

    def forward(self, *xs, test=False, **kwargs):
        if not test:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)

class saccade_module(nn.Module):
    def __init__(
        self, n, dims, modules, make_up_layer=_make_layer,
        make_pool_layer=_make_pool_layer, make_hg_layer=_make_layer,
        make_low_layer=_make_layer, make_hg_layer_revr=_make_layer_revr,
        make_unpool_layer=_make_unpool_layer, make_merge_layer=_make_merge_layer
    ):
        super(saccade_module, self).__init__()

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.n    = n
        self.up1  = make_up_layer(curr_dim, curr_dim, curr_mod)
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(curr_dim, next_dim, curr_mod)
        self.low2 = saccade_module(
            n - 1, dims[1:], modules[1:],
            make_up_layer=make_up_layer,
            make_pool_layer=make_pool_layer,
            make_hg_layer=make_hg_layer,
            make_low_layer=make_low_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer
        ) if n > 1 else make_low_layer(next_dim, next_dim, next_mod)
        self.low3 = make_hg_layer_revr(next_dim, curr_dim, curr_mod)
        self.up2  = make_unpool_layer(curr_dim)
        self.merg = make_merge_layer(curr_dim)

    def forward(self, x):
        up1  = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        if self.n > 1:
            low2, mergs = self.low2(low1)
        else:
            low2, mergs = self.low2(low1), []
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        merg = self.merg(up1, up2)
        mergs.append(merg)
        return merg, mergs

class saccade(nn.Module):
    def __init__(self, pre, hg_modules, cnvs, inters, cnvs_, inters_):
        super(saccade, self).__init__()

        self.pre  = pre
        self.hgs  = hg_modules
        self.cnvs = cnvs

        self.inters  = inters
        self.inters_ = inters_
        self.cnvs_   = cnvs_

    def forward(self, x):
        inter = self.pre(x)

        cnvs  = []
        atts  = []
        for ind, (hg_, cnv_) in enumerate(zip(self.hgs, self.cnvs)):
            hg, ups = hg_(inter)
            cnv = cnv_(hg)
            cnvs.append(cnv)
            atts.append(ups)

            if ind < len(self.hgs) - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = nn.functional.relu_(inter)
                inter = self.inters[ind](inter)
        return cnvs, atts

class saccade_net(nn.Module):
    def __init__(
        self, hg, tl_modules, br_modules, tl_heats, br_heats, 
        tl_tags, br_tags, tl_offs, br_offs, att_modules, up_start=0
    ):
        super(saccade_net, self).__init__()

        self._decode = _decode

        self.hg = hg

        self.tl_modules = tl_modules
        self.br_modules = br_modules
        self.tl_heats   = tl_heats
        self.br_heats   = br_heats
        self.tl_tags    = tl_tags
        self.br_tags    = br_tags
        self.tl_offs    = tl_offs
        self.br_offs    = br_offs

        self.att_modules = att_modules
        self.up_start    = up_start

    def _train(self, *xs):
        image = xs[0]

        cnvs, ups  = self.hg(image)
        ups = [up[self.up_start:] for up in ups]

        tl_modules = [tl_mod_(cnv) for tl_mod_, cnv in zip(self.tl_modules, cnvs)]
        br_modules = [br_mod_(cnv) for br_mod_, cnv in zip(self.br_modules, cnvs)]
        tl_heats   = [tl_heat_(tl_mod) for tl_heat_, tl_mod in zip(self.tl_heats, tl_modules)]
        br_heats   = [br_heat_(br_mod) for br_heat_, br_mod in zip(self.br_heats, br_modules)]
        tl_tags    = [tl_tag_(tl_mod)  for tl_tag_,  tl_mod in zip(self.tl_tags,  tl_modules)]
        br_tags    = [br_tag_(br_mod)  for br_tag_,  br_mod in zip(self.br_tags,  br_modules)]
        tl_offs    = [tl_off_(tl_mod)  for tl_off_,  tl_mod in zip(self.tl_offs,  tl_modules)]
        br_offs    = [br_off_(br_mod)  for br_off_,  br_mod in zip(self.br_offs,  br_modules)]
        atts       = [[att_mod_(u) for att_mod_, u in zip(att_mods, up)] for att_mods, up in zip(self.att_modules, ups)]
        return [tl_heats, br_heats, tl_tags, br_tags, tl_offs, br_offs, atts]

    def _test(self, *xs, no_att=False, **kwargs):
        image = xs[0]
        cnvs, ups = self.hg(image)
        ups = [up[self.up_start:] for up in ups]

        if not no_att:
            atts = [att_mod_(up) for att_mod_, up in zip(self.att_modules[-1], ups[-1])]
            atts = [torch.sigmoid(att) for att in atts]

        tl_mod = self.tl_modules[-1](cnvs[-1])
        br_mod = self.br_modules[-1](cnvs[-1])

        tl_heat, br_heat = self.tl_heats[-1](tl_mod), self.br_heats[-1](br_mod)
        tl_tag,  br_tag  = self.tl_tags[-1](tl_mod),  self.br_tags[-1](br_mod)
        tl_off,  br_off  = self.tl_offs[-1](tl_mod),  self.br_offs[-1](br_mod)

        outs = [tl_heat, br_heat, tl_tag, br_tag, tl_off, br_off]
        if not no_att:
            return self._decode(*outs, **kwargs), atts
        else:
            return self._decode(*outs, **kwargs)

    def forward(self, *xs, test=False, **kwargs):
        if not test:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)
