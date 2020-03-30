import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
# from ..utils import flatten, unflatten_like


def swag_parameters(model):
    swag_params = {}
    for layer, m in model.named_modules():
        if len(m._parameters) == 0:
            continue
        for name in m._parameters:
            if m._parameters[name] is None:
                continue
            data = m._parameters[name].data
            m.register_buffer(f"{name}_mean", data.new_zeros(data.size()))
            m.register_buffer(f"{name}_sq_mean", data.new_zeros(data.size()))
            m.register_buffer(f"{name}_cov_sqrt", data.new_zeros((0, data.numel())))

        swag_params[layer] = m
    return swag_params

class SWAG(nn.Module):
    '''
    SWA-Gaussian
    Reference:
        https://github.com/wjmaddox/swa_gaussian/blob/master/swag/posteriors/swag.py
    
    Args:
        model (nn.Module): the target model to be applied SWAG
        K (int): number of rank, to use low rank of cov matrix in paper
    '''
    def __init__(self, model, K=0, var_clamp=1e-30):
        super(SWAG, self).__init__()
        self.register_buffer("n", torch.zeros([1], dtype=torch.long))
        self.K = K
        self.model = model
        self.swag_params = swag_parameters(self.model)
        self.initialized = False

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def save(self, path, epoch=-1):
        torch.save(self.model.state_dict(), os.path.join(path,f"swag_{self.n.item():03}.pth"))

    def sample(self, scale=1.0, block=False):
        if not block:
            self.sample_fullrank(scale)
        else:
            self.sample_blockwise(scale)

    def sample_blockwise(self, scale):
        scale = scale ** 0.5
        for layer in self.swag_params:
            m = self.swag_params[layer]
            for name in m._parameters:
                if m._parameters[name] is None:
                    continue

                mean = m.__getattr__(f"{name}_mean")
                sq_mean = m.__getattr__(f"{name}_sq_mean")

                z1 = torch.randn_like(mean)
                cov_diag = torch.clamp(sq_mean - mean ** 2, 1e-30)
                cov_diag_sqrt = torch.sqrt(cov_diag) * z1

                cov_sqrt = m.__getattr__(f"{name}_cov_sqrt")
                z2 = cov_sqrt.new_empty((cov_sqrt.size(0), 1)).normal_()
                cov_low_rank_sqrt = cov_sqrt.t().matmul(z2).view_as(mean) / ((self.K - 1) ** 0.5)

                # eq(1)
                w = mean + scale * (cov_diag_sqrt + cov_low_rank_sqrt)
                m._parameters[name].data = w

    # def sample_fullrank(self, scale, cov, fullrank):
    #     scale = scale ** 0.5

    #     mean_list = []
    #     sq_mean_list = []

    #     _cov_sqrt_list = []

    #     for (module, layer) in self.params:
    #         mean = module.__getattr__(f"{layer}_mean")
    #         sq_mean = module.__getattr__(f"{layer}_sq_mean")

    #         if cov:
    #         _cov_sqrt = module.__getattr__(f"{name}_cov_sqrt")
    #         _cov_sqrt_list.append_cov_sqrt.cpu())

    #         mean_list.append(mean.cpu())
    #         sq_mean_list.append(sq_mean.cpu())

    #     mean = flatten(mean_list)
    #     sq_mean = flatten(sq_mean_list)

    #     # draw diagonal variance sample
    #     var = torch.clamp(sq_mean - mean ** 2, self.var_clamp)
    #     var_sample = var.sqrt() * torch.randn_like(var, requires_grad=False)

    #     # if covariance draw low rank sample
    #     if cov:
    #     _cov_sqrt = torch.cat(sqrt_cov_list, dim=1)

    #         cov_sample =_cov_sqrt.t().matmul(
    #         _cov_sqrt.new_empty(
    #                 _cov_sqrt.size(0),), requires_grad=False
    #             ).normal_()
    #         )
    #         cov_sample /= (self.K - 1) ** 0.5

    #         rand_sample = var_sample + cov_sample
    #     else:
    #         rand_sample = var_sample

    #     # update sample with mean and scale
    #     sample = mean + scale_sqrt * rand_sample
    #     sample = sample.unsqueeze(0)

    #     # unflatten new sample like the mean sample
    #     samples_list = unflatten_like(sample, mean_list)

    #     for (module, layer), sample in zip(self.params, samples_list):
    #         module.__setattr__(layer, sample.cuda())

    def init_swag_weight(self, src_model):
        for layer, src_m in src_model.named_modules():
            if layer in self.swag_params:
                trt_m = self.swag_params[layer]
                for name in trt_m._parameters:
                    if trt_m._parameters[name] is None:
                        continue
                    src_data = src_m._parameters[name].data
                    trt_m.__setattr__(f"{name}_mean", src_data)
                    trt_m.__setattr__(f"{name}_sq_mean", src_data ** 2)
        self.initialized = True

    def collect_model(self, src_model):
        if not self.initialized:
            print("SWAG model is not initialized")
            raise RuntimeError
        else:
            for layer, src_m in src_model.named_modules():
                if layer in self.swag_params:
                    trt_m = self.swag_params[layer]
                    for name in trt_m._parameters:
                        if src_m._parameters[name] is None:
                            continue

                        src_data = src_m._parameters[name].data

                        mean = trt_m.__getattr__(f"{name}_mean")
                        sq_mean = trt_m.__getattr__(f"{name}_sq_mean")

                        # first moment
                        mean = mean * (self.n.item() / (self.n.item() + 1.0)) + src_data / (self.n.item() + 1.0)

                        # second moment
                        sq_mean = sq_mean * (self.n.item() / (self.n.item() + 1.0)) + src_data ** 2 / (self.n.item() + 1.0)

                        # square root of covariance matrix
                        cov_sqrt = trt_m.__getattr__(f"{name}_cov_sqrt")

                        # block covariance matrices, store deviation from current mean
                        dev = (src_data - mean).view(1, -1)
                        cov_sqrt = torch.cat([cov_sqrt, dev], dim=0)

                        # remove first row if we have stored too many models
                        if (self.n.item() + 1) > self.K:
                            cov_sqrt = cov_sqrt[1:, :]

                        trt_m.__setattr__(f"{name}_mean", mean)
                        trt_m.__setattr__(f"{name}_sq_mean", sq_mean)
                        trt_m.__setattr__(f"{name}_cov_sqrt", cov_sqrt)
        self.n.add_(1)


if __name__ == "__main__":
    from src.model.backbone.shufflenetv2_plus import shufflenetv2
    model = shufflenetv2()
    swag_model = SWAG(shufflenetv2(), K=10)
    swag_model.init_swag_weight(model)

    for _ in range(5):
        swag_model.collect_model(model)
    
    swag_model.sample_blockwise(0.5)
    
    print(swag_model.n)


    