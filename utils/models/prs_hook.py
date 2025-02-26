""" 
Adapted from https://github.com/yossigandelsman/clip_text_span. MIT License Copyright (c) 2024 Yossi Gandelsman
"""
import numpy as np
import torch


class PRSLogger(object):
    """
    Class computing Residual Sum of components of a CLIP-like model to the final output.
    Can be run on cuda devices.
    l = number of layers, n = number of patches, n = number of patches value, h = number of heads, d = dimension of the attention
    """
    def __init__(self, model, device, spatial: bool = True, vision_projection: bool = True, full_output: bool = False):
        self.current_layer = 0
        self.device = device
        self.attentions = []
        self.mlps = []
        self.spatial = spatial # If we want to compute also the spatial contributions of each patch in the attention
        self.post_ln_std = None # If the CLIP model uses pre-projection Layer Norm
        self.post_ln_mean = None
        self.model = model
        self.vision_projection = vision_projection # If we want to apply the vision proejction
        self.full_output = full_output # If we want to keep the output and not looking only at the CLS tokens part

    @torch.no_grad()
    def compute_attentions_spatial(self, ret):
        assert len(ret.shape) == 5, "Verify that you use method=`head` and not method=`head_no_spatial`" # [b, n, m, h, d]
        assert self.spatial, "Verify that you use method=`head` and not method=`head_no_spatial`"
        orig_type = ret.dtype
        bias_term = self.model.visual.transformer.resblocks[
            self.current_layer
        ].attn.out_proj.bias
        self.current_layer += 1
        return_value = ret[:, 0].detach().cpu()  # This is only for the cls token
        self.attentions.append(
            (return_value.to(dtype=torch.float32)
            + bias_term[np.newaxis, np.newaxis, np.newaxis].cpu()
            / (return_value.shape[1] * return_value.shape[2])).to(dtype=orig_type)
        )  # [b, n, h, d]
        return ret

    @torch.no_grad()
    def compute_attentions_non_spatial_full(self, ret):
        assert len(ret.shape) == 4, "Verify that you use method=`head_no_spatial` and not method=`head`" # [b, n, h, d]
        assert not self.spatial, "Verify that you use method=`head_no_spatial` and not method=`head`"
        orig_type = ret.dtype
        bias_term = self.model.visual.transformer.resblocks[
            self.current_layer
        ].attn.out_proj.bias
        self.current_layer += 1
        return_value = ret.detach().cpu()  # Keep all the tokens
        self.attentions.append((
            return_value.to(dtype=torch.float32)
            + bias_term[np.newaxis, np.newaxis, np.newaxis].cpu()
            / (return_value.shape[2])).to(dtype=orig_type)
        )  # [b, n, h, d]
        return ret

    @torch.no_grad()
    def compute_attentions_non_spatial(self, ret):
        assert len(ret.shape) == 4, "Verify that you use method=`head_no_spatial` and not method=`head`" # [b, n, h, d]
        assert not self.spatial, "Verify that you use method=`head_no_spatial` and not method=`head`"
        orig_type = ret.dtype
        bias_term = self.model.visual.transformer.resblocks[
            self.current_layer
        ].attn.out_proj.bias
        self.current_layer += 1
        return_value = ret[:, 0].detach().cpu()  # This is only for the cls token
        self.attentions.append(
            (return_value.to(dtype=torch.float32)
            + bias_term[np.newaxis, np.newaxis].cpu()
            / (return_value.shape[1])).to(dtype=orig_type)
        )  # [b, h, d]
        return ret

    @torch.no_grad()
    def compute_mlps_full(self, ret):
        self.mlps.append(ret.detach().cpu())  # [b, n, d]
        return ret
    
    @torch.no_grad()
    def compute_mlps(self, ret):
        self.mlps.append(ret[:, 0].detach().cpu())  # [b, d]
        return ret

    @torch.no_grad()
    def log_post_ln_mean(self, ret):
        self.post_ln_mean = ret.detach().cpu()  # [b, 1]
        return ret

    @torch.no_grad()
    def log_post_ln_std(self, ret):
        self.post_ln_std = ret.detach().cpu()  # [b, 1]
        return ret

    def _normalize_mlps(self):
        orig_dtype = self.mlps.dtype
        # b, l + 1, d
        len_intermediates = self.attentions.shape[1] + self.mlps.shape[1]
        # This is just the normalization layer:
        mean_centered = (
            self.mlps.to(torch.float32)
            - self.post_ln_mean[:, :, np.newaxis].to(self.device, torch.float32) / len_intermediates
        )
        weighted_mean_centered = (
            self.model.visual.ln_post.weight.detach().to(self.device, torch.float32) * mean_centered
        )
        weighted_mean_by_std = weighted_mean_centered / self.post_ln_std[
            :, :, np.newaxis
        ].to(self.device, torch.float32)
        bias_term = (
            self.model.visual.ln_post.bias.detach().to(self.device, torch.float32) / len_intermediates
        )
        post_ln = weighted_mean_by_std + bias_term
        if self.vision_projection:
            return (post_ln @ self.model.visual.proj.detach().to(self.device, torch.float32)).to(orig_dtype)
        else:
            return post_ln.to(orig_dtype)

    def _normalize_mlps_full(self):
        orig_dtype = self.mlps.dtype
        # b, l + 1, n, d
        len_intermediates = self.attentions.shape[1] + self.mlps.shape[1]
        # This is just the normalization layer:
        mean_centered = (
            self.mlps.to(torch.float32)
            - self.post_ln_mean[:, :, np.newaxis, np.newaxis].to(self.device, torch.float32) / len_intermediates
        )
        weighted_mean_centered = (
            self.model.visual.ln_post.weight.detach().to(self.device, torch.float32) * mean_centered
        )
        weighted_mean_by_std = weighted_mean_centered / self.post_ln_std[
            :, :, np.newaxis, np.newaxis
        ].to(self.device, torch.float32)
        bias_term = (
            self.model.visual.ln_post.bias.detach().to(self.device, torch.float32) / len_intermediates
        )
        post_ln = weighted_mean_by_std + bias_term
        
        if self.vision_projection:
            return (post_ln @ self.model.visual.proj.detach().to(self.device, torch.float32)).to(orig_dtype)
        else:
            return post_ln.to(orig_dtype)
    
    def _normalize_attentions_spatial(self):
        orig_dtype = self.attentions.dtype
        # [b, l, m, h, d]
        len_intermediates = self.attentions.shape[1] + self.mlps.shape[1]  # 2*l + 1
        normalization_term = (
            self.attentions.shape[2] * self.attentions.shape[3]
        )  # n * h
        # This is just the normalization layer:
        mean_centered = self.attentions.to(torch.float32) - self.post_ln_mean[
            :, :, np.newaxis, np.newaxis, np.newaxis
        ].to(self.device, torch.float32) / (len_intermediates * normalization_term)
        weighted_mean_centered = (
            self.model.visual.ln_post.weight.detach().to(self.device, torch.float32) * mean_centered
        )
        weighted_mean_by_std = weighted_mean_centered / self.post_ln_std[
            :, :, np.newaxis, np.newaxis, np.newaxis
        ].to(self.device, torch.float32)
        bias_term = self.model.visual.ln_post.bias.detach().to(self.device, torch.float32) / (
            len_intermediates * normalization_term
        )
        post_ln = weighted_mean_by_std + bias_term
        
        if self.vision_projection:
            return (post_ln @ self.model.visual.proj.detach().to(self.device, torch.float32)).to(orig_dtype)
        else:
            return post_ln.to(orig_dtype)

    def _normalize_attentions_non_spatial_full(self):
        orig_dtype = self.attentions.dtype
        # [b, l, n, h, d], need to normalize on last 
        len_intermediates = self.attentions.shape[1] + self.mlps.shape[1]  # 2*l + 1
        normalization_term = (
            self.attentions.shape[3]
        )  # h
        # This is just the normalization layer:
        mean_centered = self.attentions.to(torch.float32) - self.post_ln_mean[
            :, :, np.newaxis, np.newaxis, np.newaxis
        ].to(self.device, torch.float32) / (len_intermediates * normalization_term)
        weighted_mean_centered = (
            self.model.visual.ln_post.weight.detach().to(self.device, torch.float32) * mean_centered
        )
        weighted_mean_by_std = weighted_mean_centered / self.post_ln_std[
            :, :, np.newaxis, np.newaxis, np.newaxis
        ].to(self.device, torch.float32)
        bias_term = self.model.visual.ln_post.bias.detach().to(self.device, torch.float32) / (
            len_intermediates * normalization_term
        )
        post_ln = weighted_mean_by_std + bias_term

        if self.vision_projection:
            return (post_ln @ self.model.visual.proj.detach().to(self.device, torch.float32)).to(orig_dtype)
        else:
            return post_ln.to(orig_dtype)

    def _normalize_attentions_non_spatial(self):
        orig_dtype = self.attentions.dtype
        # b, l, h, d
        len_intermediates = self.attentions.shape[1] + self.mlps.shape[1]  # 2*l + 1
        normalization_term = (
            self.attentions.shape[2]
        )  # h
        # This is just the normalization layer:
        mean_centered = self.attentions.to(torch.float32) - self.post_ln_mean[
            :, :, np.newaxis, np.newaxis
        ].to(self.device, torch.float32) / (len_intermediates * normalization_term)
        weighted_mean_centered = (
            self.model.visual.ln_post.weight.detach().to(self.device, torch.float32) * mean_centered
        )
        weighted_mean_by_std = weighted_mean_centered / self.post_ln_std[
            :, :, np.newaxis, np.newaxis
        ].to(self.device, torch.float32)
        bias_term = self.model.visual.ln_post.bias.detach().to(self.device, torch.float32) / (
            len_intermediates * normalization_term
        )
        post_ln = weighted_mean_by_std + bias_term
        if self.vision_projection:
            return (post_ln @ self.model.visual.proj.detach().to(self.device, torch.float32)).to(orig_dtype)
        else:
            return post_ln.to(orig_dtype)

    @torch.no_grad()
    def finalize(self, representation):
        """We calculate the post-ln scaling, project it and normalize by the last norm."""
        self.attentions = torch.stack(self.attentions, axis=1).to(
            self.device
        )  # [b, l, ..., n, h, d]
        self.mlps = torch.stack(self.mlps, axis=1).to(self.device)  # [b, l + 1, ..., d]
        if self.full_output:
            projected_attentions = self.attentions
            projected_mlps = self.mlps
        else:
            if self.spatial:
                projected_attentions = self._normalize_attentions_spatial()
                projected_mlps = self._normalize_mlps()
            else:
                projected_attentions = self._normalize_attentions_non_spatial()
                projected_mlps = self._normalize_mlps()

        norm = representation.norm(dim=-1).detach()

        if self.vision_projection:
            return (
                projected_attentions
                / norm[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis],
                projected_mlps / norm[:, np.newaxis, np.newaxis, np.newaxis],
            )
        else:
            return (
                projected_attentions,
                projected_mlps
            )
        
    def reinit(self):
        self.current_layer = 0
        self.attentions = []
        self.mlps = []
        self.post_ln_mean = None
        self.post_ln_std = None
        torch.cuda.empty_cache()


def hook_prs_logger(model, device, spatial: bool = True, vision_projection: bool = True, full_output: bool = False):
    """Hooks a projected residual stream logger to the model."""
    prs = PRSLogger(model, device, spatial=spatial, vision_projection=vision_projection, full_output=full_output)

    if full_output:
        model.hook_manager.register(
            "visual.transformer.resblocks.*.attn.out.post", prs.compute_attentions_non_spatial_full
        )
        model.hook_manager.register(
            "visual.transformer.resblocks.*.mlp.c_proj.post", prs.compute_mlps_full
        )
        model.hook_manager.register("visual.ln_pre_post", prs.compute_mlps_full)

    else: 
        if spatial:
            model.hook_manager.register(
                "visual.transformer.resblocks.*.attn.out.post", prs.compute_attentions_spatial
            )
        else:
            model.hook_manager.register(
                "visual.transformer.resblocks.*.attn.out.post", prs.compute_attentions_non_spatial
            )

        model.hook_manager.register(
            "visual.transformer.resblocks.*.mlp.c_proj.post", prs.compute_mlps
        )
        model.hook_manager.register("visual.ln_pre_post", prs.compute_mlps)
        
    model.hook_manager.register("visual.ln_post.mean", prs.log_post_ln_mean)
    model.hook_manager.register("visual.ln_post.sqrt_var", prs.log_post_ln_std)
    return prs
