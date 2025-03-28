import torch 
import torch.nn as nn

# --- Define the Inversion Network ---
# A small two-layer network with GELU activation that maps from 768-dim to 1024-dim.
class InversionNet(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=1024, output_dim=1024):
        super(InversionNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


@torch.no_grad()
def retrieve_proj_matrix(model_CLIP):
    """
    Retrieves the projection matrix from the visual encoder of a CLIP model.
    
    Parameters:
    model_CLIP (CLIP model): The pre-trained CLIP model from which the visual projection 
                             matrix is extracted.
    
    Returns:
    torch.Tensor: The projection matrix used in the visual encoder.
    """
    return model_CLIP.visual.proj


@torch.no_grad()
def retrieve_post_layer_norm_par(model_CLIP):
    """
    Retrieves the parameters (weight and bias) of the post-layer normalization in 
    the visual encoder of a CLIP model.
    
    Parameters:
    model_CLIP (CLIP model): The pre-trained CLIP model from which the normalization 
                             parameters are extracted.
    
    Returns:
    tuple(torch.Tensor, torch.Tensor, torch.Tensor): A tuple containing the weight, bias and eps of the 
                                       post-layer normalization layer.
    """
    return model_CLIP.visual.ln_post.weight, model_CLIP.visual.ln_post.bias, model_CLIP.visual.ln_post.eps


@torch.no_grad()
def invert_proj_layer_norm(clip_emb, P, ln_weight, ln_bias, std, mean, eps):
    """
    Inverts the projection and layer normalization applied to CLIP embeddings.
    
    Parameters:
    clip_emb (torch.Tensor): The input CLIP embedding.
    P (torch.Tensor): The projection matrix from the visual encoder.
    ln_weight (torch.Tensor): The weight parameter of the post-layer normalization.
    ln_bias (torch.Tensor): The bias parameter of the post-layer normalization.
    std (torch.Tensor): The mean ablated standard deviation of the original feature distribution.
    mean (torch.Tensor): The mean ablated mean of the original feature distribution.
    eps (float): A small constant to prevent division by zero in normalization computations.

    Returns:
    torch.Tensor: The inverted feature representation before projection and layer normalization.
    """

    inv_proj =  (clip_emb @ torch.linalg.pinv(P.to(dtype=torch.float32)).to(clip_emb.dtype))

    # Handle op in float32
    return ((inv_proj.to(dtype=torch.float32) - ln_bias.to(dtype=torch.float32)) / \
         (ln_weight.to(dtype=torch.float32)/torch.sqrt(std.to(dtype=torch.float32)**2 + eps)) + \
        mean.to(dtype=torch.float32)*torch.ones_like(ln_weight.to(dtype=torch.float32))).to(clip_emb.dtype)
