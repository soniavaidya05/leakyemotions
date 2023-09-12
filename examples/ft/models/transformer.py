# --------------- #
# region: Imports #
# --------------- #

import torch
import torch.nn as nn

from typing import Union
from numpy.typing import ArrayLike
from einops.layers.torch import Rearrange

from examples.ft.models.buffer import ActionBatchReplayBuffer as ABRBuffer

# --------------- #
# endregion       #
# --------------- #

# -------------------------- #
# region: Layers and modules #
# -------------------------- #

class PatchEmbedding(nn.Module):
    """
    Patch embedding module.
    """
    def __init__(
        self,
        state_size: ArrayLike,
        patch_size: int,
        layer_size: int,
    ):
        super().__init__()
        self.layer_size = layer_size

        # Unpack state size into Channels, height, width
        c, h, w = tuple(state_size)
        p = patch_size
        
        # break down the image into (patch_size x patch_size) patches, then flatten
        self.projection = nn.Sequential(
            # Rearrange batch and timesteps... we will fix this later
            Rearrange('b t c (h p1) (w p2) -> (b t) (h w) (p1 p2 c)', p1=p, p2=p),
            nn.Linear(
                in_features=p*p*c,
                out_features=layer_size
            )
        )

        # Position embedding
        self.position_embedding = nn.Parameter(
            torch.randn(1, h*w//p//p, layer_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Return data in format (B x T), (H x W), (P * P * C)
        return self.projection(x) + self.position_embedding
    
class ActionEmbedding(nn.Module):
    """
    Action embedding module.
    """
    def __init__(
        self,
        action_space: int,
        layer_size: int,
    ):
        
        self.action_embedding = nn.Embedding(action_space, layer_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Reshape to go into the embedding layer
        x = x.view(-1, 1)

        # Pass through the embedding layer
        return self.action_embedding(x)

class JointEmbedding(nn.Module):
    """
    Embed patches and actions simultaneously
    """
    def __init__(
        self,
        state_size: ArrayLike,
        patch_size: int,
        action_space: int,
        layer_size: int
    ):

        self.layer_size = layer_size
        self.patch_size = patch_size
        self.patch_embedding = PatchEmbedding(state_size, patch_size, layer_size)
        self.action_embedding = ActionEmbedding(action_space, layer_size)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:

        # Get the batch size and timesteps
        B, T, _, _, _ = states.size()

        # Get the embeddings and reshape them with batch size and timesteps restored
        state_embeddings = self.patch_embedding(states).view(B, T, -1, self.layer_size)
        action_embeddings = self.action_embedding(actions).view(B, T, -1, self.layer_size)

        # Concatenate the embeddings
        local_tokens = torch.cat((state_embeddings, action_embeddings), dim = 2)

        return local_tokens
    
class Attention(nn.Module):
    """
    Multi-head self-attention module.
    """

    def __init__(
        self,
        layer_size: int,
        num_heads: int,
        dropout: float = 0.
    ):
        assert layer_size % num_heads == 0, "Layer size must be evenly divisible by the number of attention heads."


        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(
            embed_dim=layer_size,
            num_heads=num_heads,
            dropout=dropout
        )

        # Query, key, and value layers
        self.q = nn.Linear(in_features=layer_size, out_features=layer_size)
        self.k = nn.Linear(in_features=layer_size, out_features=layer_size)
        self.v = nn.Linear(in_features=layer_size, out_features=layer_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:

        # Get Q, K, V
        q, k, v = self.q(x), self.k(x), self.v(x)
        # Pass them into the attention module (not keeping weights for now)
        output, _ = self.attention(q, k, v)

        return output
 
class TransformerBlock(nn.Module):
    """
    Single transformer block.
    """
    def __init__(
        self,
        layer_size: int,
        dropout: float = 0.,
        **kwargs
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(layer_size)
        self.norm2 = nn.LayerNorm(layer_size)
        self.attention = Attention(layer_size=layer_size, dropout=dropout,**kwargs)
        self.ff = nn.Sequential(
                nn.Linear(layer_size, layer_size * 4),
                nn.GELU(),
                nn.Linear(layer_size * 4, layer_size),
                nn.Dropout(dropout)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Attention layer: layernormalize + skip connection
        y = x + self.attention(self.norm1(x))
        # Feedforward layer: Layernormalize + skip_connection
        y = y + self.ff(self.norm1(x))

        return y



    

    
class Transformer(nn.Module):
    """
    Full multilayer transformer.
    """
    def __init__(
        self,
        depth,
        **kwargs

    ):
        
        self.norm = nn.LayerNorm(kwargs['layer_size'])
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                TransformerBlock
            )
            


# -------------------------- #
# endregion:                 #
# -------------------------- #

class VisionTransformer(nn.Module):
    
    def __init__(
        self,
        state_size: ArrayLike,
        action_size: int,
        layer_size: int,
        patch_size: int,
        num_frames: int,
        batch_size: int,
        memory: ABRBuffer,
        LR: float,
        device: Union[str, torch.device],
        seed: int
    ):
        """Vision transformer adapted from Phil Wang's ViT (https://github.com/lucidrains/vit-pytorch/tree/main). 
        Takes a sequence of visual inputs plus prior actions and returns an action prediction.

        Parameters:
            state_size: (ArrayLike) An array-like sequence of the form 
            C x H x W defining the input image size. \n
            action_size: (int) The number of possible actions PLUS ONE (used for the masked actions). \n
            layer_size: (int) The size of the embedding layer. \n
            patch_size: (int) The size of patches to use in the model. \n
            num_frames: (int) The number of timesteps passed into the model. \n
            batch_size: (int) The size of the training batches. \n
            memory: (ActionBatchReplayBuffer) A model object with stored memories. \n
            LR: (float) The learning rate of the model. \n
            device: (str, torch.device) The device to perform computations on. \n
            seed: (int) Manual seed for replication purposes.
        """
        
        # Image and patch dimensions
        self.state_size = state_size
        self.patch_size = patch_size
        self.num_patches = self.patch()

        # Additional input dimensions
        self.batch_size = batch_size
        self.num_frames = num_frames # timesteps

        # Layer sizes
        self.layer_size = layer_size
        self.action_size = action_size # Output dimensions

        # Additional elements
        self.memory = memory
        self.device = device
        self.seed = seed

        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            in_channels=state_size[0],
            patch_size=patch_size,
            layer_size=layer_size
        )






    def patch(self) -> int:
        """
        Compute the number of patches based on the input size.
        """
        num_patches = 1
        for i in [1, 2]:
            assert self.state_size[i] % self.patch_size == 0, f"Image dimensions {self.state_size[1]} x {self.state_size[2]} must be evenly divisible by the patch size {self.patch_size} x {self.patch_size}."
            num_patches *= self.state_size // self.patch_size
        return num_patches
    

vit = VisionTransformer()