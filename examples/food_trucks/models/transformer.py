import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

'''
Step 1
------

Segment image into N x N patches

Rearrange

from: batch_size channels (height patch1) (width patch2) 
to: batch_size (height width) (patch1 patch2 channels)

Each patch is passed through a fully connected linear layer
'''

class PatchEmb(nn.Module):
    '''
    Patch embedding for 5-dimensional state inputs
    (batch, timestep, channel, height, width).

                      Parameters                  
    -----------------------------------------------
    config: A dictionary with the following vars...
    
    patch_size (tuple): Patch size in HxW
    img_size (tuple): Image size in CxHxW
    vocab_size (int): Action

    '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        D, iD = config.D, config.local_D
        p1, p2 = config.patch_size
        c, h, w = config.img_size

        self.patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b t) (h w) (p1 p2 c)', p1 = p1, p2 = p2),
            nn.Linear(p1*p2*c, iD)
        )

        # +1 for mask
        self.action_emb = nn.Embedding(config.vocab_size+1, iD)
    
    
    def forward(self, states, actions, rewards=None):
#         print (states.size(), actions.size())
        B, T, C, H, W = states.size()
        local_state_tokens = (self.patch_embedding(states) + self.spatial_emb).reshape(B, T, -1, self.config.local_D)
        local_action_tokens = self.action_emb(actions.reshape(-1, 1)).reshape(B, T, -1).unsqueeze(2) # B T 1 iD

        print (local_action_tokens.size(), local_state_tokens.size())

        local_tokens = torch.cat((local_action_tokens, local_state_tokens), dim=2)
        
        return local_tokens, global_state_tokens, self.temporal_emb[:, :T]