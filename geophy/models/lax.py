import numpy as np
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, latent_dim=1024, last_bias=True):
        super().__init__()
        #input_dim = np.prod(tip_state_size)
        latent_dims = latent_dim if isinstance(latent_dim, list) else [latent_dim]
        net = [ nn.Linear(input_dim, latent_dims[0]) ]
        for i in range(len(latent_dims) - 1):
            net.append(nn.SiLU())
            net.append(nn.Linear(latent_dims[i], latent_dims[i+1]))
        net.append(nn.SiLU())
        net.append(nn.Linear(latent_dims[0], 1, bias=last_bias))
        self._net = nn.Sequential(*net)

    def forward(self, x):
        return self._net(x)

class LAXModel(nn.Module):
    def __init__(self, n_tips, embed_space, embed_dim, network='mlp', scale_factor=5, last_bias=True):
        super().__init__()
        self.n_tips = n_tips
        self.embed_space = embed_space
        self.embed_dim = embed_dim
        self.network = network
        if network == 'mlp':
            input_dim = n_tips * embed_dim
            latent_dim = n_tips * embed_dim * scale_factor
            self._net = MLP(input_dim=input_dim, latent_dim=latent_dim, last_bias=last_bias)
        else:
            raise NotImplementedError
        #self._dtype = next(self.parameters()).dtype

    def _transform(self, tip_embeds):
        tip_states = torch.stack([embed.vecs for embed in tip_embeds])  # (n_batch, n_tips, embed_dim)
        n_batch = tip_states.shape[0]
        state = tip_states.reshape((n_batch, -1)) # (., n_tips * embed_dim)
        return state

    def forward(self, tip_embeds): #, inv_temp=1.):
        state = self._transform(tip_embeds)
        return self._net(state)
