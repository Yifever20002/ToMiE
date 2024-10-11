import torch 
import torch.nn as nn
import math

class ExtraPoseTuner(nn.Module):
    def __init__(self,
                 total_bones=24,
                 mlp_width=256,
                 mlp_depth=4,
                 **_):
        super(ExtraPoseTuner, self).__init__()
        # pose index embedding and time embedding
        embedding_size = p_dim + t_dim
        print('ExtraPoseTuner Embedding Input', embedding_size)
        block_mlps = [nn.Linear(embedding_size, mlp_width), nn.ReLU()]
        
        for _ in range(0, mlp_depth-1):
            block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]
        
        self.total_bones = total_bones - 1   # 24 for simple_SMPLx
        block_mlps += [nn.Linear(mlp_width, 3)]

        self.block_mlps = nn.Sequential(*block_mlps)
        initseq(self.block_mlps)

        # init the weights of the last layer as very small value
        # -- at the beginning, we hope the rotation matrix can be identity
        init_val = 1e-2
        last_layer = self.block_mlps[-1]
        last_layer.weight.data.uniform_(-init_val, init_val)
        last_layer.bias.data.zero_()

        self.rodriguez = RodriguesModule()

    def forward(self, t, p):

        t_feature = t_embedder(t)
        p_feature = p_embedder(p)
        input = torch.cat((t_feature, p_feature), dim=-1)
        rvec = self.block_mlps(input).view(-1, 3)
        Rs = self.rodriguez(rvec).view(-1, 3, 3)
        
        
        return {
            "Rs_extra": Rs,
            "rvec_extra": rvec,
        }

###############################################################################
## Init Functions
###############################################################################

def xaviermultiplier(m, gain):
    """ 
        Args:
            m (torch.nn.Module)
            gain (float)

        Returns:
            std (float): adjusted standard deviation
    """ 
    if isinstance(m, nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // m.stride[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] \
                // m.stride[0] // m.stride[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] \
                // m.stride[0] // m.stride[1] // m.stride[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * math.sqrt(2.0 / (n1 + n2))
    else:
        return None

    return std

def xavier_uniform_(m, gain):
    """ Set module weight values with a uniform distribution.

        Args:
            m (torch.nn.Module)
            gain (float)
    """ 
    std = xaviermultiplier(m, gain)
    m.weight.data.uniform_(-(std * math.sqrt(3.0)), std * math.sqrt(3.0))

def initmod(m, gain=1.0, weightinitfunc=xavier_uniform_):
    """ Initialized module weights.

        Args:
            m (torch.nn.Module)
            gain (float)
            weightinitfunc (function)
    """ 
    validclasses = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, 
                    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
    if any([isinstance(m, x) for x in validclasses]):
        weightinitfunc(m, gain)
        if hasattr(m, 'bias'):
            m.bias.data.zero_()

    # blockwise initialization for transposed convs
    if isinstance(m, nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    if isinstance(m, nn.ConvTranspose3d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 0::2, 1::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 0::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 1::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 0::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 1::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 0::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 1::2] = m.weight.data[:, :, 
                                                              0::2, 0::2, 0::2]

def initseq(s):
    """ Initialized weights of all modules in a module sequence.

        Args:
            s (torch.nn.Sequential)
    """ 
    for a, b in zip(s[:-1], s[1:]):
        if isinstance(b, nn.ReLU):
            initmod(a, nn.init.calculate_gain('relu'))
        elif isinstance(b, nn.LeakyReLU):
            initmod(a, nn.init.calculate_gain('leaky_relu', b.negative_slope))
        elif isinstance(b, nn.Sigmoid):
            initmod(a)
        elif isinstance(b, nn.Softplus):
            initmod(a)
        else:
            initmod(a)

    initmod(s[-1])

class RodriguesModule(nn.Module):
    def forward(self, rvec):
        r''' Apply Rodriguez formula on a batch of rotation vectors.

            Args:
                rvec: Tensor (B, 3)
            
            Returns
                rmtx: Tensor (B, 3, 3)
        '''
        theta = torch.sqrt(1e-5 + torch.sum(rvec ** 2, dim=1))
        rvec = rvec / theta[:, None]
        costh = torch.cos(theta)
        sinth = torch.sin(theta)
        return torch.stack((
            rvec[:, 0] ** 2 + (1. - rvec[:, 0] ** 2) * costh,
            rvec[:, 0] * rvec[:, 1] * (1. - costh) - rvec[:, 2] * sinth,
            rvec[:, 0] * rvec[:, 2] * (1. - costh) + rvec[:, 1] * sinth,

            rvec[:, 0] * rvec[:, 1] * (1. - costh) + rvec[:, 2] * sinth,
            rvec[:, 1] ** 2 + (1. - rvec[:, 1] ** 2) * costh,
            rvec[:, 1] * rvec[:, 2] * (1. - costh) - rvec[:, 0] * sinth,

            rvec[:, 0] * rvec[:, 2] * (1. - costh) - rvec[:, 1] * sinth,
            rvec[:, 1] * rvec[:, 2] * (1. - costh) + rvec[:, 0] * sinth,
            rvec[:, 2] ** 2 + (1. - rvec[:, 2] ** 2) * costh), 
        dim=1).view(-1, 3, 3)
    
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(
                    lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim

# input extra joint index
p_embedder, p_dim = get_embedder(5, input_dims=1)
# input time frame
t_embedder, t_dim = get_embedder(10, input_dims=1)