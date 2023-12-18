import torch
from conformer import ConformerBlock

class ConformerEncoder(torch.nn.Module):
    def __init__(self,input_dim, embed_dim, dim_head, heads, kernel_size):
        super(ConformerEncoder, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.dim_head = dim_head
        self.heads = heads
        self.kernel_size = kernel_size
        self.input_layer = torch.nn.Linear(self.input_dim, 64)

        self.conformer_blocks_64 = ConformerBlock(dim=64, dim_head=self.dim_head, heads=self.heads, conv_kernel_size=self.kernel_size,ff_mult=4)
        self.linear_layer_64_32 = torch.nn.Linear(64, 32)
        self.conformer_blocks_32 = ConformerBlock(dim=32, dim_head=self.dim_head, heads=self.heads, conv_kernel_size=self.kernel_size,ff_mult=4)
        self.linear_layer_32_16 = torch.nn.Linear(32, 16)
        self.conformer_blocks_16 = ConformerBlock(dim=16, dim_head=self.dim_head, heads=self.heads, conv_kernel_size=self.kernel_size,ff_mult=4)
        self.embed_layer = torch.nn.Linear(16, self.embed_dim)

    def forward(self, x):
        x=self.input_layer(x)
        x=self.conformer_blocks_64(x)
        x=self.linear_layer_64_32(x)
        x=self.conformer_blocks_32(x)
        x=self.linear_layer_32_16(x)
        x=self.conformer_blocks_16(x)
        x=self.embed_layer(x)
        return x

class ConformerDecoder(torch.nn.Module):
    def __init__(self, input_dim, embed_dim, dim_head, heads, kernel_size):
        super(ConformerDecoder, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.dim_head = dim_head
        self.heads = heads
        self.kernel_size = kernel_size

        self.embed_layer = torch.nn.Linear(self.embed_dim, 16)
        self.conformer_blocks_16 = ConformerBlock(dim=16, dim_head=self.dim_head, heads=self.heads, conv_kernel_size=self.kernel_size,ff_mult=4)
        self.embed_layer = torch.nn.Linear(16, 32)
        self.conformer_blocks_32 = ConformerBlock(dim=32, dim_head=self.dim_head, heads=self.heads, conv_kernel_size=self.kernel_size,ff_mult=4)
        self.linear_layer_64_32 = torch.nn.Linear(32,64)
        self.conformer_blocks_64 = ConformerBlock(dim=64, dim_head=self.dim_head, heads=self.heads, conv_kernel_size=self.kernel_size,ff_mult=4)
        self.output_layer = torch.nn.Linear(64, self.input_dim)

    def forward(self, x):
        x=self.embed_layer(x)
        x=self.conformer_blocks_16(x)
        x=self.linear_layer_16_32(x)
        x=self.conformer_blocks_32(x)
        x=self.linear_layer_64_32(x)
        x=self.conformer_blocks_64(x)
        x=self.output_layer(x)
        return x

class ConformerAutoEncoder(torch.nn.Module):
    def __init__(self, input_dim,embed_dim, dim_head, heads, kernel_size):
        super(ConformerAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.dim_head = dim_head
        self.heads = heads
        self.kernel_size = kernel_size  

        self.encoder = ConformerEncoder(self.input_dim, self.embed_dim, self.dim_head, self.heads, self.kernel_size)
        self.decoder = ConformerDecoder(self.input_dim, self.embed_dim, self.dim_head, self.heads, self.kernel_size)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class ConformerAutoEncoder_(torch.nn.Module):
    def __init__(self, input_dim,output_dim,embed_dim, dim_head, heads, kernel_size):
        super(ConformerAutoEncoder_, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.dim_head = dim_head
        self.heads = heads
        self.kernel_size = kernel_size  

        self.encoder = ConformerEncoder(self.input_dim, self.embed_dim, self.dim_head, self.heads, self.kernel_size)
        self.decoder = ConformerDecoder(self.output_dim, self.embed_dim, self.dim_head, self.heads, self.kernel_size)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

'''# Noam optimizer
class NoamOptimizer(torch.optim.Adam):
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self, closure=None):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups():
            p['lr'] = rate
        self._rate = rate
        return self.optimizer.step(closure)

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) *
                              min(step ** (-0.5), step * self.warmup ** (-1.5)))'''