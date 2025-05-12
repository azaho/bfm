import torch
import torch.nn as nn

class BFModule(nn.Module):
    """
    This module is a base class for all modules that need to be compatible with this project.
    It ensures that the module stores its current device and dtype.
    """
    def __init__(self):
        super().__init__()
        self._device = None
        self._dtype = None
    def to(self, *args, **kwargs):
        output = super().to(*args, **kwargs)
        # Extract device and dtype from args/kwargs
        device = next((torch.device(arg) for arg in args if isinstance(arg, (torch.device, str))), 
                     kwargs.get('device', None))
        dtype = next((arg for arg in args if isinstance(arg, torch.dtype)),
                    kwargs.get('dtype', None))
        if device is not None: self._device = device 
        if dtype is not None: self._dtype = dtype
        return output
    @property
    def device(self):
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device
    @property 
    def dtype(self):
        if self._dtype is None:
            self._dtype = next(self.parameters()).dtype
        return self._dtype

class BFModel(BFModule):
    """Base model class for brain-feature models.
    
    The model accepts batches of shape (batch_size, n_electrodes, n_timebins, d_model)
    and electrode embedding of shape (n_electrodes, d_model).

    The model's forward pass must return a tuple where:
    - First element is a batch of shape (batch_size, n_timebins, d_output) containing the model's output
    - Remaining elements can be anything

    The model must implement calculate_pretrain_loss(self, x, y) for pretraining.
    """
    def __init__(self):
        super().__init__()

    def forward(self, electrode_embedded_data):
        pass

    def calculate_pretrain_loss(self, electrode_embedded_data):
        pass

    def generate_frozen_evaluation_features(self, electrode_embedded_data, feature_aggregation_method='concat'):
        pass


class LinearBinTransformer(BFModule):
    def __init__(self, overall_sampling_rate=2048, sample_timebin_size=0.125, identity_init=True):
        super(LinearBinTransformer, self).__init__()
        self.overall_sampling_rate = overall_sampling_rate
        self.sample_timebin_size = sample_timebin_size
        self.linear = nn.Linear(int(overall_sampling_rate*sample_timebin_size), int(overall_sampling_rate*sample_timebin_size), bias=False)

        if identity_init:
            self.linear.weight.data = torch.eye(int(overall_sampling_rate*sample_timebin_size)).to(self.device, dtype=self.dtype)

    def forward(self, electrode_data):
        # (batch_size, n_electrodes, n_samples)
        batch_size, n_electrodes, n_samples = electrode_data.shape
        electrode_data = electrode_data.reshape(batch_size, n_electrodes, -1, int(self.overall_sampling_rate*self.sample_timebin_size))

        batch_size, n_electrodes, n_timebins, sample_timebin_size = electrode_data.shape
        electrode_data = self.linear(electrode_data)

        return electrode_data

from model_transformers import Transformer
class BinTransformer(BFModule):
    def __init__(self, first_kernel=16, d_model=192, n_layers=5, n_heads=6, overall_sampling_rate=2048, sample_timebin_size=0.125, n_downsample_factor=1, identity_init=True):
        super(BinTransformer, self).__init__()
        self.transformer = Transformer(d_input=first_kernel//n_downsample_factor, d_model=d_model, d_output=first_kernel//n_downsample_factor, 
                                            n_layer=n_layers, n_head=n_heads, causal=True, 
                                            rope=True, cls_token=False, rope_base=int(overall_sampling_rate*sample_timebin_size)*n_downsample_factor//first_kernel, identity_init=identity_init)
        self.first_kernel = first_kernel
        self.n_downsample_factor = n_downsample_factor
        self.sample_timebin_size = sample_timebin_size
        self.overall_sampling_rate = overall_sampling_rate
    
    def forward(self, electrode_data):
        # (batch_size, n_electrodes, n_samples)
        batch_size, n_electrodes, n_samples = electrode_data.shape
        electrode_data = electrode_data.reshape(batch_size, n_electrodes, -1, int(self.overall_sampling_rate*self.sample_timebin_size))

        batch_size, n_electrodes, n_timebins, sample_timebin_size = electrode_data.shape
        electrode_data = electrode_data.reshape(batch_size, n_electrodes, n_timebins, -1, self.n_downsample_factor, self.first_kernel//self.n_downsample_factor)
        electrode_data = electrode_data.reshape(batch_size, -1, self.first_kernel//self.n_downsample_factor)
        electrode_data = self.transformer(electrode_data) 
        electrode_data = electrode_data.reshape(batch_size, n_electrodes, n_timebins, -1, self.n_downsample_factor, self.first_kernel//self.n_downsample_factor).mean(axis=-2)
        electrode_data = electrode_data.reshape(batch_size, n_electrodes, n_timebins, sample_timebin_size//self.n_downsample_factor)
        
        return electrode_data
    

class FFTaker(BFModule):
    def __init__(self, d_model=192, overall_sampling_rate=2048, sample_timebin_size=0.125, max_frequency_bin=64, power=True):
        super(FFTaker, self).__init__()
        self.sample_timebin_size = sample_timebin_size
        self.overall_sampling_rate = overall_sampling_rate
        self.max_frequency_bin = max_frequency_bin
        self.power = power
        self.d_model = d_model
        # Transform FFT output to match expected output dimension
        self.output_transform = nn.Linear(max_frequency_bin if power else 2*max_frequency_bin, 
                                         int(overall_sampling_rate*sample_timebin_size))
    
    def forward(self, electrode_data):
        # electrode_data is of shape (batch_size, n_electrodes, n_samples)
        batch_size, n_electrodes, n_samples = electrode_data.shape
        electrode_data = electrode_data.reshape(batch_size, n_electrodes, -1, int(self.overall_sampling_rate*self.sample_timebin_size))

        batch_size, n_electrodes, n_timebins, samples_per_bin = electrode_data.shape
        
        # Calculate FFT for each timebin
        x = electrode_data.reshape(-1, samples_per_bin)
        x = x.to(dtype=torch.float32)  # Convert to float32 for FFT
        x = torch.fft.rfft(x, dim=-1)  # Using rfft for real-valued input

        # Pad or trim to max_frequency_bin dimension
        if x.shape[1] < self.max_frequency_bin:
            x = torch.nn.functional.pad(x, (0, self.max_frequency_bin - x.shape[1]))
        else:
            x = x[:, :self.max_frequency_bin]

        x = x.reshape(batch_size, n_electrodes, n_timebins, -1)  # shape: (batch_size, n_electrodes, n_timebins, max_frequency_bin)
        
        if self.power:
            # Calculate magnitude (equivalent to scipy.signal.stft's magnitude)
            x = torch.abs(x)
            # Convert to power
            x = torch.log(x + 1e-5)
        else:
            x = torch.cat([torch.real(x), torch.imag(x)], dim=-1)  # shape: (batch_size, n_electrodes, n_timebins, 2*max_frequency_bin)

        # Batchnorm after taking FFT
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / (x.std(dim=-1, keepdim=True) + 1e-5)
        
        # Transform to match expected output dimension
        x = self.output_transform(x)  # shape: (batch_size, n_electrodes, n_timebins, first_kernel//n_downsample_factor)
        
        return x.to(dtype=electrode_data.dtype)  # Convert back to original dtype



class TransformerModel(BFModel):
    def __init__(self, d_model, d_output=None, n_layers_electrode=5, n_layers_time=5, n_heads=12, use_cls_token=True):
        super().__init__()
        self.d_model = d_model
        self.use_cls_token = use_cls_token
        
        if d_output is None:
            d_output = d_model

        self.electrode_transformer = Transformer(d_input=d_model, d_model=d_model, d_output=d_model, 
                                                 n_layer=n_layers_electrode, n_head=n_heads, causal=False, 
                                                 rope=False, cls_token=self.use_cls_token)
        self.time_transformer = Transformer(d_input=d_model, d_model=d_model, d_output=d_output, 
                                            n_layer=n_layers_time, n_head=n_heads, causal=True, 
                                            rope=True, cls_token=False)
        self.temperature_param = nn.Parameter(torch.tensor(1.0))

    def forward(self, embedded_electrode_data, only_electrode_output=False):
        # electrode_embeddings is of shape (n_electrodes, d_model)
        # embedded_electrode_data is of shape (batch_size, n_electrodes, n_timebins, d_model)

        electrode_data = embedded_electrode_data.permute(0, 2, 1, 3) # shape: (batch_size, n_timebins, n_electrodes, d_model)
        batch_size, n_timebins, n_electrodes, d_model = electrode_data.shape

        electrode_output = self.electrode_transformer(electrode_data.reshape(batch_size * n_timebins, n_electrodes, d_model)) # shape: (batch_size*n_timebins, n_electrodes+1, d_model)
        if self.use_cls_token:
            electrode_output = electrode_output[:, -1:, :].view(batch_size, n_timebins, d_model) # just the CLS token. Shape: (batch_size, n_timebins, d_model)
        else:
            electrode_output = electrode_output[:, 0:1, :].view(batch_size, n_timebins, d_model) # (just the first token, which can be anything)
        if only_electrode_output:
            return electrode_output, None
        
        time_output = self.time_transformer(electrode_output) # shape: (batch_size, n_timebins, d_model)
        return electrode_output, time_output
    
    def generate_frozen_evaluation_features(self, electrode_embedded_data, feature_aggregation_method='concat'):
        batch_size, n_electrodes, n_timebins, d_model = electrode_embedded_data.shape
        if feature_aggregation_method == 'concat':
            return self(electrode_embedded_data, only_electrode_output=True)[0].reshape(batch_size, -1)
        elif feature_aggregation_method == 'mean':
            return self(electrode_embedded_data, only_electrode_output=True)[0].mean(dim=1)
        else:
            raise ValueError(f"Invalid feature aggregation method: {feature_aggregation_method}")


from model_transformers import Transformer, CrossAttentionTransformer
class GranularModel(BFModel):
    def __init__(self, sample_timebin_size, d_model, n_layers=5, n_heads=12, identity_init=True, n_cls_tokens=0):
        super().__init__()
        self.d_model = d_model
        self.sample_timebin_size = sample_timebin_size
        self.n_cls_tokens = n_cls_tokens

        self.transformer = Transformer(d_input=self.sample_timebin_size, d_model=d_model, d_output=self.sample_timebin_size, 
                                            n_layer=n_layers, n_head=n_heads, causal=True, 
                                            rope=True, cls_token=False, identity_init=identity_init)
        self.temperature_param = nn.Parameter(torch.tensor(0.0))
        self.temperature_param2 = nn.Parameter(torch.tensor(0.0))
        
        if n_cls_tokens > 0:
            self.cls_token_embeddings = nn.Parameter(torch.zeros(1, n_cls_tokens, 1, d_model)) # batch_size, n_cls_tokens, 1, d_model

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, d_model))

    def forward(self, electrode_data, embeddings, masked_tokens=None, return_cls_token=False):
        # electrode_data is of shape (batch_size, n_electrodes, n_timebins, sample_timebin_size)
        # embeddings is of shape (batch_size, n_electrodes, d_model)
        # masked_tokens is of shape (batch_size, n_electrodes, n_timebins)
        assert not return_cls_token or self.n_cls_tokens > 0, "Cannot return CLS tokens if n_cls_tokens is 0"

        batch_size, n_electrodes, n_timebins, sample_timebin_size = electrode_data.shape
        positions = torch.arange(n_timebins, device=self.device, dtype=torch.long).unsqueeze(0).unsqueeze(0).repeat(batch_size, n_electrodes, 1) # shape: (batch_size, n_electrodes, n_timebins)
        embeddings = embeddings.unsqueeze(-2).repeat(1, 1, n_timebins, 1) # shape: (batch_size, n_electrodes, n_timebins, d_model)


        if masked_tokens is not None:
            embeddings = embeddings + self.mask_token * masked_tokens.unsqueeze(-1)

        if self.n_cls_tokens > 0:
            # Create cls tokens for each timebin
            cls_token_embeddings = self.cls_token_embeddings.repeat(batch_size, 1, n_timebins, 1)  # shape: (batch_size, n_cls_tokens, n_timebins, d_model)
            cls_token_positions = torch.arange(n_timebins, device=self.device, dtype=torch.long).unsqueeze(0).unsqueeze(0).repeat(batch_size, self.n_cls_tokens, 1) # shape: (batch_size, n_cls_tokens, n_timebins)
            
            embeddings = torch.cat([cls_token_embeddings, embeddings], dim=1)
            positions = torch.cat([cls_token_positions, positions], dim=1)
            electrode_data = torch.cat([torch.zeros(batch_size, self.n_cls_tokens, n_timebins, sample_timebin_size, device=self.device, dtype=self.dtype), electrode_data], dim=1)

        new_n_electrodes = electrode_data.shape[1] # might have changed if we added cls tokens
        electrode_data = electrode_data.reshape(batch_size, new_n_electrodes * n_timebins, sample_timebin_size)
        embeddings = embeddings.reshape(batch_size, new_n_electrodes * n_timebins, self.d_model)
        positions = positions.reshape(batch_size, new_n_electrodes * n_timebins)
    
        total_output = self.transformer(electrode_data, positions=positions, embeddings=embeddings) # shape: (batch_size, n_electrodes * n_timebins, d_model)
        
        if self.n_cls_tokens > 0:
            electrode_output = total_output[:, n_timebins * self.n_cls_tokens:, :]
            cls_output = total_output[:, :n_timebins * self.n_cls_tokens, :].reshape(batch_size, self.n_cls_tokens, n_timebins, sample_timebin_size)
        else:
            electrode_output = total_output
        
        electrode_output = electrode_output.reshape(batch_size, n_electrodes, n_timebins, sample_timebin_size) # shape: (batch_size, n_electrodes, n_timebins, d_model) # XXX Can probably avoid this transpose since I'm passing in positions and creating a mask

        if return_cls_token and self.n_cls_tokens > 0:
            return electrode_output, cls_output
        else:
            return electrode_output
    
    def generate_frozen_evaluation_features(self, electrode_data, embeddings, feature_aggregation_method='concat', just_return_cls_token=False):
        batch_size, n_electrodes, n_timebins, d_model = electrode_data.shape
        #if just_return_cls_token is None: just_return_cls_token = (self.n_cls_tokens > 0)
        assert not just_return_cls_token or self.n_cls_tokens > 0, "Cannot return CLS tokens if n_cls_tokens is 0"

        if self.n_cls_tokens > 0:
            output = self(electrode_data, embeddings, return_cls_token=True)
            if just_return_cls_token:
                output = output[1]
            else:
                output = torch.cat([output[1], output[0]], dim=1)
        else:
            output = self(electrode_data, embeddings, return_cls_token=False)

        if feature_aggregation_method == 'concat':
            return output.reshape(batch_size, -1)
        elif feature_aggregation_method == 'mean':
            return output.mean(dim=[1, 2])
        else:
            raise ValueError(f"Invalid feature aggregation method: {feature_aggregation_method}")


class CrossModel(BFModel):
    def __init__(self, sample_timebin_size, d_model, n_layers=5, n_heads=12):
        super().__init__()
        self.d_model = d_model
        self.sample_timebin_size = sample_timebin_size

        self.cross_transformer = CrossAttentionTransformer(d_input=self.sample_timebin_size, d_model=d_model, d_output=self.sample_timebin_size, 
                                            n_layer=n_layers, n_head=n_heads, rope=True)
        self.temperature_param = nn.Parameter(torch.tensor(0.0))

    def forward(self, embeddings_x, electrode_data_y, positions_offset=0):
        # embeddings_x is of shape (batch_size, n_electrodes_x, d_model)
        # electrode_data_y is of shape (batch_size, n_electrodes_y, n_timebins, sample_timebin_size)

        batch_size, n_electrodes_y, n_timebins, sample_timebin_size = electrode_data_y.shape
        n_electrodes_x = embeddings_x.shape[1]

        embeddings_x = embeddings_x.unsqueeze(1).repeat(1, n_timebins, 1, 1) # shape: (batch_size, n_timebins, n_electrodes_x, d_model)
        embeddings_x = embeddings_x.reshape(batch_size, n_electrodes_x * n_timebins, self.d_model)
        positions_x = torch.arange(n_timebins, device=self.device, dtype=torch.long).repeat(batch_size, n_electrodes_x, 1).reshape(batch_size, n_electrodes_x * n_timebins)
        positions_x = positions_x + positions_offset

        electrode_data_y = electrode_data_y.transpose(1, 2) # shape: (batch_size, n_timebins, n_electrodes_y, sample_timebin_size)
        electrode_data_y = electrode_data_y.reshape(batch_size, n_electrodes_y * n_timebins, sample_timebin_size)
        positions_y = torch.arange(n_timebins, device=self.device, dtype=torch.long).repeat(batch_size, n_electrodes_y, 1).reshape(batch_size, n_electrodes_y * n_timebins)

        electrode_output = self.cross_transformer(embeddings_x, electrode_data_y, positions_x=positions_x, positions_y=positions_y) # shape: (batch_size, n_electrodes * n_timebins, d_model)
        electrode_output = electrode_output.reshape(batch_size, n_timebins, n_electrodes_x, sample_timebin_size).transpose(1, 2) # shape: (batch_size, n_electrodes, n_timebins, d_model)

        return electrode_output


# class PerceiverIO(BFModel):
#     """PerceiverIO architecture that can handle inputs of arbitrary size.
    
#     The model uses cross-attention to project inputs into a fixed-size latent array,
#     followed by self-attention processing. This allows handling inputs of arbitrary size
#     while maintaining a fixed computational budget.
#     """
#     def __init__(self, d_model, d_output=None, n_latents=256, n_layers=6, n_heads=8, 
#                  n_cross_attn_layers=1, n_self_attn_layers=1, use_cls_token=False):
#         super().__init__()
#         self.d_model = d_model
#         self.n_latents = n_latents
#         self.use_cls_token = use_cls_token
        
#         if d_output is None:
#             d_output = d_model

#         # Learnable latent array
#         self.latents = nn.Parameter(torch.randn(n_latents, d_model))
        
#         # Cross-attention layers
#         self.cross_attn_layers = nn.ModuleList([
#             CrossAttentionTransformer(
#                 d_input=d_model,
#                 d_model=d_model,
#                 d_output=d_model,
#                 n_layer=n_cross_attn_layers,
#                 n_head=n_heads,
#                 rope=False
#             ) for _ in range(n_layers)
#         ])
        
#         # Self-attention layers
#         self.self_attn_layers = nn.ModuleList([
#             Transformer(
#                 d_input=d_model,
#                 d_model=d_model,
#                 d_output=d_model,
#                 n_layer=n_self_attn_layers,
#                 n_head=n_heads,
#                 causal=False,
#                 rope=False,
#                 cls_token=False
#             ) for _ in range(n_layers)
#         ])
        
#         # Output projection
#         self.output_proj = nn.Linear(d_model, d_output)
#         self.temperature_param = nn.Parameter(torch.tensor(1.0))

#     def forward(self, electrode_embedded_data):
#         # electrode_embedded_data shape: (batch_size, n_electrodes, n_timebins, d_model)
#         batch_size, n_electrodes, n_timebins, d_model = electrode_embedded_data.shape
        
#         # Reshape input for processing
#         x = electrode_embedded_data.reshape(batch_size, -1, d_model)  # (batch_size, n_electrodes*n_timebins, d_model)
        
#         # Initialize latent array for this batch
#         latents = self.latents.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, n_latents, d_model)
        
#         # Process through cross-attention and self-attention layers
#         for cross_attn, self_attn in zip(self.cross_attn_layers, self.self_attn_layers):
#             # Cross-attention: latents attend to input
#             latents = cross_attn(latents, x)
            
#             # Self-attention: latents attend to themselves
#             latents = self_attn(latents)
        
#         # Project to output dimension
#         output = self.output_proj(latents)  # (batch_size, n_latents, d_output)
        
#         return output

#     def generate_frozen_evaluation_features(self, electrode_embedded_data, feature_aggregation_method='concat'):
#         batch_size = electrode_embedded_data.shape[0]
#         output = self(electrode_embedded_data)  # (batch_size, n_latents, d_output)
        
#         if feature_aggregation_method == 'concat':
#             return output.reshape(batch_size, -1)
#         elif feature_aggregation_method == 'mean':
#             return output.mean(dim=1)
#         else:
#             raise ValueError(f"Invalid feature aggregation method: {feature_aggregation_method}")
