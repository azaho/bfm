import torch
import torch.nn as nn
from model_transformers import BFModule

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
    def __init__(self, overall_sampling_rate=2048, sample_timebin_size=0.125, identity_init=True, reverse=False):
        super(LinearBinTransformer, self).__init__()
        self.overall_sampling_rate = overall_sampling_rate
        self.sample_timebin_size = sample_timebin_size
        self.linear = nn.Linear(int(overall_sampling_rate*sample_timebin_size), int(overall_sampling_rate*sample_timebin_size), bias=False)

        if identity_init:
            self.linear.weight.data = torch.eye(int(overall_sampling_rate*sample_timebin_size)).to(self.device, dtype=self.dtype)
            if reverse:
                self.linear.weight.data = self.linear.weight.data.flip(dims=(0,))
        
    def forward(self, electrode_data):
        # (batch_size, n_electrodes, n_samples)
        batch_size, n_electrodes, n_samples = electrode_data.shape
        electrode_data = electrode_data.reshape(batch_size, n_electrodes, -1, int(self.overall_sampling_rate*self.sample_timebin_size))

        batch_size, n_electrodes, n_timebins, sample_timebin_size = electrode_data.shape
        electrode_data = self.linear(electrode_data)

        return electrode_data

class LinearKernelTransformer(BFModule):
    def __init__(self, first_kernel=16, overall_sampling_rate=2048, sample_timebin_size=0.125, identity_init=True, reverse=False):
        super(LinearKernelTransformer, self).__init__()
        self.overall_sampling_rate = overall_sampling_rate
        self.sample_timebin_size = sample_timebin_size
        self.linear = nn.Linear(first_kernel, first_kernel, bias=False)
        self.first_kernel = first_kernel

        self.temperature_param = nn.Parameter(torch.tensor(0.0))

        if identity_init:
            self.linear.weight.data = torch.eye(first_kernel).to(self.device, dtype=self.dtype)
            if reverse:
                self.linear.weight.data = self.linear.weight.data.flip(dims=(0,))
        
    def forward(self, electrode_data):
        # (batch_size, n_electrodes, n_samples)
        batch_size, n_electrodes = electrode_data.shape[:2]
        electrode_data = electrode_data.reshape(batch_size, n_electrodes, -1, self.first_kernel)
        electrode_data = self.linear(electrode_data)
        return electrode_data
    
    def generate_frozen_evaluation_features(self, electrode_data, embeddings, feature_aggregation_method='concat'):
        batch_size = electrode_data.shape[0]
        return self.forward(electrode_data).reshape(batch_size, -1)

from model_transformers import Transformer
class BinTransformer(BFModule):
    def __init__(self, first_kernel=16, d_model=192, n_layers=5, n_heads=6, overall_sampling_rate=2048, sample_timebin_size=0.125, n_downsample_factor=1, identity_init=True, keep_kernel_dim=False):
        super(BinTransformer, self).__init__()
        self.transformer = Transformer(d_input=first_kernel//n_downsample_factor, d_model=d_model, d_output=first_kernel//n_downsample_factor, 
                                            n_layer=n_layers, n_head=n_heads, causal=True, 
                                            rope=True, cls_token=False, rope_base=int(overall_sampling_rate*sample_timebin_size)*n_downsample_factor//first_kernel, identity_init=identity_init)
        self.first_kernel = first_kernel
        self.sample_timebin_size = sample_timebin_size
        self.overall_sampling_rate = overall_sampling_rate  
        self.keep_kernel_dim = keep_kernel_dim
        self.n_layers = n_layers
    
    def forward(self, electrode_data):
        # (batch_size, n_electrodes, n_samples)
        batch_size, n_electrodes = electrode_data.shape[:2]
        electrode_data = electrode_data.reshape(batch_size, n_electrodes, -1, self.first_kernel)
        # Combine batch_size and n_electrodes dimensions
        electrode_data = electrode_data.reshape(batch_size * n_electrodes, -1, self.first_kernel)
        #electrode_data = electrode_data.reshape(batch_size, -1, self.first_kernel//self.n_downsample_factor)
        electrode_data = self.transformer(electrode_data) 

        electrode_data = electrode_data.reshape(batch_size, n_electrodes, -1, self.first_kernel)
        
        if not self.keep_kernel_dim:
            sample_timebin_size = int(self.overall_sampling_rate*self.sample_timebin_size)
            electrode_data = electrode_data.reshape(batch_size, n_electrodes, -1, sample_timebin_size)
        
        return electrode_data
    
    def generate_frozen_evaluation_features(self, electrode_data, embeddings, feature_aggregation_method='concat'):
        # (batch_size, n_electrodes, n_samples)
        batch_size, n_electrodes = electrode_data.shape[:2]
        electrode_data = electrode_data.reshape(batch_size, n_electrodes, -1, self.first_kernel)
        # Combine batch_size and n_electrodes dimensions
        electrode_data = electrode_data.reshape(batch_size * n_electrodes, -1, self.first_kernel)
        #electrode_data = electrode_data.reshape(batch_size, -1, self.first_kernel//self.n_downsample_factor)
        electrode_data = self.transformer(electrode_data, stop_at_block=self.n_layers) # shape: (batch_size * n_electrodes, n_timebins, d_model)
        
        electrode_data = electrode_data.reshape(batch_size, -1)
        return electrode_data


from model_transformers import Transformer, CrossAttentionTransformer
class GranularModel(BFModel):
    def __init__(self, sample_timebin_size, d_model, n_layers=5, n_heads=12, identity_init=True, n_cls_tokens=0):
        super().__init__()
        self.d_model = d_model
        self.sample_timebin_size = sample_timebin_size
        self.n_cls_tokens = n_cls_tokens
        self.n_layers = n_layers

        self.transformer = Transformer(d_input=self.sample_timebin_size, d_model=d_model, d_output=self.sample_timebin_size, 
                                            n_layer=n_layers, n_head=n_heads, causal=True, 
                                            rope=True, cls_token=False, identity_init=identity_init)
        self.temperature_param = nn.Parameter(torch.tensor(0.0))
        self.temperature_param2 = nn.Parameter(torch.tensor(0.0))
        
        if n_cls_tokens > 0:
            self.cls_token_embeddings = nn.Parameter(torch.zeros(n_cls_tokens, d_model)) # batch_size, n_cls_tokens, 1, d_model

        self.mask_token = nn.Parameter(torch.zeros(1, d_model))

    def forward(self, electrode_data, embeddings, masked_tokens=None, return_cls_token=False, strict_positions=False, stop_at_block=None):
        # electrode_data is of shape (batch_size, n_electrodes, n_timebins, sample_timebin_size)
        # embeddings is of shape (batch_size, n_electrodes, d_model)
        # masked_tokens is of shape (batch_size, n_electrodes, n_timebins)
        assert not return_cls_token or self.n_cls_tokens > 0, "Cannot return CLS tokens if n_cls_tokens is 0"

        batch_size, n_electrodes, n_timebins, sample_timebin_size = electrode_data.shape
        positions = torch.arange(n_timebins, device=self.device, dtype=torch.long).unsqueeze(0).unsqueeze(0).repeat(batch_size, n_electrodes, 1) # shape: (batch_size, n_electrodes, n_timebins)
        
        if len(embeddings.shape) == 3:
            embeddings = embeddings.unsqueeze(-2).repeat(1, 1, n_timebins, 1) # shape: (batch_size, n_electrodes, n_timebins, d_model)

        if masked_tokens is not None:
            embeddings = embeddings + self.mask_token.reshape(1, 1, 1, -1) * masked_tokens.unsqueeze(-1)

        if self.n_cls_tokens > 0:
            # Create cls tokens for each timebin
            cls_token_embeddings = self.cls_token_embeddings.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, n_timebins, 1)  # shape: (batch_size, n_cls_tokens, n_timebins, d_model)
            cls_token_positions = torch.arange(n_timebins, device=self.device, dtype=torch.long).unsqueeze(0).unsqueeze(0).repeat(batch_size, self.n_cls_tokens, 1) # shape: (batch_size, n_cls_tokens, n_timebins)
            
            embeddings = torch.cat([cls_token_embeddings, embeddings], dim=1)
            positions = torch.cat([cls_token_positions, positions], dim=1)
            electrode_data = torch.cat([torch.zeros(batch_size, self.n_cls_tokens, n_timebins, sample_timebin_size, device=self.device, dtype=self.dtype), electrode_data], dim=1)

        new_n_electrodes = electrode_data.shape[1] # might have changed if we added cls tokens
        electrode_data = electrode_data.reshape(batch_size, new_n_electrodes * n_timebins, sample_timebin_size)
        embeddings = embeddings.reshape(batch_size, new_n_electrodes * n_timebins, self.d_model)
        positions = positions.reshape(batch_size, new_n_electrodes * n_timebins)
    
        total_output = self.transformer(electrode_data, positions=positions, embeddings=embeddings, strict_positions=strict_positions, stop_at_block=stop_at_block) # shape: (batch_size, n_electrodes * n_timebins, d_model)
        sample_timebin_size = total_output.shape[-1] # if stop at block, this will be the d_model of the last block

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
            output = self(electrode_data, embeddings, return_cls_token=True, stop_at_block=self.n_layers)
            if just_return_cls_token:
                output = output[1]
            else:
                output = torch.cat([output[1], output[0]], dim=1)
        else:
            output = self(electrode_data, embeddings, return_cls_token=False, stop_at_block=self.n_layers)

        if feature_aggregation_method == 'concat':
            return output.reshape(batch_size, -1)
        elif feature_aggregation_method == 'mean':
            return output.mean(dim=[1, 2])
        else:
            raise ValueError(f"Invalid feature aggregation method: {feature_aggregation_method}")