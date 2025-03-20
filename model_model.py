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

    def calculate_pretrain_test_loss(self, electrode_data_embedding_class, test_dataloader, all_subjects, calculate_loss_function=None):
        def _calculate_loss_function(batch, subject_identifier, trial_id):
            electrode_embedded_data = electrode_data_embedding_class.forward(subject_identifier, all_subjects[subject_identifier].get_electrode_indices(trial_id), batch)
            return self.calculate_pretrain_loss(electrode_embedded_data)
        if calculate_loss_function is None:
            calculate_loss_function = _calculate_loss_function

        losses = {}
        n_batches = 0
        for batch, (subject_identifiers, trial_ids) in test_dataloader:
            subject_identifier = subject_identifiers[0]
            trial_id = trial_ids[0].item()
            
            batch = batch.to(self.device, dtype=self.dtype, non_blocking=True)
            loss = calculate_loss_function(batch, subject_identifier, trial_id)
            
            if isinstance(loss, dict):
                for key, value in loss.items():
                    if key not in losses: losses[key] = 0
                    losses[key] += value
            else:
                if 'loss' not in losses: losses['loss'] = 0
                losses['loss'] += loss
            n_batches += 1
            
        if isinstance(loss, dict):
            return {k: v / n_batches for k, v in losses.items()}
        else:
            return losses['loss'] / n_batches

from model_transformers import Transformer
class TransformerModel(BFModel):
    def __init__(self, d_model, n_layers_electrode=5, n_layers_time=5, frequency_cutoff_dim=64):
        super().__init__()
        self.d_model = d_model
        self.frequency_cutoff_dim = frequency_cutoff_dim

        self.electrode_transformer = Transformer(d_input=d_model, d_model=d_model, d_output=d_model, n_layer=n_layers_electrode, n_head=12, causal=False, rope=False, cls_token=True)
        self.time_transformer = Transformer(d_input=d_model, d_model=d_model, d_output=d_model, n_layer=n_layers_time, n_head=12, causal=True, rope=True, cls_token=False)
        self.temperature_param = nn.Parameter(torch.tensor(1.0))

    
    def forward(self, embedded_electrode_data, only_electrode_output=False):
        # electrode_embeddings is of shape (n_electrodes, d_model)
        # embedded_electrode_data is of shape (batch_size, n_electrodes, n_timebins, d_model)

        electrode_data = embedded_electrode_data.permute(0, 2, 1, 3) # shape: (batch_size, n_timebins, n_electrodes, d_model)
        batch_size, n_timebins, n_electrodes, d_model = electrode_data.shape

        electrode_output = self.electrode_transformer(electrode_data.reshape(batch_size * n_timebins, n_electrodes, d_model)) # shape: (batch_size*n_timebins, n_electrodes+1, d_model)
        electrode_output = electrode_output[:, 0:1, :].view(batch_size, n_timebins, d_model) # just the CLS token. Shape: (batch_size, n_timebins, d_model)
        if only_electrode_output:
            return electrode_output, None
        
        time_output = self.time_transformer(electrode_output) # shape: (batch_size, n_timebins, d_model)
        return electrode_output, time_output


    def calculate_pretrain_loss(self, electrode_embedded_data, p_electrodes_per_stream=0.5, symmetric_loss=False):
        batch_size, n_electrodes, n_timebins, d_model = electrode_embedded_data.shape

        permutation = torch.randperm(n_electrodes)
        electrode_embedded_data = electrode_embedded_data[:, permutation]

        n_electrodes_per_stream = int(n_electrodes * p_electrodes_per_stream)
        o1_e, o1_t = self(electrode_embedded_data[:, :n_electrodes_per_stream, :, :]) # shape: (batch_size, n_timebins, d_model)
        o2_e, o2_t = self(electrode_embedded_data[:, -n_electrodes_per_stream:, :, :]) # shape: (batch_size, n_timebins, d_model)
        
        similarity_1 = torch.matmul(o1_t[:, :-1].permute(1, 0, 2), o2_e[:, 1:].permute(1, 2, 0)) * self.temperature_param
        expanded_arange = torch.arange(batch_size).unsqueeze(0).repeat(n_timebins-1, 1).to(self.device, dtype=torch.long).reshape(-1)
        loss = nn.functional.cross_entropy(similarity_1.view(-1, batch_size), expanded_arange)

        if symmetric_loss:
            similarity_2 = torch.matmul(o2_t[:, :-1].permute(1, 0, 2), o1_e[:, 1:].permute(1, 2, 0)) * self.temperature_param
            loss += nn.functional.cross_entropy(similarity_2.view(-1, batch_size), expanded_arange)
            loss /= 2

        return loss
    
    def generate_frozen_evaluation_features(self, electrode_embedded_data, feature_aggregation_method='concat'):
        batch_size, n_electrodes, n_timebins, d_model = electrode_embedded_data.shape
        if feature_aggregation_method == 'concat':
            return self(electrode_embedded_data, only_electrode_output=True)[0].reshape(batch_size, -1)
        elif feature_aggregation_method == 'mean':
            return self(electrode_embedded_data, only_electrode_output=True)[0].mean(dim=1)
        else:
            raise ValueError(f"Invalid feature aggregation method: {feature_aggregation_method}")
