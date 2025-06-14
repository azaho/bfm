from model.andrii_original_model import OriginalModel
from model.electrode_embedding import ElectrodeEmbedding_Learned, ElectrodeEmbedding_NoisyCoordinate, ElectrodeEmbedding_Learned_CoordinateInit, ElectrodeEmbedding_Zero
from model.preprocessing.laplacian_rereferencing import laplacian_rereference_batch
from utils.training_config import log
import torch
from training_setup.training_setup import TrainingSetup

class andrii0(TrainingSetup):
    def __init__(self, all_subjects, config, verbose=True):
        super().__init__(all_subjects, config, verbose)

    def initialize_model(self):
        """
            This function initializes the model.

            It must set the self.model_components dictionary to a dictionary of the model components, like
            {"model": model, "electrode_embeddings": electrode_embeddings}, where model and electrode_embeddings are PyTorch modules (those classes must inherit from model.BFModule)
        """
        config = self.config
        device = config['device']
        assert config['model']['signal_preprocessing']['spectrogram'] == True, "For the moment, we only support spectrogram"

        ### LOAD MODEL ###

        self.model = OriginalModel(
            d_model=config['model']['transformer']['d_model'],
            n_layers_electrode=config['model']['transformer']['n_layers_electrode'],
            n_layers_time=config['model']['transformer']['n_layers_time'],
            n_heads=config['model']['transformer']['n_heads'],
            dropout=config['training']['dropout']
        ).to(device, dtype=config['model']['dtype'])
        config['model']['name'] = "AndriiOriginalModel"

        ### LOAD ELECTRODE EMBEDDINGS ###

        electrode_embeddings_class = { # Select the right class based on the config
            'learned': ElectrodeEmbedding_Learned,
            'zero': ElectrodeEmbedding_Zero,
            'coordinate_init': ElectrodeEmbedding_Learned_CoordinateInit,
            'noisy_coordinate': ElectrodeEmbedding_NoisyCoordinate,
        }[config['model']['electrode_embedding']['type']]

        self.electrode_embeddings = electrode_embeddings_class( # Initialize the electrode embeddings
            config['model']['transformer']['d_model'], 
            embedding_dim=config['model']['electrode_embedding']['dim'],
            coordinate_noise_std=config['model']['electrode_embedding']['coordinate_noise_std'],
        ).to(device, dtype=config['model']['dtype'])

        for subject in self.all_subjects.values(): # we need to add every subject one by one to create the embeddings map (every electrode of every subject gets its own embedding)
            if self.verbose:
                log(f"Adding subject {subject.subject_identifier} to electrode embeddings...", priority=0)
            self.electrode_embeddings.add_subject(subject)
        self.electrode_embeddings = self.electrode_embeddings.to(device, dtype=config['model']['dtype']) # moving to device again to ensure the new parameters are on the correct device

        self.model_components['model'] = self.model
        self.model_components['electrode_embeddings'] = self.electrode_embeddings


    def calculate_pretrain_loss(self, batch, output_accuracy=True):
        # INPUT:
        #   batch['data'] shape: (batch_size, n_electrodes, n_timesamples)
        #   batch['electrode_index'] shape: (batch_size, n_electrodes)
        #   batch['metadata']: dictionary containing metadata like the subject identifier and trial id, sampling rate, etc.
        # OUTPUT:
        #   This function will output a dictionary of losses, with the keys being the loss names and the values being the loss values.
        #   The final loss is the mean of all the losses. Accuracies are exempt and are just used for logging.
        
        losses = {}
        config = self.config
        def _add_to_loss_contrastive(losses, output, target, loss_suffix):
            # output and target shape: (batch_size, n_electrodes, n_timebins-future_bin_idx, d_model)
            if config['training']['normalize_features']:
                output_ = output / (torch.norm(output, dim=-1, keepdim=True) + 0.001)
                target_ = target / (torch.norm(target, dim=-1, keepdim=True) + 0.001)
            similarity = output_.permute(1, 2, 0, 3) @ target_.permute(1, 2, 3, 0) # shape: (n_electrodes, n_timebins-future_bin_idx, batch_size, batch_size)
            if config['training']['use_temperature_param']:
                similarity = similarity * torch.minimum(torch.exp(self.model.temperature_param), torch.tensor(config['training']['max_temperature_param'], device=self.model.device, dtype=self.model.dtype))
            expanded_arange = torch.arange(batch_size).unsqueeze(0).unsqueeze(0).repeat(output.shape[1], output.shape[2], 1).to(self.model.device, dtype=torch.long).reshape(-1)

            loss = torch.nn.functional.cross_entropy(similarity.view(-1, batch_size), expanded_arange)
            losses[f'contrastive_{loss_suffix}'] = loss
            if output_accuracy:
                accuracy_bin = (similarity.view(-1, batch_size).argmax(dim=-1) == expanded_arange).float().mean()
                losses[f'accuracy_{loss_suffix}'] = accuracy_bin
            return losses
        future_bin_idx = config['training']['future_bin_idx']

        # Note that due to the RandomELectrodeCollator in the dataset class, the electrodes are already shuffled and cut to max_n_electrodes
        batch_size, n_electrodes, n_samples = batch['data'].shape
        
        if config['model']['signal_preprocessing']['laplacian_rereference']:
            laplacian_rereference_batch(batch, remove_non_laplacian=False, inplace=True)
            
        if config['model']['signal_preprocessing']['normalize_voltage']:
            batch['data'] = batch['data'] - torch.mean(batch['data'], dim=[0, 2], keepdim=True)
            batch['data'] = batch['data'] / (torch.std(batch['data'], dim=[0, 2], keepdim=True) + 1)

        # Split the batch into two halves, so that we can compute the contrastive loss on the two halves
        batch_a = {
            'data': batch['data'][:, :n_electrodes//2, :],
            'electrode_index': batch['electrode_index'][:, :n_electrodes//2],
            'metadata': batch['metadata'],
        }
        batch_b = {
            'data': batch['data'][:, n_electrodes//2:, :],
            'electrode_index': batch['electrode_index'][:, n_electrodes//2:],
            'metadata': batch['metadata'],
        }

        embeddings_a = self.electrode_embeddings(batch_a)
        embeddings_b = self.electrode_embeddings(batch_b)
        electrode_transformed_data_a, time_transformed_data_a = self.model(batch_a, embeddings_a) # shape: (batch_size, 1, n_timebins, d_model)
        electrode_transformed_data_b, time_transformed_data_b = self.model(batch_b, embeddings_b) # shape: (batch_size, 1, n_timebins, d_model)

        # add two symmetric loss components
        losses = _add_to_loss_contrastive(losses, time_transformed_data_a[:, :, :-future_bin_idx], electrode_transformed_data_b[:, :, future_bin_idx:], 'a')
        losses = _add_to_loss_contrastive(losses, time_transformed_data_b[:, :, :-future_bin_idx], electrode_transformed_data_a[:, :, future_bin_idx:], 'b')

        return losses


    def generate_frozen_features(self, batch):
        # INPUT:
        #   batch['data'] shape: (batch_size, n_electrodes, n_timebins)
        #   batch['electrode_labels'] shape: list of length 1 (since it's the same across the batch), each element is a list of electrode labels
        #   batch['metadata']: dictionary containing metadata like the subject identifier and trial id, sampling rate, etc.
        # OUTPUT:
        #   features shape: (batch_size, feature_vector_length) where feature_vector_length can be arbitrary
        
        config = self.config
        electrode_indices = []
        subject_identifier = batch['metadata']['subject_identifier']
        for electrode_label in batch['electrode_labels'][0]:
            key = (subject_identifier, electrode_label)
            electrode_indices.append(self.electrode_embeddings.embeddings_map[key])
        batch['electrode_index'] = torch.tensor(electrode_indices, device=self.model.device, dtype=torch.long).unsqueeze(0).expand(batch['data'].shape[0], -1) # shape: (batch_size, n_electrodes)
            
        if config['model']['signal_preprocessing']['laplacian_rereference']:
            laplacian_rereference_batch(batch, remove_non_laplacian=False, inplace=True)
        
        if config['model']['signal_preprocessing']['normalize_voltage']:
            batch['data'] = batch['data'] - torch.mean(batch['data'], dim=[0, 2], keepdim=True)
            batch['data'] = batch['data'] / (torch.std(batch['data'], dim=[0, 2], keepdim=True) + 1)

        embeddings = self.electrode_embeddings(batch)
        features = self.model(batch, embeddings, evaluation_features=True) # shape: (batch_size, 1, n_timebins, d_model)

        if config['cluster']['eval_aggregation_method'] == 'mean':
            features = features.mean(dim=[1, 2])
        elif config['cluster']['eval_aggregation_method'] == 'concat':
            features = features.reshape(batch['data'].shape[0], -1)
        return features