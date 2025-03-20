import torch
import torch.nn as nn
from model_model import BFModule
    

class ElectrodeDataEmbedding(BFModule):
    def __init__(self, electrode_embedding_class):
        super(ElectrodeDataEmbedding, self).__init__()
        self.electrode_embedding_class = electrode_embedding_class
        self.d_model = electrode_embedding_class.d_model
        self.normalization_means = nn.ParameterDict({})
        self.normalization_stds = nn.ParameterDict({})

    def add_subject(self, subject):
        subject_identifier = subject.subject_identifier
        assert subject_identifier not in self.normalization_means, f"Subject identifier {subject_identifier} already in the class"
        self.electrode_embedding_class.add_embedding(subject)

    def forward(self, subject_identifier, electrode_data, permutation=None):
        # electrode_data is of shape (n_electrodes, n_timebins, n_samples)
        #   where n_samples = sample_timebin_size * n_timebins
        pass

class ElectrodeDataEmbeddingFFT(ElectrodeDataEmbedding):
    def __init__(self, electrode_embedding_class, sample_timebin_size, normalization_requires_grad=True):
        super(ElectrodeDataEmbeddingFFT, self).__init__(electrode_embedding_class)
        self.sample_timebin_size = sample_timebin_size
        self.normalization_requires_grad = normalization_requires_grad

        self.linear_embed = nn.ParameterDict({})
        self.sampling_rates = nn.ParameterDict({})
        self.n_frequency_bins = {}

    def add_subject(self, subject, sampling_rate, max_frequency_bin=None):
        super(ElectrodeDataEmbeddingFFT, self).add_subject(subject)
        subject_identifier = subject.subject_identifier

        n_frequency_bins = int(self.sample_timebin_size * sampling_rate) // 2 + 1 if max_frequency_bin is None else max_frequency_bin
        
        dataset_identifier = subject_identifier.rstrip('0123456789')
        if dataset_identifier not in self.linear_embed:
            self.linear_embed[dataset_identifier] = nn.Linear(n_frequency_bins, self.d_model)

            # Saving them as parameters so that they can be saved and loaded with the model
            self.sampling_rates[dataset_identifier] = nn.Parameter(torch.tensor(int(sampling_rate)), requires_grad=False)
            self.n_frequency_bins[dataset_identifier] = n_frequency_bins

        self.normalization_means[subject_identifier] = nn.Parameter(torch.zeros(subject.get_n_electrodes(), n_frequency_bins, dtype=self.dtype, device=self.device), requires_grad=self.normalization_requires_grad)
        self.normalization_stds[subject_identifier] = nn.Parameter(torch.ones(subject.get_n_electrodes(), n_frequency_bins, dtype=self.dtype, device=self.device), requires_grad=self.normalization_requires_grad)

    def initialize_normalization(self, subject, session_id, init_normalization_window_to=2048 * 60 * 10):
        subject_identifier = subject.subject_identifier
        dataset_identifier = subject_identifier.rstrip('0123456789')
        indices = subject.get_electrode_indices(session_id)

        with torch.no_grad():
            all_electrode_data = subject.get_all_electrode_data(session_id, window_to=init_normalization_window_to)

            timebin_samples = int(self.sample_timebin_size * self.sampling_rates[dataset_identifier])
            n_timebins = all_electrode_data.shape[1] // timebin_samples
            all_electrode_data = all_electrode_data[:, :n_timebins*timebin_samples]
            all_electrode_data = all_electrode_data.unsqueeze(0).to(self.device)

            all_electrode_data = self._forward_fft(subject_identifier, all_electrode_data).squeeze(0) # remove the batch dimension

            self.normalization_means[subject_identifier].data[indices] = all_electrode_data.mean(dim=1)
            self.normalization_stds[subject_identifier].data[indices] = all_electrode_data.std(dim=1) + 1e-5


    def _forward_fft(self, subject_identifier, x):
        # x is of shape (batch_size, n_electrodes, n_samples)
        #   where n_samples = sample_timebin_size * n_timebins
        batch_size, n_electrodes, n_samples = x.shape
        dataset_identifier = subject_identifier.rstrip('0123456789')


        timebin_samples = int(self.sample_timebin_size * self.sampling_rates[dataset_identifier])
        n_timebins = n_samples // timebin_samples
        
        # Calculate FFT for each timebin
        x = x.reshape(-1, timebin_samples)
        x = x.to(dtype=torch.float32)  # Convert to float32 for FFT
        x = torch.fft.rfft(x, dim=-1)  # Using rfft for real-valued input
        x = x[:, :self.n_frequency_bins[dataset_identifier]] # trimming in case max_frequency_bin is not None # XXX this should not be done here; for different datasets, this number could be different, so it should be set in add_subject
        x = x.reshape(batch_size, n_electrodes, n_timebins, -1)  # shape: (batch_size, n_electrodes, n_timebins, n_frequency_bins)
        
        # Calculate magnitude (equivalent to scipy.signal.stft's magnitude)
        x = torch.abs(x)
        # Convert to power
        x = torch.log(x + 1e-5)
        return x.to(dtype=self.dtype)  # Convert back to original dtype; shape (batch_size, n_electrodes, n_timebins, n_frequency_bins)

    def forward(self, subject_identifier, electrode_indices, electrode_data, max_n_electrodes=None):
        # electrode_data is of shape (batch_size, n_electrodes, n_samples)
        #   where n_samples = sample_timebin_size * n_timebins
        assert electrode_data.shape[1] == len(electrode_indices), f"Electrode data must have the same number of electrodes as the electrode indices, got {electrode_data.shape[1]} and {len(electrode_indices)}"
        max_n_electrodes = max_n_electrodes if max_n_electrodes is not None else 10000
        if max_n_electrodes is not None and electrode_data.shape[1] > max_n_electrodes:
            # Randomly select max_n_electrodes from all available electrodes
            perm = torch.randperm(electrode_data.shape[1])
            idx = perm[:max_n_electrodes]
            electrode_data = electrode_data[:, idx, :]
            electrode_means = self.normalization_means[subject_identifier][electrode_indices][idx]
            electrode_stds = self.normalization_stds[subject_identifier][electrode_indices][idx]
            electrode_embeddings = self.electrode_embedding_class.forward(subject_identifier)[electrode_indices][idx]
        else:
            electrode_means = self.normalization_means[subject_identifier][electrode_indices]
            electrode_stds = self.normalization_stds[subject_identifier][electrode_indices]
            electrode_embeddings = self.electrode_embedding_class.forward(subject_identifier)[electrode_indices]

        electrode_data = self._forward_fft(subject_identifier, electrode_data) # shape: (batch_size, n_electrodes, n_timebins, n_frequency_bins)
        batch_size, n_electrodes, n_timebins, n_frequency_bins = electrode_data.shape
        dataset_identifier = subject_identifier.rstrip('0123456789')

        electrode_data = electrode_data - electrode_means.view(1, n_electrodes, 1, n_frequency_bins)
        electrode_data = electrode_data / electrode_stds.view(1, n_electrodes, 1, n_frequency_bins)

        electrode_data = self.linear_embed[dataset_identifier](electrode_data) # shape: (batch_size, n_electrodes, n_timebins, d_model)
        electrode_data = electrode_data + electrode_embeddings.view(1, n_electrodes, 1, self.d_model)

        return electrode_data # shape: (batch_size, n_electrodes, n_timebins, d_model)
    

    def load_state_dict(self, state_dict, strict=True):
        # Extract all unique subject identifiers from the state dict keys
        subject_identifiers = set()
        for key in state_dict.keys():
            if key.startswith('normalization_means.') or key.startswith('normalization_stds.') or key.startswith('electrode_embedding_class.embeddings.'):
                subject_identifier = key.split('.')[-1]
                subject_identifiers.add(subject_identifier)
        
        # Create dummy subjects to initialize the parameter dictionaries
        for subject_identifier in subject_identifiers:
            if subject_identifier not in self.normalization_means:
                # Get shapes from the state dict
                means_shape = state_dict[f'normalization_means.{subject_identifier}'].shape
                stds_shape = state_dict[f'normalization_stds.{subject_identifier}'].shape
                embedding_shape = state_dict[f'electrode_embedding_class.embeddings.{subject_identifier}'].shape
                # Initialize parameters with correct shapes
                self.normalization_means[subject_identifier] = nn.Parameter(
                    torch.zeros(means_shape, dtype=self.dtype, device=self.device),
                    requires_grad=self.normalization_requires_grad
                )
                self.normalization_stds[subject_identifier] = nn.Parameter(
                    torch.ones(stds_shape, dtype=self.dtype, device=self.device),
                    requires_grad=self.normalization_requires_grad
                )
                self.electrode_embedding_class.embeddings[subject_identifier] = nn.Parameter(
                    torch.zeros(embedding_shape, dtype=self.dtype, device=self.device)
                )

                dataset_identifier = subject_identifier.rstrip('0123456789')
                if dataset_identifier not in self.linear_embed:
                    self.linear_embed[dataset_identifier] = nn.Linear(
                        state_dict[f'linear_embed.{dataset_identifier}.weight'].size(1),
                        state_dict[f'linear_embed.{dataset_identifier}.weight'].size(0)
                    )
                    self.linear_embed[dataset_identifier].weight = nn.Parameter(state_dict[f'linear_embed.{dataset_identifier}.weight'])
                    self.linear_embed[dataset_identifier].bias = nn.Parameter(state_dict[f'linear_embed.{dataset_identifier}.bias'])
                    self.sampling_rates[dataset_identifier] = nn.Parameter(
                        state_dict[f'sampling_rates.{dataset_identifier}'],
                        requires_grad=False
                    )
                    self.n_frequency_bins[dataset_identifier] = state_dict[f'normalization_means.{subject_identifier}'].size(1)
        # Now we can load the state dict normally
        return super().load_state_dict(state_dict, strict=strict)

class ElectrodeEmbedding(BFModule):
    def __init__(self, d_model):
        super(ElectrodeEmbedding, self).__init__()
        # Every key must be a unique string identifier for a subject
        #   and be a NumPy array of shape (n_electrodes, *) where * is any number of additional dimensions of any size
        self.embeddings = nn.ParameterDict({})
        self.d_model = d_model

    def forward(self, subject_identifier):
        assert subject_identifier in self.embeddings, f"Subject identifier {subject_identifier} not found in embeddings"
        return self.embeddings[subject_identifier]
    
    def add_embedding(self, subject, embedding_init, requires_grad=True):
        subject_identifier = subject.subject_identifier
        assert subject_identifier not in self.embeddings, f"Subject identifier {subject_identifier} already in embeddings"
        self.embeddings[subject_identifier] = nn.Parameter(embedding_init, requires_grad=requires_grad)


class ElectrodeEmbedding_Learned(ElectrodeEmbedding):
    def __init__(self, d_model, embedding_dim=None, embedding_fanout_requires_grad=True):
        super(ElectrodeEmbedding_Learned, self).__init__(d_model)
        self.embedding_dim = embedding_dim if embedding_dim is not None else d_model
        if self.embedding_dim < d_model:
            self.linear_embed = nn.Linear(self.embedding_dim, self.d_model)
            self.linear_embed.weight.requires_grad = embedding_fanout_requires_grad
            self.linear_embed.bias.requires_grad = embedding_fanout_requires_grad
        else: 
            self.linear_embed = lambda x: x # just identity function if embedding dim is already at d_model
    
    def add_embedding(self, subject, embedding_init=None, requires_grad=True):
        subject_identifier = subject.subject_identifier
        n_electrodes = subject.get_n_electrodes()
        if embedding_init is None: embedding_init = torch.zeros(n_electrodes, self.embedding_dim)
        super(ElectrodeEmbedding_Learned, self).add_embedding(subject, embedding_init, requires_grad)
    
    def forward(self, subject_identifier):
        embedding = super(ElectrodeEmbedding_Learned, self).forward(subject_identifier)
        return self.linear_embed(embedding)
    
class ElectrodeEmbedding_Learned_FixedVocabulary(BFModule):
    def __init__(self, d_model, vocabulary_channels, embedding_dim=None, embedding_fanout_requires_grad=True):
        """
            vocabulary is a list of strings, each corresponding to each recorded channel name.
            All the channels that are not in the vocabulary will be set to an "overflow" vector that can be learned.
        """
        super(ElectrodeEmbedding_Learned_FixedVocabulary, self).__init__()
        self.d_model = d_model
        self.vocabulary_channels = [x.upper() for x in vocabulary_channels]
        self.embedding_dim = embedding_dim if embedding_dim is not None else d_model
        if self.embedding_dim < d_model:
            self.linear_embed = nn.Linear(self.embedding_dim, self.d_model)
            self.linear_embed.weight.requires_grad = embedding_fanout_requires_grad
            self.linear_embed.bias.requires_grad = embedding_fanout_requires_grad
        else: 
            self.linear_embed = lambda x: x # just identity function if embedding dim is already at d_model

        # this will only store the list of indices to the vocabulary
        self.embedding = nn.ParameterDict({})
        self.vocabulary = nn.Embedding(len(vocabulary_channels)+1, self.embedding_dim)
        self.vocabulary.weight.data.zero_()
    
    def add_embedding(self, subject, embedding_init=None, requires_grad=False):
        # The embedding is just a list of indices to the vocabulary
        subject_identifier = subject.subject_identifier
        n_electrodes = subject.get_n_electrodes()
        if embedding_init is None: embedding_init = torch.ones(n_electrodes) * len(self.vocabulary_channels) # by default, all electrodes are set to the last vector in the vocabulary
        electrode_labels = subject.get_electrode_labels()
        for i, label in enumerate(electrode_labels):
            if label.upper() in self.vocabulary_channels:
                embedding_init[i] = self.vocabulary_channels.index(label.upper())
        self.embedding[subject_identifier] = nn.Parameter(embedding_init.long(), requires_grad=False)

    def forward(self, subject_identifier):
        all_indices = self.embedding[subject_identifier]
        embeddings = self.vocabulary(all_indices)  # Shape: (n_electrodes, embedding_dim)
        return self.linear_embed(embeddings)

    
class ElectrodeEmbedding_Learned_CoordinateInit(ElectrodeEmbedding_Learned):
    """
        This class will initialize the embedding with the coordinates of the electrodes, and then (if requires_grad is True)
        they can behave just the same as the ElectrodeEmbedding_Learned
    """
    def __init__(self, d_model, embedding_dim=None, embedding_fanout_requires_grad=True):
        super(ElectrodeEmbedding_Learned_CoordinateInit, self).__init__(d_model, embedding_dim=embedding_dim, 
                                                             embedding_fanout_requires_grad=embedding_fanout_requires_grad)
    
    def add_embedding(self, subject, coordinate_scale=150, requires_grad=True):
        """
            "Electrode Coordinates" must be LPI coordinates, normalized to [0,1] range given min/max
        """
        subject_identifier = subject.subject_identifier
        n_electrodes = subject.get_n_electrodes()
        electrode_coordinates = subject.get_electrode_coordinates().to(dtype=self.dtype) / coordinate_scale
        assert electrode_coordinates.shape == (n_electrodes, 3), f"Electrode coordinates must be of shape (n_electrodes, 3), got {electrode_coordinates.shape}"

        dim_freq = self.embedding_dim//6
        freq = 100.0 / (200 ** (torch.arange(0, dim_freq, dtype=self.dtype)/dim_freq)) # coordinates are normalized in the add_embedding to lie between 0 and 1

        # Calculate position encodings for each coordinate dimension
        l_enc = electrode_coordinates[:, 0:1] @ freq.unsqueeze(0)  # For L coordinate
        i_enc = electrode_coordinates[:, 1:2] @ freq.unsqueeze(0)  # For I coordinate  
        p_enc = electrode_coordinates[:, 2:3] @ freq.unsqueeze(0)  # For P coordinate
        padding_zeros = torch.zeros(n_electrodes, self.embedding_dim % dim_freq, dtype=self.dtype) # padding in case model dimension is not divisible by 6

        # Combine sin and cos encodings
        embedding = torch.cat([torch.sin(l_enc), torch.cos(l_enc), torch.sin(i_enc), torch.cos(i_enc), torch.sin(p_enc), torch.cos(p_enc), padding_zeros], dim=-1)
        embedding = embedding.to(self.device)
        super(ElectrodeEmbedding_Learned_CoordinateInit, self).add_embedding(subject, embedding_init=embedding, requires_grad=requires_grad)


class ElectrodeEmbedding_NoisyCoordinate(ElectrodeEmbedding_Learned):
    """
        This class will create the embedding with the positional encoding of the coordinates of the electrodes, 
        and then in each forward pass add random noise to the coordinates. 
        These embeddings are not learnable apart from a fanout linear layer (in case embedding_dim<d_model).
    """
    def __init__(self, d_model, coordinate_noise_std=0.0, embedding_dim=None, embedding_fanout_requires_grad=True):
        super(ElectrodeEmbedding_NoisyCoordinate, self).__init__(d_model, embedding_dim=embedding_dim, 
                                                             embedding_fanout_requires_grad=embedding_fanout_requires_grad)
        self.coordinate_noise_std = coordinate_noise_std
    
    def add_embedding(self, subject, coordinate_scale=150):
        """
            "Electrode Coordinates" must be LPI coordinates, normalized to [0,1] range given min/max
        """
        n_electrodes = subject.get_n_electrodes()
        electrode_coordinates = subject.get_electrode_coordinates().to(dtype=self.dtype) / coordinate_scale
        assert electrode_coordinates.shape == (n_electrodes, 3), f"Electrode coordinates must be of shape (n_electrodes, 3), got {electrode_coordinates.shape}"

        super(ElectrodeEmbedding_NoisyCoordinate, self).add_embedding(subject, embedding_init=electrode_coordinates, requires_grad=False)

    def forward(self, subject_identifier):
        electrode_coordinates = super(ElectrodeEmbedding_NoisyCoordinate, self).forward(subject_identifier)
        electrode_coordinates += torch.randn_like(electrode_coordinates) * self.coordinate_noise_std
        n_electrodes = electrode_coordinates.shape[0]

        dim_freq = self.embedding_dim//6
        freq = 100.0 / (200 ** (torch.arange(0, dim_freq, dtype=self.dtype, device=self.device)/dim_freq)) # coordinates are normalized in the add_embedding to lie between 0 and 1

        # Calculate position encodings for each coordinate dimension
        l_enc = electrode_coordinates[:, 0:1] @ freq.unsqueeze(0)  # For L coordinate
        i_enc = electrode_coordinates[:, 1:2] @ freq.unsqueeze(0)  # For I coordinate  
        p_enc = electrode_coordinates[:, 2:3] @ freq.unsqueeze(0)  # For P coordinate
        padding_zeros = torch.zeros(n_electrodes, self.embedding_dim % dim_freq, dtype=self.dtype, device=self.device) # padding in case model dimension is not divisible by 6

        # Combine sin and cos encodings
        embedding = torch.cat([torch.sin(l_enc), torch.cos(l_enc), torch.sin(i_enc), torch.cos(i_enc), torch.sin(p_enc), torch.cos(p_enc), padding_zeros], dim=-1)
        return embedding.to(self.device)