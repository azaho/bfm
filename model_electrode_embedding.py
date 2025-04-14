import torch
import torch.nn as nn
from model_model import BFModule
    

class ElectrodeDataEmbedding(BFModule):
    def __init__(self, electrode_embedding_class, 
                 sample_timebin_size, overall_sampling_rate, 
                 normalization_requires_grad=True, std_smoothing=1e-5, 
                 initial_capacity=100, normalization_shape=(1, )):
        super(ElectrodeDataEmbedding, self).__init__()
        self.electrode_embedding_class = electrode_embedding_class
        self.d_model = electrode_embedding_class.d_model
        self.std_smoothing = std_smoothing
        self.sample_timebin_size = sample_timebin_size
        self.normalization_requires_grad = normalization_requires_grad
        self.normalization_shape = normalization_shape

        # Initialize embeddings and maps
        self._initialize_parameters(initial_capacity)

        self.overall_sampling_rate = overall_sampling_rate # XXX think about how to be flexible here regarding the sampling rate. Maybe, resample? Or learn a linear layer?
        self.linear_embed = nn.Linear(int(self.sample_timebin_size * self.overall_sampling_rate), self.d_model)

    def _get_current_capacity(self):
        return self.normalization_means.shape[0]
    def _get_current_size(self):
        return len(self.electrode_embedding_class.embeddings_map)
    def _initialize_parameters(self, initial_capacity):
        self.normalization_means = nn.Parameter(
            torch.zeros((initial_capacity, *self.normalization_shape), dtype=self.dtype, device=self.device),
            requires_grad=self.normalization_requires_grad
        )
        self.normalization_stds = nn.Parameter(
            torch.ones((initial_capacity, *self.normalization_shape), dtype=self.dtype, device=self.device),
            requires_grad=self.normalization_requires_grad
        )
        self.sampling_rates = nn.Parameter(
            torch.zeros(initial_capacity, dtype=torch.int64, device=self.device),
            requires_grad=False
        )
    def _ensure_capacity(self, needed_size):
        if needed_size >= self._get_current_capacity():
            new_capacity = needed_size
            # Save old data
            old_means = self.normalization_means.data
            old_stds = self.normalization_stds.data
            old_rates = self.sampling_rates.data
            # Initialize new parameters
            self._initialize_parameters(new_capacity)
            # Copy old data
            self.normalization_means.data[:old_means.shape[0]] = old_means
            self.normalization_stds.data[:old_stds.shape[0]] = old_stds
            self.sampling_rates.data[:old_rates.shape[0]] = old_rates

    def add_subject(self, subject, sampling_rate):
        self.electrode_embedding_class.add_subject(subject)
        # Get the new size directly from the embedding class after adding the subject
        needed_size = self._get_current_size()
        self._ensure_capacity(needed_size)

        for electrode_label in subject.get_electrode_labels():
            electrode_index = self.electrode_embedding_class.embeddings_map[(subject.subject_identifier, electrode_label)]
            self.sampling_rates.data[electrode_index] = sampling_rate

    def preprocess_electrode_data(self, electrode_data, sampling_rate, allow_trim=False):
        batch_size, n_electrodes, n_samples = electrode_data.shape

        timebin_samples = int(self.sample_timebin_size * sampling_rate)
        n_timebins = n_samples // timebin_samples
        
        # trim the data to a multiple of the timebin size
        if n_samples % timebin_samples != 0:
            assert allow_trim or (n_samples % timebin_samples == 0), f"n_samples must be a multiple of timebin_samples, got {n_samples} and {timebin_samples}"
            electrode_data = electrode_data[:, :, :n_timebins*timebin_samples]

        # reshape data
        electrode_data = electrode_data.view(batch_size, n_electrodes, n_timebins, timebin_samples)
        return electrode_data

    def forward(self, electrode_data, electrode_indices, normalization_means=None, normalization_stds=None):
        # electrode_data is of shape (batch_size, n_electrodes, n_samples) 
        #   where n_samples = sample_timebin_size * n_timebins 
        # electrode_indices is of shape (batch_size, *normalization_shape)
        
        if normalization_means is None: normalization_means = self.normalization_means[electrode_indices] # shape: (batch_size, n_electrodes, *normalization_shape)
        if normalization_stds is None: normalization_stds = self.normalization_stds[electrode_indices]
        electrode_embeddings = self.electrode_embedding_class.forward(electrode_indices) # shape: (batch_size, n_electrodes, d_model)
        batch_size, n_electrodes, d_model = electrode_embeddings.shape

        electrode_data = self.preprocess_electrode_data(electrode_data, self.sampling_rates[electrode_indices][0][0]) # XXX: Assuming all items in the batch have the sane sampling rate
        # shape: (batch_size, n_electrodes, n_timebins, d_timebin), by default d_timebin = sample_timebin_size

        electrode_data = electrode_data - normalization_means.view(batch_size, n_electrodes, 1, *self.normalization_shape)
        electrode_data = electrode_data / (normalization_stds.view(batch_size, n_electrodes, 1, *self.normalization_shape) + self.std_smoothing)

        electrode_data = self.linear_embed(electrode_data) # shape: (batch_size, n_electrodes, n_timebins, d_model)
        electrode_data = electrode_data + electrode_embeddings.view(batch_size, n_electrodes, 1, d_model)

        return electrode_data # shape: (batch_size, n_electrodes, n_timebins, d_model)

    def initialize_normalization(self, subject, session_id, init_normalization_window_to=2048 * 60 * 10):
        subject_identifier = subject.subject_identifier
        indices = subject.get_electrode_indices(session_id)
        electrode_labels = subject.get_electrode_labels()
        electrode_labels = [electrode_labels[i] for i in indices]
        electrode_indices = [self.electrode_embedding_class.embeddings_map[(subject.subject_identifier, electrode_label)] for electrode_label in electrode_labels]

        with torch.no_grad():
            all_electrode_data = subject.get_all_electrode_data(session_id, window_to=init_normalization_window_to).to(self.device, dtype=self.dtype)
            electrode_means, electrode_stds = self.calculate_electrode_normalization(all_electrode_data, self.sampling_rates[electrode_indices[0]]) # XXX: Assuming all electrodes have the same sampling rate

            for idx, electrode_index in enumerate(electrode_indices):
                self.normalization_means.data[electrode_index] = electrode_means[idx]
                self.normalization_stds.data[electrode_index] = electrode_stds[idx]

    def calculate_electrode_normalization(self, electrode_data, sampling_rate):
        """
            x is of shape (batch_size, n_electrodes, n_samples) or (n_electrodes, n_samples)
            where n_samples = sample_timebin_size * n_timebins 
                (will be trimmed to a multiple of timebin_samples)
        """
        if len(electrode_data.shape) == 2: # no batch dimension; add it
            electrode_data = electrode_data.unsqueeze(0)

        return electrode_data.mean(dim=(0, 2)).unsqueeze(1), electrode_data.std(dim=(0, 2)).unsqueeze(1)

class ElectrodeDataEmbeddingFFT(ElectrodeDataEmbedding):
    def __init__(self, electrode_embedding_class, sample_timebin_size, max_frequency_bin=64, normalization_requires_grad=True, std_smoothing=1e-5):
        super(ElectrodeDataEmbeddingFFT, self).__init__(electrode_embedding_class, sample_timebin_size, 
                                                        overall_sampling_rate=2048, normalization_requires_grad=normalization_requires_grad, 
                                                        std_smoothing=std_smoothing, normalization_shape=(max_frequency_bin,)) # XXX overall sampling rate doesnt matter, remove once fixed
        self.max_frequency_bin = max_frequency_bin
        self.linear_embed = nn.Linear(max_frequency_bin, self.d_model)

    def add_subject(self, subject, sampling_rate):
        super(ElectrodeDataEmbeddingFFT, self).add_subject(subject, sampling_rate)

    def preprocess_electrode_data(self, electrode_data, sampling_rate, allow_trim=False):
        # x is of shape (batch_size, n_electrodes, n_samples)
        #   where n_samples = sample_timebin_size * n_timebins
        electrode_data = super(ElectrodeDataEmbeddingFFT, self).preprocess_electrode_data(electrode_data, sampling_rate, allow_trim=allow_trim) # shape: (batch_size, n_electrodes, n_timebins, samples_per_bin)
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
        
        # Calculate magnitude (equivalent to scipy.signal.stft's magnitude)
        x = torch.abs(x)
        # Convert to power
        x = torch.log(x + 1e-5)
        return x.to(dtype=self.dtype)  # Convert back to original dtype; shape (batch_size, n_electrodes, n_timebins, max_frequency_bin)

    def calculate_electrode_normalization(self, electrode_data, sampling_rate):
        """
            x is of shape (batch_size, n_electrodes, n_samples) or (n_electrodes, n_samples)
            where n_samples = sample_timebin_size * n_timebins 
                (will be trimmed to a multiple of timebin_samples)
        """
        if len(electrode_data.shape) == 2: # no batch dimension; add it
            electrode_data = electrode_data.unsqueeze(0)
    
        electrode_data = self.preprocess_electrode_data(electrode_data, sampling_rate, allow_trim=True)
        means, stds = electrode_data.mean(dim=(0, 2)), electrode_data.std(dim=(0, 2))
        return means, stds


class ElectrodeEmbedding(BFModule):
    def __init__(self, d_model, embedding_dim=None, embedding_fanout_requires_grad=True, embedding_requires_grad=True, initial_capacity=100):
        super(ElectrodeEmbedding, self).__init__()

        self.embedding_dim = embedding_dim if embedding_dim is not None else d_model
        self.embedding_requires_grad = embedding_requires_grad
        if self.embedding_dim < d_model:
            self.linear_embed = nn.Linear(self.embedding_dim, self.d_model, requires_grad=embedding_fanout_requires_grad)
        else: 
            self.linear_embed = lambda x: x # just identity function if embedding dim is already at d_model
        
        self.d_model = d_model
        self._initialize_embeddings(initial_capacity)
        self.embeddings_map = {} # map from subject / electrode labels to embedding indices

        # For compatibility
        self.add_embedding = lambda subject, embedding_init: self.add_raw(subject.subject_identifier, subject.get_electrode_labels())
        self.add_subject = lambda subject: self.add_raw(subject.subject_identifier, subject.get_electrode_labels())

    def _get_current_capacity(self):
        return self.embeddings.weight.shape[0]
    def _get_current_size(self):
        return len(self.embeddings_map)
    def _initialize_embeddings(self, initial_capacity):
        self.embeddings = nn.Embedding(initial_capacity, self.d_model)
        self.embeddings.weight.data.zero_()
    def _ensure_capacity(self, needed_size):
        if needed_size >= self._get_current_capacity():
            new_capacity = needed_size #max(self._get_current_capacity() * 2, needed_size + 1)
            old_embeddings = self.embeddings.weight.data
            self._initialize_embeddings(new_capacity)
            self.embeddings.weight.data[:old_embeddings.shape[0]] = old_embeddings

    def forward(self, electrode_indices):
        # electrode_indices is a tensor of shape (batch_size, n_electrodes)
        return self.linear_embed(self.embeddings(electrode_indices))

    def add_raw(self, subject_identifier, electrode_labels):
        self._ensure_capacity(self._get_current_size() + len(electrode_labels))
        for electrode_label in electrode_labels:
            key = (subject_identifier, electrode_label)
            if key not in self.embeddings_map:
                self.embeddings_map[key] = self._get_current_size()

    def load_state_dict(self, state_dict, strict=True):
        # Load embeddings_map from state dict if it exists
        if 'embeddings_map' in state_dict:
            self.embeddings_map = state_dict['embeddings_map']
            del state_dict['embeddings_map'] # Remove embeddings_map from state dict since it's not a tensor parameter
        return super().load_state_dict(state_dict, strict=strict)

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        state_dict['embeddings_map'] = self.embeddings_map
        return state_dict
ElectrodeEmbedding_Learned = ElectrodeEmbedding

EEG_CHANNEL_NAME_MAPPING = {
    'T3': 'T7',
    'T4': 'T8',
    'T5': 'P7',
    'T6': 'P8'
}
class ElectrodeEmbedding_Learned_FixedVocabulary(ElectrodeEmbedding):
    def __init__(self, d_model, vocabulary_channels, embedding_dim=None, embedding_fanout_requires_grad=True, embedding_requires_grad=True):
        """
            vocabulary is a list of strings, each corresponding to each recorded channel name.
            All the channels that are not in the vocabulary will be set to an "overflow" vector that can be learned.
        """
        super(ElectrodeEmbedding_Learned_FixedVocabulary, self).__init__(d_model, initial_capacity=len(vocabulary_channels)+1, 
                                                                         embedding_dim=embedding_dim, 
                                                                         embedding_fanout_requires_grad=embedding_fanout_requires_grad, 
                                                                         embedding_requires_grad=embedding_requires_grad)
        self.vocabulary_channels = [EEG_CHANNEL_NAME_MAPPING.get(x.upper(), x.upper()) for x in vocabulary_channels]

    def add_raw(self, subject_identifier, channel_names):
        # New embeddings are never added, only the existing ones are mapped to the vocabulary
        # Non-existing channels are mapped to the last vector in the vocabulary (can be learned)
        for electrode_label in channel_names:
            mapped_label = EEG_CHANNEL_NAME_MAPPING.get(electrode_label.upper(), electrode_label.upper())
            if mapped_label in self.vocabulary_channels:
                self.embeddings_map[(subject_identifier, electrode_label)] = self.vocabulary_channels.index(mapped_label)
            else:
                self.embeddings_map[(subject_identifier, electrode_label)] = len(self.vocabulary_channels)
    

def coordinates_positional_encoding(electrode_coordinates, embedding_dim):
    """
        "Electrode Coordinates" must be LPI coordinates, normalized to [0,1] range given min/max
    """
    dim_freq = embedding_dim//6
    freq = 100.0 / (200 ** (torch.arange(0, dim_freq, dtype=electrode_coordinates.dtype)/dim_freq)) # coordinates are normalized in the add_embedding to lie between 0 and 1

    # Calculate position encodings for each coordinate dimension
    l_enc = electrode_coordinates[:, 0:1] @ freq.unsqueeze(0)  # For L coordinate
    i_enc = electrode_coordinates[:, 1:2] @ freq.unsqueeze(0)  # For I coordinate  
    p_enc = electrode_coordinates[:, 2:3] @ freq.unsqueeze(0)  # For P coordinate
    padding_zeros = torch.zeros(len(electrode_coordinates), embedding_dim % dim_freq, dtype=electrode_coordinates.dtype) # padding in case model dimension is not divisible by 6

    # Combine sin and cos encodings
    embedding = torch.cat([torch.sin(l_enc), torch.cos(l_enc), torch.sin(i_enc), torch.cos(i_enc), torch.sin(p_enc), torch.cos(p_enc), padding_zeros], dim=-1)
    return embedding

class ElectrodeEmbedding_Learned_CoordinateInit(ElectrodeEmbedding):
    """
        This class will initialize the embedding with the coordinates of the electrodes, and then (if requires_grad is True)
        they can behave just the same as the ElectrodeEmbedding_Learned
    """
    def __init__(self, d_model, embedding_dim=None, embedding_fanout_requires_grad=True, embedding_requires_grad=True):
        super(ElectrodeEmbedding_Learned_CoordinateInit, self).__init__(d_model, embedding_dim=embedding_dim, 
                                                             embedding_fanout_requires_grad=embedding_fanout_requires_grad,
                                                             embedding_requires_grad=embedding_requires_grad)
        self.add_embedding = self.add_subject # for backwards compatibility

    def add_subject(self, subject, coordinate_scale=150):
        super(ElectrodeEmbedding_Learned_CoordinateInit, self).add_subject(subject)
        electrode_coordinates = subject.get_electrode_coordinates().to(dtype=self.dtype) / coordinate_scale
        for i, electrode_label in enumerate(subject.get_electrode_labels()):
            embedding_index = self.embeddings_map[(subject.subject_identifier, electrode_label)]
            self.embeddings.weight.data[embedding_index] = coordinates_positional_encoding(electrode_coordinates[i:i+1], self.embedding_dim)


class ElectrodeEmbedding_NoisyCoordinate(ElectrodeEmbedding_Learned):
    """
        This class will create the embedding with the positional encoding of the coordinates of the electrodes, 
        and then in each forward pass add random noise to the coordinates. 
        These embeddings are not learnable apart from a fanout linear layer (in case embedding_dim<d_model).
    """
    def __init__(self, d_model, embedding_dim=None, embedding_fanout_requires_grad=True, coordinate_noise_std=0.01):
        super(ElectrodeEmbedding_NoisyCoordinate, self).__init__(3, embedding_dim=None, # Just store the L, I, P coordinates   
                                                             embedding_fanout_requires_grad=embedding_fanout_requires_grad,
                                                             embedding_requires_grad=False) # No need to learn the coordinates
        self.embedding_dim = d_model if embedding_dim is None else embedding_dim
        self.coordinate_noise_std = coordinate_noise_std
        self.add_embedding = self.add_subject # for backwards compatibility

    def add_subject(self, subject, coordinate_scale=150):
        super(ElectrodeEmbedding_NoisyCoordinate, self).add_subject(subject)
        electrode_coordinates = subject.get_electrode_coordinates().to(dtype=self.dtype) / coordinate_scale
        for i, electrode_label in enumerate(subject.get_electrode_labels()):
            embedding_index = self.embeddings_map[(subject.subject_identifier, electrode_label)]
            self.embeddings.weight.data[embedding_index] = electrode_coordinates[i:i+1] # Just store the L, I, P coordinates

    def forward(self, electrode_indices):
        electrode_coordinates = super().forward(electrode_indices)
        electrode_coordinates += torch.randn_like(electrode_coordinates) * self.coordinate_noise_std
        return coordinates_positional_encoding(electrode_coordinates, self.embedding_dim)
