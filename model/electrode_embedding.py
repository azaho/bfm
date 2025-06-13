import torch
import torch.nn as nn
from model.transformer_implementation import BFModule

class ElectrodeEmbedding(BFModule):
    def __init__(self, d_model, embedding_dim=None, embedding_fanout_requires_grad=True, embedding_requires_grad=True, initial_capacity=100, **kwargs):
        super(ElectrodeEmbedding, self).__init__()

        self.embedding_dim = embedding_dim if embedding_dim is not None else d_model
        self.embedding_requires_grad = embedding_requires_grad
        if self.embedding_dim < d_model:
            self.embed_transformation = nn.Linear(self.embedding_dim, self.d_model, requires_grad=embedding_fanout_requires_grad)
        else: 
            self.embed_transformation = lambda x: x # just identity function if embedding dim is already at d_model
        
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
        self.embeddings.weight.data.zero_() # XXX do not zero out the embeddings
    def _ensure_capacity(self, needed_size):
        if needed_size >= self._get_current_capacity():
            new_capacity = needed_size #max(self._get_current_capacity() * 2, needed_size + 1)
            old_embeddings = self.embeddings.weight.data
            self._initialize_embeddings(new_capacity)
            self.embeddings.weight.data[:old_embeddings.shape[0]] = old_embeddings

    def forward(self, electrode_indices):
        # electrode_indices is a tensor of shape (batch_size, n_electrodes)
        return self.embed_transformation(self.embeddings(electrode_indices))

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

class ElectrodeEmbedding_Zero(ElectrodeEmbedding):
    def __init__(self, d_model, embedding_dim=None, **kwargs):
        super(ElectrodeEmbedding_Zero, self).__init__(d_model, embedding_dim=embedding_dim, embedding_fanout_requires_grad=False, embedding_requires_grad=False)
        self.embeddings.weight.data.zero_()
        self.embeddings.weight.requires_grad = False

EEG_CHANNEL_NAME_MAPPING = {
    'T3': 'T7',
    'T4': 'T8',
    'T5': 'P7',
    'T6': 'P8'
}
class ElectrodeEmbedding_Learned_FixedVocabulary(ElectrodeEmbedding):
    def __init__(self, d_model, vocabulary_channels, embedding_dim=None, embedding_fanout_requires_grad=True, embedding_requires_grad=True, **kwargs):
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
