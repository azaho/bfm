import torch

def get_all_laplacian_electrodes(electrode_labels):
    """
        Get all laplacian electrodes for a given subject. This function is originally from
        https://github.com/czlwang/BrainBERT repository (Wang et al., 2023)
    """
    def stem_electrode_name(name):
        #names look like 'O1aIb4', 'O1aIb5', 'O1aIb6', 'O1aIb7'
        #names look like 'T1b2
        found_stem_end = False
        stem, num = [], []
        for c in reversed(name):
            if c.isalpha():
                found_stem_end = True
            if found_stem_end:
                stem.append(c)
            else:
                num.append(c)
        return ''.join(reversed(stem)), int(''.join(reversed(num)))
    def has_neighbors(stem, stems):
        (x,y) = stem
        return ((x,y+1) in stems) or ((x,y-1) in stems)
    def get_neighbors(stem, stems):
        (x,y) = stem
        return [f'{x}{y}' for (x,y) in [(x,y+1), (x,y-1)] if (x, y) in stems]
    stems = [stem_electrode_name(e) for e in electrode_labels]
    laplacian_stems = [x for x in stems if has_neighbors(x, stems)]
    electrodes = [f'{x}{y}' for (x,y) in laplacian_stems]
    neighbors = {e: get_neighbors(stem_electrode_name(e), stems) for e in electrodes}
    return electrodes, neighbors


def laplacian_rereference_neural_data(electrode_data, electrode_labels, remove_non_laplacian=True):
    """
    Rereference the neural data using the laplacian method (subtract the mean of the neighbors, as determined by the electrode labels)
    inputs:
        electrode_data: torch tensor of shape (batch_size, n_electrodes, n_samples) or (n_electrodes, n_samples)
        electrode_labels: list of electrode labels
        remove_non_laplacian: boolean, if True, remove the non-laplacian electrodes from the data; if false, keep them without rereferencing
    outputs:
        rereferenced_data: torch tensor of shape (batch_size, n_electrodes_rereferenced, n_samples) or (n_electrodes_rereferenced, n_samples)
        rereferenced_labels: list of electrode labels of length n_electrodes_rereferenced (n_electrodes_rereferenced could be different from n_electrodes if remove_non_laplacian is True)
    """
    batch_unsqueeze = False
    if len(electrode_data.shape) == 2:
        batch_unsqueeze = True
        electrode_data = electrode_data.unsqueeze(0)

    laplacian_electrodes, laplacian_neighbors = get_all_laplacian_electrodes(electrode_labels)
    laplacian_neighbor_indices = {laplacian_electrode_label: [electrode_labels.index(neighbor_label) for neighbor_label in neighbors] for laplacian_electrode_label, neighbors in laplacian_neighbors.items()}

    batch_size, n_electrodes, n_samples = electrode_data.shape
    rereferenced_n_electrodes = len(laplacian_electrodes) if remove_non_laplacian else n_electrodes
    rereferenced_data = torch.zeros((batch_size, rereferenced_n_electrodes, n_samples), dtype=electrode_data.dtype, device=electrode_data.device)

    electrode_i = 0
    original_electrode_indices = []
    for original_electrode_index, electrode_label in enumerate(electrode_labels):
        if electrode_label in laplacian_electrodes:
            rereferenced_data[:, electrode_i] = electrode_data[:, electrode_i] - torch.mean(electrode_data[:, laplacian_neighbor_indices[electrode_label]], axis=1)
            original_electrode_indices.append(original_electrode_index)
            electrode_i += 1
        else:
            if remove_non_laplacian: 
                continue # just skip the non-laplacian electrodes
            else:
                rereferenced_data[:, electrode_i] = electrode_data[:, electrode_i]
                original_electrode_indices.append(original_electrode_index)
                electrode_i += 1
                
    if batch_unsqueeze:
        rereferenced_data = rereferenced_data.squeeze(0)
    return rereferenced_data, laplacian_electrodes if remove_non_laplacian else electrode_labels, original_electrode_indices


def laplacian_rereference_batch(batch, remove_non_laplacian=True, inplace=False):
    """
    Rereference the neural data using the laplacian method (subtract the mean of the neighbors, as determined by the electrode labels)
    inputs:
        batch: dictionary containing the following keys:
            'data': torch tensor of shape (batch_size, n_electrodes, n_samples) or (n_electrodes, n_samples)
            'electrode_label': [lists of electrode labels] of length batch_size
            'electrode_index': [lists of electrode indices] of length batch_size
    outputs:
        batch: dictionary containing the following keys:
            'data': torch tensor of shape (batch_size, n_electrodes_rereferenced, n_samples) or (n_electrodes_rereferenced, n_samples)
            'electrode_labels': lists of electrode labels of length n_electrodes_rereferenced (n_electrodes_rereferenced could be different from n_electrodes if remove_non_laplacian is True)
            'electrode_index': lists of electrode indices of length n_electrodes_rereferenced (n_electrodes_rereferenced could be different from n_electrodes if remove_non_laplacian is True)
    """
    assert inplace, "laplacian_rereference_batch currently only supports inplace=True"

    electrode_data = batch['data'] # shape: (batch_size, n_electrodes, n_samples)
    electrode_labels = batch['electrode_labels'] # shape: (batch_size, n_electrodes)

    rereferenced_data, rereferenced_labels, original_electrode_indices = laplacian_rereference_neural_data(electrode_data, electrode_labels[0], remove_non_laplacian=remove_non_laplacian)
    # XXX The line above assumes that the electrode labels are the same for all subjects in the batch. This may or may not be the case. Need to find a more efficient way later to do it item per item.

    batch['data'] = rereferenced_data
    batch['electrode_labels'] = [rereferenced_labels] * batch['data'].shape[0]

    if 'electrode_index' in batch:
        batch['electrode_index'] = batch['electrode_index'][:, original_electrode_indices]

    return batch


if __name__ == "__main__":
    from subject.braintreebank import BrainTreebankSubject
    subject = BrainTreebankSubject(1, cache=False)
    print(f"Subject electrode labels (length: {len(subject.get_electrode_labels())}): ", subject.get_electrode_labels())

    electrode_data = subject.get_all_electrode_data(1, window_from=100, window_to=200)
    rereferenced_data, rereferenced_labels = laplacian_rereference_neural_data(electrode_data, subject.electrode_labels)
    print(f"Rereferenced data (length: {len(rereferenced_labels)}): ", rereferenced_labels)
    print(f"Rereferenced data (shape: {rereferenced_data.shape}): ", rereferenced_data)