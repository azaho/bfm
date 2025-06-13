class SubjectInterface:
    """
    Abstract base class defining the interface for subject neural data access.
    All subject-specific implementations (MGH, BrainTreebank, etc.) should inherit from this.

    The interface provides standardized methods to:
    - Access electrode metadata (labels, coordinates, etc.)
    - Load and retrieve neural data for specific electrodes/trials/sessions
    - Manage data caching
    """

    def get_n_electrodes(self, session_id=None):
        """
        Get number of electrodes for this subject.

        Args:
            session_id: Optional session ID if electrode count varies by session

        Returns:
            int: Number of electrodes
        """
        raise NotImplementedError

    def get_electrode_labels(self, session_id=None):
        """
        Get list of electrode labels.

        Args:
            session_id: Optional session ID if labels vary by session

        Returns:
            list: List of electrode label strings
        """
        raise NotImplementedError

    def get_electrode_indices(self, session_id=None):
        """
        Get array of electrode indices.

        Args:
            session_id: Optional session ID if indices vary by session

        Returns:
            numpy.ndarray: Array of electrode indices
        """
        raise NotImplementedError

    def get_sampling_rate(self, session_id=None):
        """
        Get sampling rate in Hz.

        Args:
            session_id: Optional session ID if sampling rate varies by session

        Returns:
            float: Sampling rate in Hz
        """
        raise NotImplementedError

    def get_electrode_coordinates(self, session_id=None):
        """
        Get electrode coordinates in standardized space.

        Args:
            session_id: Optional session ID if coordinates vary by session

        Returns:
            torch.Tensor: (n_electrodes, 3) tensor of coordinates
        """
        raise NotImplementedError

    def load_neural_data(self, trial_id):
        """
        Load neural data for a specific trial/session.
        Implementation should handle caching if enabled.

        Args:
            trial_id: Trial/session identifier
        """
        raise NotImplementedError

    def get_electrode_data(self, electrode_label, trial_id, window_from=None, window_to=None):
        """
        Get data for a specific electrode and time window.

        Args:
            electrode_label: Label of electrode to get data for
            trial_id: Trial/session identifier
            window_from: Start sample index (optional)
            window_to: End sample index (optional)

        Returns:
            torch.Tensor: Neural data for specified electrode and window
        """
        raise NotImplementedError

    def get_all_electrode_data(self, trial_id, window_from=None, window_to=None):
        """
        Get data for all electrodes in a time window.

        Args:
            trial_id: Trial/session identifier
            window_from: Start sample index (optional)
            window_to: End sample index (optional)

        Returns:
            torch.Tensor: Neural data for all electrodes in window
        """
        raise NotImplementedError

    def clear_neural_data_cache(self, trial_id=None):
        """
        Clear cached neural data.

        Args:
            trial_id: Optional specific trial/session to clear, otherwise clears all
        """
        raise NotImplementedError
