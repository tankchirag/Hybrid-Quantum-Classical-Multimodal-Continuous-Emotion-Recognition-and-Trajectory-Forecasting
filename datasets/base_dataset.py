"""
Base dataset class for multimodal continuous emotion recognition
and trajectory forecasting.

This class defines the required interface for all datasets used in the
thesis pipeline (RECOLA, SEWA, synthetic). Every dataset must return:

    audio_emb:  (T, 128)
    video_emb:  (T, 128)
    ecg_emb:    (T, 128)
    eda_emb:    (T, 128)
    va_hist:    (T, 2)
    va_now:     (2,)
    va_fut:     (S, 2)

Where:
    T = history window length
    S = forecasting horizon
"""

from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseMultimodalDataset(Dataset, ABC):
    """Abstract base class for all multimodal datasets."""

    def __init__(self, T: int, S: int):
        """
        Parameters
        ----------
        T : int
            Length of the history window.
        S : int
            Forecasting horizon (number of future VA steps).
        """
        super().__init__()
        self.T = T
        self.S = S

    @abstractmethod
    def __len__(self):
        """Return number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """
        Return a dictionary containing all modalities and labels.

        Returns
        -------
        dict
            {
                "audio":  (T,128),
                "video":  (T,128),
                "ecg":    (T,128),
                "eda":    (T,128),
                "va_hist":(T,2),
                "va_now": (2,),
                "va_fut": (S,2)
            }
        """
        pass
