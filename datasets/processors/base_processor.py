from abc import ABC
from abc import abstractmethod


class BaseDatasetProcessor(ABC):

    @abstractmethod
    def process_subject(
        self,
        subject_id: int,
    ):
        pass