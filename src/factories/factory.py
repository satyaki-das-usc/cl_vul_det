from abc import ABC, abstractmethod

class Factory(ABC):
    def __init__(self, instance_name: str):
        self.instance_name = instance_name

    @abstractmethod
    def create_product(self):
        pass