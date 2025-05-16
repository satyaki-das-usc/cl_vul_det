from src.factories.factory import Factory

class PreprocessingFactory(Factory):
    def create_product(self):
        return f"{self.instance_name} Preprocessing"