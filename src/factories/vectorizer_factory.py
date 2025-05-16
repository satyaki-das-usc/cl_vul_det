from src.factories.factory import Factory

class VectorizerFactory(Factory):
    def create_product(self):
        return f"{self.instance_name} Vectorizer"