from src.factories.factory import Factory

class DistanceMetricFactory(Factory):
    def create_product(self):
        return f"{self.instance_name} Distance Metric"