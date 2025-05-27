from gensim.models import Word2Vec

from src.factories.factory import Factory

class VectorizerFactory(Factory):
    def create_product(self, sentences, vector_size, max_vocab_size, num_workers):
        if self.instance_name == "word2vec":
            return Word2Vec(sentences=sentences, min_count=3, vector_size=vector_size, window=5,
                    max_vocab_size=max_vocab_size, workers=num_workers, sg=1)
        return f"{self.instance_name} Vectorizer"