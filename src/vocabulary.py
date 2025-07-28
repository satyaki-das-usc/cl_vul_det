from os.path import exists
import pickle
from typing import Dict, List
from dataclasses import dataclass
from gensim.models import KeyedVectors

PAD = "<PAD>"
UNK = "<UNK>"
MASK = "<MASK>"
BOS = "<BOS>"
EOS = "<EOS>"

dict_key = "token_to_id"

@dataclass
class Vocabulary:
    token_to_id: Dict[str, int]

    @staticmethod
    def from_w2v(w2v_path: str, speicial_tokens: List[str] = [PAD, UNK, MASK]):
        """
        Load vocabulary from word2vec model.
        :param w2v_path: path to the word2vec model
        :param speicial_tokens: list of special tokens to add to the vocabulary
        :return: Vocabulary object
        """
        assert exists(w2v_path), f"{w2v_path} not exists!"
        vectorizer = KeyedVectors.load(w2v_path, mmap="r")
        vocab = {token: i for i, token in enumerate(vectorizer.index_to_key)}
        for token in speicial_tokens:
            vocab[token] = len(vocab)
        return Vocabulary(token_to_id=vocab)
    
    @staticmethod
    def load_vocabulary(vocab_path: str):
        """
        Load vocabulary from a file.
        :param vocab_path: path to the vocabulary file
        :param speicial_tokens: list of special tokens to add to the vocabulary
        :return: Vocabulary object
        """
        assert exists(vocab_path), f"{vocab_path} not exists!"
        with open(vocab_path, "rb") as rbfi:
            vocab = pickle.load(rbfi)
        return Vocabulary(token_to_id=vocab[dict_key])
    
    def save_vocabulary(self, vocab_path: str):
        """
        Save vocabulary to a file.
        :param vocab_path: path to the vocabulary file
        :return: None
        """
        with open(vocab_path, "wb") as wbfi:
            pickle.dump({dict_key: self.token_to_id}, wbfi)
        
    def get_vocab_size(self):
        """
        Get the size of the vocabulary.
        :return: size of the vocabulary
        """
        return len(self.token_to_id)
    
    def get_id(self, token: str):
        """
        Get the id of a token.
        :param token: token to get the id for
        :return: id of the token
        """
        if token not in self.token_to_id:
            return self.token_to_id[UNK]
        return self.token_to_id[token]
    
    def get_pad_id(self):
        """
        Get the id of the padding token.
        :return: id of the padding token
        """
        return self.token_to_id[PAD]
    
    def get_unk_id(self):
        """
        Get the id of the unknown token.
        :return: id of the unknown token
        """
        return self.token_to_id[UNK]
    
    def convert_tokens_to_ids(self, tokens: List[str]):
        """
        Convert a list of tokens to a list of ids.
        :param tokens: list of tokens to convert
        :return: list of ids
        """
        return [self.get_id(token) for token in tokens]