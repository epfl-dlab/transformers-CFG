import json
import warnings
from pprint import pprint
from collections import defaultdict
from typing import List

from transformers import AutoTokenizer


def get_tokenizer_class(tokenizer):
    return f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}"


def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


class TokenizerDict:
    def __init__(self, tokenizer_ids: List[str]):
        self.unique_tokenizers = {}  # Stores tokenizer hash: tokenizer object
        self.hashes = set()  # Keeps track of existing hashes for quick lookup
        self.tokenizers = {}  # Stores hash: list of tokenizer objects
        self.models_by_tokenizer_class = {}
        self.models_by_hash = {}

        if tokenizer_ids:
            for tokenizer_id in tokenizer_ids:
                if type(tokenizer_id) == str:
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
                    except Exception as e:
                        # print warning
                        warnings.warn(f"Can't load tokenizer for '{tokenizer_id}': {e}")
                        continue
                else:
                    raise ValueError(
                        f"Tokenizer id should be a string, but got {type(tokenizer_id)}"
                    )
                self.add_tokenizer(tokenizer, model_id=tokenizer_id)

    def hash_tokenizer(self, tokenizer):
        # Your hash_tokenizer function implementation here
        """
        Hash the tokenizer based on the vocabulary.
        Two tokenizers with the same vocabulary will have the same hash as they represent the same model.
        If a tokenizer is slightly different from another, such as adding a new token, the hash will be different.
        This makes sense as the two tokenizers represent different models.
        c.f. https://stackoverflow.com/a/5884123/12234753
        """
        if type(tokenizer) == str:
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            except OSError:
                # print warning
                warnings.warn(f"Can't load tokenizer for '{tokenizer}'")
                return None
        vocab = tokenizer.get_vocab()
        frozen_vocab = frozenset(vocab.items())
        return hash(frozen_vocab)

    def add_tokenizer(self, tokenizer, model_id):
        hash_value = self.hash_tokenizer(tokenizer)
        if hash_value in self.hashes:
            print(f"Tokenizer already exists in the collection. Skipping.")
            self.tokenizers[hash_value].append(tokenizer)
            self.models_by_hash[hash_value].append(model_id)

        else:
            self.unique_tokenizers[hash_value] = tokenizer
            self.hashes.add(hash_value)
            self.models_by_hash[hash_value] = [model_id]
            self.tokenizers[hash_value] = [tokenizer]
            print(f"Tokenizer added to the collection.")
        self.tokenizers[hash_value].append(tokenizer)
        tokenizer_cls = get_tokenizer_class(tokenizer)
        if tokenizer_cls not in self.models_by_tokenizer_class:
            self.models_by_tokenizer_class[tokenizer_cls] = [model_id]
        else:
            self.models_by_tokenizer_class[tokenizer_cls].append(model_id)

    def get_tokenizer_by_hash(self, hash_value):
        return self.unique_tokenizers.get(hash_value, None)

    def is_duplicate(self, tokenizer):
        hash_value = self.hash_tokenizer(tokenizer)
        return hash_value in self.hashes


if __name__ == "__main__":

    most_popular_models: dict = read_json("HF_top_100_TGI_model_2024_01_25.json")

    tokenizers = [model_info["id"] for model_info in most_popular_models]

    td = TokenizerDict(tokenizers)

    pprint(td.models_by_tokenizer_class)

    # output to yaml
    import yaml

    with open("../docs/supported_models.yaml", "w") as file:
        documents = yaml.dump(
            td.models_by_tokenizer_class,
            file,
            indent=4,
            sort_keys=True,
            default_flow_style=False,
        )
