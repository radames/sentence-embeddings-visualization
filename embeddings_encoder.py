# from https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


class EmbeddingsEncoder:
    def __init__(self):
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained(
            'sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained(
            'sentence-transformers/all-MiniLM-L6-v2')

    # Mean Pooling - Take average of all tokens

    def mean_pooling(self, model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Encode text

    def encode(self,  texts):
        # Tokenize sentences
        print("Tokenizing...")
        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        print("Computing embeddings...")
        with torch.no_grad():
            model_output = self.model(**encoded_input, return_dict=True)

        # Perform pooling
        print("Performing pooling...")
        embeddings = self.mean_pooling(
            model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        print("Normalizing embeddings...")
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings
