from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import nltk
import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('punkt_tab')

class EmbeddingsGenerator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = "anferico/bert-for-patents"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.max_length = 500

    def split_text_by_sentences(self, text):
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = ''
        for sentence in sentences:
            if len(self.tokenizer.encode(current_chunk + ' ' + sentence, add_special_tokens=False)) <= self.max_length:
                current_chunk += ' ' + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def get_embeddings_bfp(self, texts):
        all_embeddings = []
        all_segments = []
        segments_per_text = []
        
        # Dividir textos en segmentos
        for text in texts:
            segments = self.split_text_by_sentences(text)
            all_segments.extend(segments)
            segments_per_text.append(len(segments))
        
        # Generar embeddings
        segment_embeddings = []
        batch_size = 256
        for i in range(0, len(all_segments), batch_size):
            batch = all_segments[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            segment_embeddings.extend(embeddings)
        
        # Combinar embeddings
        idx = 0
        for num_segments in segments_per_text:
            text_embeddings = segment_embeddings[idx:idx + num_segments]
            text_embedding = np.mean(text_embeddings, axis=0)
            all_embeddings.append(text_embedding.tolist())  # Convertir a lista para serialización JSON
            idx += num_segments
            
        return all_embeddings

    def process_patent_data(self, patent_data):
        # Extraer textos principales y citados
        main_patent_id = next(key for key in patent_data.keys() if key != 'cited_document_id')
        main_text = patent_data[main_patent_id]
        cited_texts = list(patent_data['cited_document_id'].values())
        
        # Generar embeddings
        main_embedding = self.get_embeddings_bfp([main_text])[0]
        cited_embeddings = self.get_embeddings_bfp(cited_texts)
        
        # Preparar resultado
        result = {
            'main_patent': {
                'id': main_patent_id,
                'embedding': main_embedding
            },
            'cited_patents': [
                {
                    'id': patent_id,
                    'embedding': embedding
                }
                for patent_id, embedding in zip(patent_data['cited_document_id'].keys(), cited_embeddings)
            ]
        }
        
        return result