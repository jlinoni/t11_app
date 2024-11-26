import torch
import numpy as np
import nltk
import json
import hashlib
from pathlib import Path
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import traceback
from datetime import datetime
import time
import os
import shutil


class EmbeddingsGenerator:
    def __init__(self):
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Usando dispositivo: {self.device}")
            self.model_name = "anferico/bert-for-patents"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.max_length = 500
        except Exception as e:
            print(f"Error inicializando EmbeddingsGenerator: {str(e)}")
            raise

    def split_text_by_sentences(self, text):
        try:
            if not isinstance(text, str):
                raise ValueError(f"El texto debe ser una cadena, no {type(text)}")
            
            sentences = nltk.sent_tokenize(text)
            chunks = []
            current_chunk = ''
            
            for sentence in sentences:
                if len(self.tokenizer.encode(current_chunk + ' ' + sentence, add_special_tokens=False)) <= self.max_length:
                    current_chunk += ' ' + sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                    
            if current_chunk:
                chunks.append(current_chunk.strip())
                
            return chunks
        except Exception as e:
            print(f"Error en split_text_by_sentences: {str(e)}")
            raise

    def get_embeddings_bfp(self, texts):
        try:
            if not texts:
                raise ValueError("La lista de textos está vacía")

            all_embeddings = []
            all_segments = []
            segments_per_text = []
            
            # Dividir textos en segmentos
            for text in texts:
                segments = self.split_text_by_sentences(text)
                if not segments:
                    raise ValueError(f"No se pudieron extraer segmentos del texto: {text[:100]}...")
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
                all_embeddings.append(text_embedding.tolist())
                idx += num_segments
                
            return all_embeddings
        except Exception as e:
            print(f"Error en get_embeddings_bfp: {str(e)}")
            print(traceback.format_exc())
            raise

class EmbeddingsProcessor:
    def __init__(self, cache_dir="data/embeddings_cache"):
        try:
            # Crear un directorio de caché específico para la sesión
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.cache_dir = Path(cache_dir) / self.session_id
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Limpiar cachés antiguos
            self.clean_old_cache_dirs(Path(cache_dir))
            
            self.embeddings_generator = EmbeddingsGenerator()
            # Descargar recursos de NLTK
            nltk.download('punkt', quiet=True)
            print(f"Nueva sesión iniciada: {self.session_id}")
        except Exception as e:
            print(f"Error inicializando EmbeddingsProcessor: {str(e)}")
            raise

    def clean_old_cache_dirs(self, base_cache_dir, max_dirs=5):
        """Limpia directorios de caché antiguos, manteniendo solo los más recientes."""
        try:
            if not base_cache_dir.exists():
                return
                
            # Listar todos los directorios de sesión
            session_dirs = [d for d in base_cache_dir.iterdir() if d.is_dir()]
            
            # Ordenar por fecha de modificación (más antiguo primero)
            session_dirs.sort(key=lambda x: x.stat().st_mtime)
            
            # Eliminar directorios antiguos si hay más que el máximo permitido
            while len(session_dirs) >= max_dirs:
                oldest_dir = session_dirs.pop(0)
                try:
                    shutil.rmtree(oldest_dir)
                    print(f"Caché antiguo eliminado: {oldest_dir}")
                except Exception as e:
                    print(f"Error al eliminar directorio antiguo {oldest_dir}: {e}")

        except Exception as e:
            print(f"Error limpiando cachés antiguos: {e}")

    def generate_cache_key(self, patent_data):
        try:
            main_key = next(key for key in patent_data.keys() if key != 'cited_document_id')
            content = str(patent_data[main_key])
            for _, text in sorted(patent_data['cited_document_id'].items()):
                content += str(text)
            
            # Incluir el ID de sesión en la clave del caché
            session_content = f"{self.session_id}_{content}"
            return hashlib.sha256(session_content.encode()).hexdigest()
        except Exception as e:
            print(f"Error generando cache key: {str(e)}")
            raise

    def reduce_dimensionality(self, embeddings_list):
        try:
            if not embeddings_list:
                raise ValueError("Lista de embeddings vacía")
            
            all_embeddings = np.array(embeddings_list)
            n_samples = len(all_embeddings)
            
            if n_samples <= 2:
                print("Muy pocas muestras para t-SNE, retornando coordenadas aleatorias")
                return np.random.rand(n_samples, 3).tolist()
            else:
                perplexity = min(max(n_samples // 3, 2), 30)
                perplexity = min(perplexity, n_samples - 1)
                
                print(f"Usando perplejidad de {perplexity} para {n_samples} muestras")
                tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
                reduced_embeddings = tsne.fit_transform(all_embeddings)
                return reduced_embeddings.tolist()
                
        except Exception as e:
            print(f"Error en reduce_dimensionality: {str(e)}")
            raise

    def process_embeddings(self, embeddings_data):
        """Procesa los embeddings para incluir la reducción de dimensionalidad."""
        try:
            all_embeddings = [embeddings_data['main_patent']['embedding']]
            for patent in embeddings_data['cited_patents']:
                all_embeddings.append(patent['embedding'])
            
            reduced_embeddings = self.reduce_dimensionality(all_embeddings)
            
            embeddings_data['main_patent']['reduced_embedding'] = reduced_embeddings[0]
            for i, patent in enumerate(embeddings_data['cited_patents']):
                patent['reduced_embedding'] = reduced_embeddings[i + 1]
            
            return embeddings_data
        except Exception as e:
            print(f"Error en process_embeddings: {str(e)}")
            raise

    def process_patent_data(self, patent_data):
        """Procesa los datos de la patente, incluyendo embeddings y reducción."""
        try:
            if not isinstance(patent_data, dict):
                raise ValueError(f"patent_data debe ser un diccionario, no {type(patent_data)}")
            if 'cited_document_id' not in patent_data:
                raise ValueError("patent_data debe contener la clave 'cited_document_id'")
            
            cache_key = self.generate_cache_key(patent_data)
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    print(f"Datos recuperados de caché de sesión: {self.session_id}")
                    return {"embeddings": cached_data, "from_cache": True}

            print(f"Generando nuevos embeddings para sesión: {self.session_id}")
            
            main_patent_id = next(key for key in patent_data.keys() if key != 'cited_document_id')
            main_text = patent_data[main_patent_id]
            cited_texts = list(patent_data['cited_document_id'].values())
            
            main_embedding = self.embeddings_generator.get_embeddings_bfp([main_text])[0]
            cited_embeddings = self.embeddings_generator.get_embeddings_bfp(cited_texts)
            
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
            
            result_with_reduction = self.process_embeddings(result)
            
            with open(cache_file, 'w') as f:
                json.dump(result_with_reduction, f)
            
            print(f"Nuevos embeddings generados y guardados en caché de sesión: {self.session_id}")
            return {"embeddings": result_with_reduction, "from_cache": False}
        
        except Exception as e:
            print(f"Error en process_patent_data: {str(e)}")
            print(traceback.format_exc())
            raise

'''
class EmbeddingsProcessor:
    def __init__(self, cache_dir="data/embeddings_cache"):
        try:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # No inicializamos TSNE aquí, lo haremos dinámicamente
            self.embeddings_generator = EmbeddingsGenerator()
            # Descargar recursos de NLTK
            nltk.download('punkt', quiet=True)
        except Exception as e:
            print(f"Error inicializando EmbeddingsProcessor: {str(e)}")
            raise

    def generate_cache_key(self, patent_data):
        try:
            main_key = next(key for key in patent_data.keys() if key != 'cited_document_id')
            content = str(patent_data[main_key])
            for _, text in sorted(patent_data['cited_document_id'].items()):
                content += str(text)
            return hashlib.sha256(content.encode()).hexdigest()
        except Exception as e:
            print(f"Error generando cache key: {str(e)}")
            raise

    def reduce_dimensionality(self, embeddings_list):
        try:
            if not embeddings_list:
                raise ValueError("Lista de embeddings vacía")
            
            all_embeddings = np.array(embeddings_list)
            n_samples = len(all_embeddings)
            
            # Ajustar la perplejidad basada en el número de muestras
            if n_samples <= 2:
                # Para 1 o 2 muestras, no podemos usar t-SNE
                print("Muy pocas muestras para t-SNE, retornando coordenadas aleatorias")
                return np.random.rand(n_samples, 3).tolist()
            else:
                # Usar una perplejidad que sea aproximadamente 1/3 del número de muestras
                # pero no mayor a 30 y no menor a 2
                perplexity = min(max(n_samples // 3, 2), 30)
                perplexity = min(perplexity, n_samples - 1)  # Asegurar que sea menor que n_samples
                
                print(f"Usando perplejidad de {perplexity} para {n_samples} muestras")
                tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
                reduced_embeddings = tsne.fit_transform(all_embeddings)
                return reduced_embeddings.tolist()
                
        except Exception as e:
            print(f"Error en reduce_dimensionality: {str(e)}")
            raise

    def process_embeddings(self, embeddings_data):
        """Procesa los embeddings para incluir la reducción de dimensionalidad."""
        try:
            all_embeddings = [embeddings_data['main_patent']['embedding']]
            for patent in embeddings_data['cited_patents']:
                all_embeddings.append(patent['embedding'])
            
            reduced_embeddings = self.reduce_dimensionality(all_embeddings)
            
            embeddings_data['main_patent']['reduced_embedding'] = reduced_embeddings[0]
            for i, patent in enumerate(embeddings_data['cited_patents']):
                patent['reduced_embedding'] = reduced_embeddings[i + 1]
            
            return embeddings_data
        except Exception as e:
            print(f"Error en process_embeddings: {str(e)}")
            raise

    def process_patent_data(self, patent_data):
        """Procesa los datos de la patente, incluyendo embeddings y reducción."""
        try:
            if not isinstance(patent_data, dict):
                raise ValueError(f"patent_data debe ser un diccionario, no {type(patent_data)}")
            if 'cited_document_id' not in patent_data:
                raise ValueError("patent_data debe contener la clave 'cited_document_id'")
            
            cache_key = self.generate_cache_key(patent_data)
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    print("Datos recuperados de caché")
                    return {"embeddings": cached_data, "from_cache": True}

            print("Generando nuevos embeddings...")
            
            main_patent_id = next(key for key in patent_data.keys() if key != 'cited_document_id')
            main_text = patent_data[main_patent_id]
            cited_texts = list(patent_data['cited_document_id'].values())
            
            main_embedding = self.embeddings_generator.get_embeddings_bfp([main_text])[0]
            cited_embeddings = self.embeddings_generator.get_embeddings_bfp(cited_texts)
            
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
            
            result_with_reduction = self.process_embeddings(result)
            
            with open(cache_file, 'w') as f:
                json.dump(result_with_reduction, f)
            
            print("Nuevos embeddings generados y guardados en caché")
            return {"embeddings": result_with_reduction, "from_cache": False}
        
        except Exception as e:
            print(f"Error en process_patent_data: {str(e)}")
            print(traceback.format_exc())
            raise
'''            