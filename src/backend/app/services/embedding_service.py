"""
Embedding service for generating and managing document embeddings.
"""
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

from sentence_transformers import SentenceTransformer
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

from app.core.config import settings
from app.core.exceptions import EmbeddingException
from app.services.redis_client import RedisClient
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Service for generating and managing embeddings."""
    
    def __init__(self, redis_client: Optional[RedisClient] = None):
        self.redis_client = redis_client or RedisClient()
        self.embedding_model = settings.EMBEDDING_MODEL
        self.embedding_dimension = settings.EMBEDDING_DIMENSION
        
        # Initialize embedding model
        self._model = None
        self._langchain_embeddings = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize embedding models."""
        try:
            if settings.OPENAI_API_KEY:
                # Use OpenAI embeddings if API key is available
                self._langchain_embeddings = OpenAIEmbeddings(
                    openai_api_key=settings.OPENAI_API_KEY,
                    model="text-embedding-ada-002"
                )
                logger.info("Initialized OpenAI embeddings")
            else:
                # Use Hugging Face embeddings as fallback
                self._langchain_embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info(f"Initialized Hugging Face embeddings: {self.embedding_model}")
            
            # Initialize SentenceTransformer for direct usage
            self._model = SentenceTransformer(self.embedding_model)
            logger.info("Embedding service initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing embedding models: {e}")
            raise EmbeddingException(f"Failed to initialize embedding models: {str(e)}")
    
    def get_embeddings_model(self):
        """Get LangChain embeddings model."""
        return self._langchain_embeddings
    
    async def generate_embeddings(
        self,
        texts: List[str],
        use_cache: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            use_cache: Whether to use caching
            
        Returns:
            List[List[float]]: List of embeddings
        """
        try:
            embeddings = []
            texts_to_embed = []
            cache_keys = []
            
            # Check cache for existing embeddings
            if use_cache:
                for text in texts:
                    content_hash = hashlib.md5(text.encode()).hexdigest()
                    cache_key = f"embedding:{content_hash}"
                    
                    cached_embedding = await self.redis_client.get_cached_embeddings(content_hash)
                    
                    if cached_embedding:
                        embeddings.append(cached_embedding)
                        cache_keys.append(None)
                    else:
                        embeddings.append(None)
                        texts_to_embed.append(text)
                        cache_keys.append(cache_key)
            else:
                texts_to_embed = texts
                cache_keys = [None] * len(texts)
                embeddings = [None] * len(texts)
            
            # Generate embeddings for uncached texts
            if texts_to_embed:
                new_embeddings = await self._generate_embeddings_batch(texts_to_embed)
                
                # Cache new embeddings and fill results
                embed_idx = 0
                for i, embedding in enumerate(embeddings):
                    if embedding is None:
                        new_embedding = new_embeddings[embed_idx]
                        embeddings[i] = new_embedding
                        
                        # Cache the embedding
                        if use_cache and cache_keys[i]:
                            content_hash = hashlib.md5(texts[i].encode()).hexdigest()
                            await self.redis_client.cache_embeddings(
                                content_hash, new_embedding
                            )
                        
                        embed_idx += 1
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise EmbeddingException(f"Failed to generate embeddings: {str(e)}")
    
    async def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        try:
            # Run embedding generation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Use SentenceTransformer for batch processing
            embeddings = await loop.run_in_executor(
                self._executor,
                self._model.encode,
                texts,
                {"convert_to_tensor": False, "show_progress_bar": False}
            )
            
            # Convert to list of lists
            return [embedding.tolist() for embedding in embeddings]
            
        except Exception as e:
            logger.error(f"Error in batch embedding generation: {e}")
            raise EmbeddingException(f"Failed to generate batch embeddings: {str(e)}")
    
    async def generate_single_embedding(
        self,
        text: str,
        use_cache: bool = True
    ) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            use_cache: Whether to use caching
            
        Returns:
            List[float]: Embedding vector
        """
        embeddings = await self.generate_embeddings([text], use_cache)
        return embeddings[0]
    
    async def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            float: Cosine similarity score
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    async def find_most_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar embeddings to a query embedding.
        
        Args:
            query_embedding: Query embedding
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
            
        Returns:
            List[tuple]: List of (index, similarity_score) tuples
        """
        try:
            similarities = []
            
            for i, candidate in enumerate(candidate_embeddings):
                similarity = await self.compute_similarity(query_embedding, candidate)
                similarities.append((i, similarity))
            
            # Sort by similarity score (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding most similar embeddings: {e}")
            return []
    
    async def cluster_embeddings(
        self,
        embeddings: List[List[float]],
        num_clusters: int = 5
    ) -> List[int]:
        """
        Cluster embeddings using K-means.
        
        Args:
            embeddings: List of embeddings
            num_clusters: Number of clusters
            
        Returns:
            List[int]: Cluster assignments
        """
        try:
            from sklearn.cluster import KMeans
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings_array)
            
            return cluster_labels.tolist()
            
        except Exception as e:
            logger.error(f"Error clustering embeddings: {e}")
            return [0] * len(embeddings)
    
    async def reduce_dimensionality(
        self,
        embeddings: List[List[float]],
        target_dimension: int = 2,
        method: str = "pca"
    ) -> List[List[float]]:
        """
        Reduce dimensionality of embeddings for visualization.
        
        Args:
            embeddings: List of embeddings
            target_dimension: Target dimension
            method: Dimensionality reduction method ('pca', 'tsne', 'umap')
            
        Returns:
            List[List[float]]: Reduced embeddings
        """
        try:
            embeddings_array = np.array(embeddings)
            
            if method == "pca":
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=target_dimension)
            elif method == "tsne":
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=target_dimension, random_state=42)
            elif method == "umap":
                import umap
                reducer = umap.UMAP(n_components=target_dimension, random_state=42)
            else:
                raise ValueError(f"Unknown dimensionality reduction method: {method}")
            
            reduced_embeddings = reducer.fit_transform(embeddings_array)
            return reduced_embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Error reducing dimensionality: {e}")
            return embeddings
    
    async def evaluate_embedding_quality(
        self,
        embeddings: List[List[float]],
        labels: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate embedding quality using various metrics.
        
        Args:
            embeddings: List of embeddings
            labels: Optional labels for supervised evaluation
            
        Returns:
            Dict[str, float]: Quality metrics
        """
        try:
            embeddings_array = np.array(embeddings)
            metrics = {}
            
            # Compute average pairwise distance
            from scipy.spatial.distance import pdist
            distances = pdist(embeddings_array, metric='cosine')
            metrics['avg_cosine_distance'] = float(np.mean(distances))
            metrics['std_cosine_distance'] = float(np.std(distances))
            
            # Compute intrinsic dimensionality
            from sklearn.decomposition import PCA
            pca = PCA()
            pca.fit(embeddings_array)
            
            # Find number of components explaining 95% of variance
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            intrinsic_dim = np.argmax(cumsum >= 0.95) + 1
            metrics['intrinsic_dimensionality'] = int(intrinsic_dim)
            
            # If labels are provided, compute supervised metrics
            if labels:
                from sklearn.metrics import silhouette_score
                from sklearn.cluster import KMeans
                
                # Convert labels to numeric
                unique_labels = list(set(labels))
                numeric_labels = [unique_labels.index(label) for label in labels]
                
                # Compute silhouette score
                if len(unique_labels) > 1:
                    silhouette = silhouette_score(embeddings_array, numeric_labels)
                    metrics['silhouette_score'] = float(silhouette)
                
                # Compute clustering accuracy
                kmeans = KMeans(n_clusters=len(unique_labels), random_state=42)
                cluster_labels = kmeans.fit_predict(embeddings_array)
                
                # Simple clustering accuracy (best case assignment)
                from sklearn.metrics import accuracy_score
                from scipy.optimize import linear_sum_assignment
                
                # Create confusion matrix
                confusion_matrix = np.zeros((len(unique_labels), len(unique_labels)))
                for true_label, pred_label in zip(numeric_labels, cluster_labels):
                    confusion_matrix[true_label, pred_label] += 1
                
                # Find optimal assignment
                row_indices, col_indices = linear_sum_assignment(-confusion_matrix)
                
                # Compute accuracy
                accuracy = confusion_matrix[row_indices, col_indices].sum() / len(labels)
                metrics['clustering_accuracy'] = float(accuracy)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating embedding quality: {e}")
            return {}
    
    async def save_embeddings(
        self,
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        filename: str
    ) -> bool:
        """
        Save embeddings to file.
        
        Args:
            embeddings: List of embeddings
            metadata: List of metadata dictionaries
            filename: Output filename
            
        Returns:
            bool: Success status
        """
        try:
            import pickle
            
            data = {
                'embeddings': embeddings,
                'metadata': metadata,
                'model': self.embedding_model,
                'dimension': self.embedding_dimension,
                'created_at': np.datetime64('now').isoformat()
            }
            
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Saved {len(embeddings)} embeddings to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            return False
    
    async def load_embeddings(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Load embeddings from file.
        
        Args:
            filename: Input filename
            
        Returns:
            Optional[Dict[str, Any]]: Loaded embeddings data
        """
        try:
            import pickle
            
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            logger.info(f"Loaded {len(data['embeddings'])} embeddings from {filename}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return None
    
    async def health_check(self) -> bool:
        """Check embedding service health."""
        try:
            # Test embedding generation
            test_text = "This is a test sentence for health check."
            embedding = await self.generate_single_embedding(test_text, use_cache=False)
            
            # Check if embedding has correct dimension
            if len(embedding) != self.embedding_dimension:
                logger.error(f"Embedding dimension mismatch: expected {self.embedding_dimension}, got {len(embedding)}")
                return False
            
            # Check if embedding contains valid values
            if any(not isinstance(x, (int, float)) or np.isnan(x) or np.isinf(x) for x in embedding):
                logger.error("Embedding contains invalid values")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Embedding service health check failed: {e}")
            return False
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
