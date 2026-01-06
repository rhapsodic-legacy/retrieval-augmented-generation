"""
Fusion Strategies

Combines results from multiple search methods:
- Reciprocal Rank Fusion (RRF)
- Weighted combination
- Learned fusion (ML-based re-ranking)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any
import math
import json
from pathlib import Path


@dataclass
class FusedResult:
    """A result after fusion."""
    id: str
    content: str
    final_score: float
    
    # Scores from each source
    vector_score: Optional[float] = None
    sparse_score: Optional[float] = None
    graph_score: Optional[float] = None
    
    # Source information
    sources: list[str] = field(default_factory=list)  # Which methods found this
    metadata: dict = field(default_factory=dict)
    
    # Explanation
    explanation: str = ""


class BaseFusion(ABC):
    """Abstract base class for fusion strategies."""
    
    @abstractmethod
    def fuse(
        self,
        vector_results: list[dict],
        sparse_results: list[dict],
        graph_results: list[dict],
        k: int = 10,
    ) -> list[FusedResult]:
        """Fuse results from multiple sources."""
        pass


class ReciprocalRankFusion(BaseFusion):
    """
    Reciprocal Rank Fusion (RRF).
    
    Simple but effective fusion that doesn't require score normalization.
    Score = sum(1 / (k + rank)) for each result list.
    
    Paper: "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
    """
    
    def __init__(self, k: int = 60):
        """
        Initialize RRF.
        
        Args:
            k: Constant to prevent high rankings from dominating (typically 60)
        """
        self.k = k
    
    def fuse(
        self,
        vector_results: list[dict],
        sparse_results: list[dict],
        graph_results: list[dict],
        k: int = 10,
    ) -> list[FusedResult]:
        """Fuse using RRF."""
        # Collect all unique results
        all_results: dict[str, FusedResult] = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results):
            doc_id = result["id"]
            rrf_score = 1 / (self.k + rank + 1)
            
            if doc_id not in all_results:
                all_results[doc_id] = FusedResult(
                    id=doc_id,
                    content=result.get("content", ""),
                    final_score=0,
                    metadata=result.get("metadata", {}),
                )
            
            all_results[doc_id].final_score += rrf_score
            all_results[doc_id].vector_score = result.get("score", 0)
            all_results[doc_id].sources.append("vector")
        
        # Process sparse results
        for rank, result in enumerate(sparse_results):
            doc_id = result["id"]
            rrf_score = 1 / (self.k + rank + 1)
            
            if doc_id not in all_results:
                all_results[doc_id] = FusedResult(
                    id=doc_id,
                    content=result.get("content", ""),
                    final_score=0,
                    metadata=result.get("metadata", {}),
                )
            
            all_results[doc_id].final_score += rrf_score
            all_results[doc_id].sparse_score = result.get("score", 0)
            if "sparse" not in all_results[doc_id].sources:
                all_results[doc_id].sources.append("sparse")
        
        # Process graph results
        for rank, result in enumerate(graph_results):
            doc_id = result["id"]
            rrf_score = 1 / (self.k + rank + 1)
            
            if doc_id not in all_results:
                all_results[doc_id] = FusedResult(
                    id=doc_id,
                    content=result.get("content", ""),
                    final_score=0,
                    metadata=result.get("metadata", {}),
                )
            
            all_results[doc_id].final_score += rrf_score
            all_results[doc_id].graph_score = result.get("score", 0)
            if "graph" not in all_results[doc_id].sources:
                all_results[doc_id].sources.append("graph")
        
        # Sort by final score and return top k
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x.final_score,
            reverse=True,
        )
        
        # Add explanations
        for result in sorted_results:
            parts = []
            if result.vector_score is not None:
                parts.append(f"vector: {result.vector_score:.3f}")
            if result.sparse_score is not None:
                parts.append(f"sparse: {result.sparse_score:.3f}")
            if result.graph_score is not None:
                parts.append(f"graph: {result.graph_score:.3f}")
            result.explanation = f"RRF fusion from {', '.join(result.sources)}. Scores: {'; '.join(parts)}"
        
        return sorted_results[:k]


class WeightedFusion(BaseFusion):
    """
    Weighted score combination.
    
    Requires score normalization to make scores comparable.
    """
    
    def __init__(
        self,
        vector_weight: float = 0.4,
        sparse_weight: float = 0.3,
        graph_weight: float = 0.3,
        normalize: bool = True,
    ):
        """
        Initialize weighted fusion.
        
        Args:
            vector_weight: Weight for vector search scores
            sparse_weight: Weight for sparse search scores
            graph_weight: Weight for graph search scores
            normalize: Whether to normalize scores to [0, 1]
        """
        self.vector_weight = vector_weight
        self.sparse_weight = sparse_weight
        self.graph_weight = graph_weight
        self.normalize = normalize
    
    def _normalize_scores(self, results: list[dict]) -> list[dict]:
        """Normalize scores to [0, 1] using min-max."""
        if not results:
            return results
        
        scores = [r.get("score", 0) for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score - min_score < 1e-10:
            # All scores are the same
            return [
                {**r, "score": 1.0 if scores else 0.0}
                for r in results
            ]
        
        return [
            {**r, "score": (r.get("score", 0) - min_score) / (max_score - min_score)}
            for r in results
        ]
    
    def fuse(
        self,
        vector_results: list[dict],
        sparse_results: list[dict],
        graph_results: list[dict],
        k: int = 10,
    ) -> list[FusedResult]:
        """Fuse using weighted combination."""
        # Normalize if requested
        if self.normalize:
            vector_results = self._normalize_scores(vector_results)
            sparse_results = self._normalize_scores(sparse_results)
            graph_results = self._normalize_scores(graph_results)
        
        # Build score maps
        vector_scores = {r["id"]: r for r in vector_results}
        sparse_scores = {r["id"]: r for r in sparse_results}
        graph_scores = {r["id"]: r for r in graph_results}
        
        # Get all unique IDs
        all_ids = set(vector_scores.keys()) | set(sparse_scores.keys()) | set(graph_scores.keys())
        
        # Calculate weighted scores
        results = []
        for doc_id in all_ids:
            v_result = vector_scores.get(doc_id, {})
            s_result = sparse_scores.get(doc_id, {})
            g_result = graph_scores.get(doc_id, {})
            
            v_score = v_result.get("score", 0)
            s_score = s_result.get("score", 0)
            g_score = g_result.get("score", 0)
            
            final_score = (
                self.vector_weight * v_score +
                self.sparse_weight * s_score +
                self.graph_weight * g_score
            )
            
            # Get content from first available source
            content = v_result.get("content") or s_result.get("content") or g_result.get("content", "")
            metadata = v_result.get("metadata") or s_result.get("metadata") or g_result.get("metadata", {})
            
            sources = []
            if doc_id in vector_scores:
                sources.append("vector")
            if doc_id in sparse_scores:
                sources.append("sparse")
            if doc_id in graph_scores:
                sources.append("graph")
            
            results.append(FusedResult(
                id=doc_id,
                content=content,
                final_score=final_score,
                vector_score=v_score if doc_id in vector_scores else None,
                sparse_score=s_score if doc_id in sparse_scores else None,
                graph_score=g_score if doc_id in graph_scores else None,
                sources=sources,
                metadata=metadata,
                explanation=f"Weighted fusion (v={self.vector_weight}, s={self.sparse_weight}, g={self.graph_weight})",
            ))
        
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results[:k]


class LearnedFusion(BaseFusion):
    """
    ML-based fusion using a learned model.
    
    Features used:
    - Individual scores from each retriever
    - Rank positions
    - Score differences
    - Query features
    
    Model: Simple MLP or gradient boosting
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_lightgbm: bool = True,
    ):
        """
        Initialize learned fusion.
        
        Args:
            model_path: Path to trained model
            use_lightgbm: Use LightGBM (if False, use simple MLP)
        """
        self.model_path = model_path
        self.use_lightgbm = use_lightgbm
        self.model = None
        self.scaler = None
        
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, path: str):
        """Load a trained model."""
        import pickle
        
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data.get("scaler")
    
    def _extract_features(
        self,
        doc_id: str,
        vector_results: list[dict],
        sparse_results: list[dict],
        graph_results: list[dict],
    ) -> list[float]:
        """Extract features for a document."""
        # Find positions and scores
        v_pos, v_score = self._find_in_results(doc_id, vector_results)
        s_pos, s_score = self._find_in_results(doc_id, sparse_results)
        g_pos, g_score = self._find_in_results(doc_id, graph_results)
        
        # Feature vector
        features = [
            # Raw scores
            v_score,
            s_score,
            g_score,
            
            # Reciprocal ranks
            1 / (v_pos + 1) if v_pos >= 0 else 0,
            1 / (s_pos + 1) if s_pos >= 0 else 0,
            1 / (g_pos + 1) if g_pos >= 0 else 0,
            
            # Source counts
            float(v_pos >= 0),
            float(s_pos >= 0),
            float(g_pos >= 0),
            float((v_pos >= 0) + (s_pos >= 0) + (g_pos >= 0)),
            
            # Score differences
            v_score - s_score if v_pos >= 0 and s_pos >= 0 else 0,
            v_score - g_score if v_pos >= 0 and g_pos >= 0 else 0,
            s_score - g_score if s_pos >= 0 and g_pos >= 0 else 0,
            
            # Max and mean scores
            max(v_score, s_score, g_score),
            (v_score + s_score + g_score) / 3,
        ]
        
        return features
    
    def _find_in_results(
        self,
        doc_id: str,
        results: list[dict],
    ) -> tuple[int, float]:
        """Find document position and score in results."""
        for i, r in enumerate(results):
            if r["id"] == doc_id:
                return i, r.get("score", 0)
        return -1, 0
    
    def fuse(
        self,
        vector_results: list[dict],
        sparse_results: list[dict],
        graph_results: list[dict],
        k: int = 10,
    ) -> list[FusedResult]:
        """Fuse using learned model."""
        # Get all unique IDs
        all_ids = set()
        all_ids.update(r["id"] for r in vector_results)
        all_ids.update(r["id"] for r in sparse_results)
        all_ids.update(r["id"] for r in graph_results)
        
        if not self.model:
            # Fall back to RRF if no model loaded
            return ReciprocalRankFusion().fuse(
                vector_results, sparse_results, graph_results, k
            )
        
        # Extract features and predict
        import numpy as np
        
        results = []
        for doc_id in all_ids:
            features = self._extract_features(
                doc_id, vector_results, sparse_results, graph_results
            )
            
            # Scale if scaler available
            features_array = np.array([features])
            if self.scaler:
                features_array = self.scaler.transform(features_array)
            
            # Predict
            if self.use_lightgbm:
                score = self.model.predict(features_array)[0]
            else:
                score = self.model.predict_proba(features_array)[0, 1]
            
            # Get content
            v_result = next((r for r in vector_results if r["id"] == doc_id), {})
            s_result = next((r for r in sparse_results if r["id"] == doc_id), {})
            g_result = next((r for r in graph_results if r["id"] == doc_id), {})
            
            content = v_result.get("content") or s_result.get("content") or g_result.get("content", "")
            metadata = v_result.get("metadata") or s_result.get("metadata") or g_result.get("metadata", {})
            
            sources = []
            if v_result:
                sources.append("vector")
            if s_result:
                sources.append("sparse")
            if g_result:
                sources.append("graph")
            
            results.append(FusedResult(
                id=doc_id,
                content=content,
                final_score=float(score),
                vector_score=v_result.get("score"),
                sparse_score=s_result.get("score"),
                graph_score=g_result.get("score"),
                sources=sources,
                metadata=metadata,
                explanation="ML-based learned fusion",
            ))
        
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results[:k]
    
    def train(
        self,
        training_data: list[dict],
        save_path: Optional[str] = None,
    ):
        """
        Train the fusion model.
        
        training_data format:
        [
            {
                "query": "...",
                "relevant_docs": ["id1", "id2"],
                "vector_results": [...],
                "sparse_results": [...],
                "graph_results": [...],
            },
            ...
        ]
        """
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        
        X = []
        y = []
        
        for sample in training_data:
            relevant_docs = set(sample["relevant_docs"])
            vector_results = sample.get("vector_results", [])
            sparse_results = sample.get("sparse_results", [])
            graph_results = sample.get("graph_results", [])
            
            all_ids = set()
            all_ids.update(r["id"] for r in vector_results)
            all_ids.update(r["id"] for r in sparse_results)
            all_ids.update(r["id"] for r in graph_results)
            
            for doc_id in all_ids:
                features = self._extract_features(
                    doc_id, vector_results, sparse_results, graph_results
                )
                X.append(features)
                y.append(1 if doc_id in relevant_docs else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        if self.use_lightgbm:
            import lightgbm as lgb
            
            self.model = lgb.LGBMRanker(
                objective="lambdarank",
                n_estimators=100,
                learning_rate=0.1,
            )
            # For ranking, we need groups
            # Simplified: treat each sample as its own group
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
            )
            self.model.fit(X_scaled, y)
        else:
            from sklearn.neural_network import MLPClassifier
            
            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
            )
            self.model.fit(X_scaled, y)
        
        # Save if path provided
        if save_path:
            import pickle
            
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump({
                    "model": self.model,
                    "scaler": self.scaler,
                }, f)


def create_fusion(
    strategy: str = "rrf",
    **kwargs,
) -> BaseFusion:
    """Factory function to create a fusion strategy."""
    strategy = strategy.lower()
    
    if strategy == "rrf":
        return ReciprocalRankFusion(**kwargs)
    elif strategy == "weighted":
        return WeightedFusion(**kwargs)
    elif strategy == "learned":
        return LearnedFusion(**kwargs)
    else:
        raise ValueError(f"Unknown fusion strategy: {strategy}")
