"""
Deduplication Engine

Detects near-duplicate content using:
- Exact hash matching
- MinHash + LSH for near-duplicates
- SimHash for content similarity
"""

from dataclasses import dataclass
from typing import Optional, Set
import hashlib
import re
from collections import defaultdict


@dataclass
class DedupConfig:
    """Configuration for deduplication."""
    
    # MinHash settings
    num_hashes: int = 128
    shingle_size: int = 3  # Word n-grams
    
    # LSH settings
    num_bands: int = 16
    rows_per_band: int = 8  # num_hashes / num_bands
    
    # Similarity threshold
    similarity_threshold: float = 0.8
    
    # Behavior
    keep_duplicates: bool = True  # Store but mark as duplicate
    prefer_newer: bool = False  # If True, mark older as duplicate


class MinHasher:
    """
    MinHash signature generator for near-duplicate detection.
    
    Uses Locality-Sensitive Hashing (LSH) for efficient 
    similarity search in large datasets.
    """
    
    def __init__(self, config: Optional[DedupConfig] = None):
        self.config = config or DedupConfig()
        
        # Generate hash functions (using a*x + b mod p)
        import random
        random.seed(42)  # Reproducible
        
        self.hash_funcs = []
        p = 2**31 - 1  # Large prime
        
        for _ in range(self.config.num_hashes):
            a = random.randint(1, p - 1)
            b = random.randint(0, p - 1)
            self.hash_funcs.append((a, b, p))
    
    def _tokenize(self, text: str) -> set[str]:
        """Convert text to shingles (word n-grams)."""
        # Normalize
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        
        # Generate shingles
        shingles = set()
        n = self.config.shingle_size
        
        for i in range(len(words) - n + 1):
            shingle = ' '.join(words[i:i + n])
            shingles.add(shingle)
        
        return shingles
    
    def _hash_shingle(self, shingle: str) -> int:
        """Hash a shingle to an integer."""
        return int(hashlib.md5(shingle.encode()).hexdigest(), 16)
    
    def compute_signature(self, text: str) -> list[int]:
        """
        Compute MinHash signature for text.
        
        Returns a list of hash values (signature).
        """
        shingles = self._tokenize(text)
        
        if not shingles:
            return [0] * self.config.num_hashes
        
        # Hash each shingle
        shingle_hashes = [self._hash_shingle(s) for s in shingles]
        
        # Compute signature (min hash for each hash function)
        signature = []
        
        for a, b, p in self.hash_funcs:
            min_hash = float('inf')
            
            for h in shingle_hashes:
                hash_val = (a * h + b) % p
                min_hash = min(min_hash, hash_val)
            
            signature.append(min_hash)
        
        return signature
    
    def estimate_similarity(self, sig1: list[int], sig2: list[int]) -> float:
        """Estimate Jaccard similarity from signatures."""
        if len(sig1) != len(sig2):
            return 0.0
        
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)


class LSHIndex:
    """
    Locality-Sensitive Hashing index for fast similarity search.
    
    Divides signatures into bands and hashes each band.
    Similar items will likely hash to the same bucket in at least one band.
    """
    
    def __init__(self, config: Optional[DedupConfig] = None):
        self.config = config or DedupConfig()
        
        # Buckets for each band
        # band_index -> bucket_hash -> set of item_ids
        self.buckets: list[dict[int, set[str]]] = [
            defaultdict(set) for _ in range(self.config.num_bands)
        ]
        
        # Store signatures
        self.signatures: dict[str, list[int]] = {}
    
    def _band_hash(self, band: list[int]) -> int:
        """Hash a band of the signature."""
        return hash(tuple(band))
    
    def add(self, item_id: str, signature: list[int]):
        """Add an item to the index."""
        self.signatures[item_id] = signature
        
        # Split signature into bands
        rows = self.config.rows_per_band
        
        for band_idx in range(self.config.num_bands):
            start = band_idx * rows
            end = start + rows
            band = signature[start:end]
            
            bucket_hash = self._band_hash(band)
            self.buckets[band_idx][bucket_hash].add(item_id)
    
    def find_candidates(self, signature: list[int]) -> set[str]:
        """Find candidate duplicates (items that share a bucket)."""
        candidates = set()
        rows = self.config.rows_per_band
        
        for band_idx in range(self.config.num_bands):
            start = band_idx * rows
            end = start + rows
            band = signature[start:end]
            
            bucket_hash = self._band_hash(band)
            candidates.update(self.buckets[band_idx][bucket_hash])
        
        return candidates
    
    def remove(self, item_id: str):
        """Remove an item from the index."""
        if item_id not in self.signatures:
            return
        
        signature = self.signatures[item_id]
        rows = self.config.rows_per_band
        
        for band_idx in range(self.config.num_bands):
            start = band_idx * rows
            end = start + rows
            band = signature[start:end]
            
            bucket_hash = self._band_hash(band)
            self.buckets[band_idx][bucket_hash].discard(item_id)
        
        del self.signatures[item_id]


class DeduplicationEngine:
    """
    Complete deduplication system.
    
    Features:
    - Exact duplicate detection (hash)
    - Near-duplicate detection (MinHash + LSH)
    - Configurable similarity threshold
    - Fast candidate retrieval
    """
    
    def __init__(self, config: Optional[DedupConfig] = None):
        self.config = config or DedupConfig()
        
        self.hasher = MinHasher(self.config)
        self.lsh_index = LSHIndex(self.config)
        
        # Exact hash index
        self.content_hashes: dict[str, str] = {}  # hash -> item_id
        
        # Duplicate tracking
        self.duplicates: dict[str, str] = {}  # duplicate_id -> original_id
    
    def check_and_add(
        self,
        item_id: str,
        content: str,
        content_hash: Optional[str] = None,
    ) -> tuple[bool, Optional[str], float]:
        """
        Check if content is a duplicate and add to index.
        
        Returns:
            (is_duplicate, original_id, similarity)
        """
        # Compute content hash
        if not content_hash:
            content_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Check exact duplicate
        if content_hash in self.content_hashes:
            original_id = self.content_hashes[content_hash]
            self.duplicates[item_id] = original_id
            return True, original_id, 1.0
        
        # Compute MinHash signature
        signature = self.hasher.compute_signature(content)
        
        # Find candidates via LSH
        candidates = self.lsh_index.find_candidates(signature)
        
        # Check similarity with candidates
        best_match = None
        best_similarity = 0.0
        
        for candidate_id in candidates:
            if candidate_id == item_id:
                continue
            
            candidate_sig = self.lsh_index.signatures.get(candidate_id)
            if not candidate_sig:
                continue
            
            similarity = self.hasher.estimate_similarity(signature, candidate_sig)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = candidate_id
        
        # Determine if duplicate
        is_duplicate = best_similarity >= self.config.similarity_threshold
        
        if is_duplicate and best_match:
            self.duplicates[item_id] = best_match
        
        # Add to indexes (even if duplicate, for future comparisons)
        self.content_hashes[content_hash] = item_id
        self.lsh_index.add(item_id, signature)
        
        return is_duplicate, best_match if is_duplicate else None, best_similarity
    
    def get_signature(self, content: str) -> list[int]:
        """Get MinHash signature for content."""
        return self.hasher.compute_signature(content)
    
    def is_duplicate(self, item_id: str) -> bool:
        """Check if an item is marked as duplicate."""
        return item_id in self.duplicates
    
    def get_original(self, duplicate_id: str) -> Optional[str]:
        """Get the original item ID for a duplicate."""
        return self.duplicates.get(duplicate_id)
    
    def remove(self, item_id: str):
        """Remove an item from the dedup index."""
        self.lsh_index.remove(item_id)
        
        # Remove from content hashes
        to_remove = [h for h, i in self.content_hashes.items() if i == item_id]
        for h in to_remove:
            del self.content_hashes[h]
        
        # Remove from duplicates
        if item_id in self.duplicates:
            del self.duplicates[item_id]
    
    def get_stats(self) -> dict:
        """Get deduplication statistics."""
        return {
            "total_items": len(self.lsh_index.signatures),
            "duplicates_found": len(self.duplicates),
            "unique_hashes": len(self.content_hashes),
        }
