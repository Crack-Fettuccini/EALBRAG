import os
import json
from typing import Any, List

class SimpleCache:
    def __init__(self, cache_dir: str = "./cache/", max_size: int = 1000):
        """
        Initialize a simple file-based cache.
        :param cache_dir: Directory to store cache files.
        :param max_size: Maximum number of cache entries.
        """
        self.cache_dir = cache_dir
        self.max_size = max_size
        os.makedirs(self.cache_dir, exist_ok=True)
        self.index_file = os.path.join(self.cache_dir, "index.json")
        if not os.path.exists(self.index_file):
            with open(self.index_file, 'w') as f:
                json.dump([], f)
        with open(self.index_file, 'r') as f:
            self.index = json.load(f)
    
    def contains(self, key: str) -> bool:
        return key in self.index
    
    def get(self, key: str) -> List[str]:
        if self.contains(key):
            cache_file = os.path.join(self.cache_dir, f"{key}.json")
            with open(cache_file, 'r') as f:
                return json.load(f)
        return []
    
    def set(self, key: str, value: List[str]):
        if not self.contains(key):
            if len(self.index) >= self.max_size:
                # Remove the oldest cache entry
                oldest_key = self.index.pop(0)
                oldest_file = os.path.join(self.cache_dir, f"{oldest_key}.json")
                if os.path.exists(oldest_file):
                    os.remove(oldest_file)
            self.index.append(key)
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f)
        # Save the cache entry
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        with open(cache_file, 'w') as f:
            json.dump(value, f)
