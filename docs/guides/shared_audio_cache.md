# Audio Cache Backend

Transforms like `AddBackgroundNoise` and `ApplyImpulseResponse` load audio files from disk and resample them on each use.

When processing large datasets with parallel workers, each worker loads the same files independently. If you have `N` workers, the same impulse responses and background noises get loaded and resampled `N` times instead of once.

The audio cache backend solves this by providing a shared mutable cache across workers that trades memory for processing time.

## Example script

This example uses Ray since its object store is a great fit for this usecase, but any backend as long as it implements the `get(key)`, `put(key, audio)` interface will work.

```python
#!/usr/bin/env python
"""Audio Cache Backend Benchmark"""
import numpy as np
import tempfile
import time
import shutil
from pathlib import Path
import soundfile as sf
import ray

from audiomentations import AddBackgroundNoise, set_audio_cache_backend, clear_audio_cache_backend


@ray.remote
class AudioCacheActor:
    """Shared mutable cache with LRU eviction."""
    
    def __init__(self, max_memory_mb: float = 1000):
        self.cache = {}
        self.max_bytes = int(max_memory_mb * 1024 * 1024)
        self.order = []
        self.hits = 0
        self.misses = 0
    
    def get(self, key):
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, key, audio):
        if key in self.cache:
            return
        while self._used() + audio.nbytes > self.max_bytes and self.order:
            del self.cache[self.order.pop(0)]
        self.cache[key] = audio
        self.order.append(key)
    
    def _used(self):
        return sum(v.nbytes for v in self.cache.values())
    
    def stats(self):
        return {
            'entries': len(self.cache),
            'memory_mb': self._used() / 1024 / 1024,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
        }


class RayAudioCacheBackend:
    def __init__(self, actor):
        self.actor = actor
    
    def get(self, key):
        return ray.get(self.actor.get.remote(key))
    
    def put(self, key, audio):
        ray.get(self.actor.put.remote(key, audio))


@ray.remote
def process_with_cache(audio, noise_dir, cache_actor):
    set_audio_cache_backend(RayAudioCacheBackend(cache_actor))
    transform = AddBackgroundNoise(sounds_path=str(noise_dir), p=1.0)
    return transform(audio, sample_rate=44100)


@ray.remote
def process_without_cache(audio, noise_dir):
    clear_audio_cache_backend()
    transform = AddBackgroundNoise(sounds_path=str(noise_dir), p=1.0)
    return transform(audio, sample_rate=44100)


if __name__ == "__main__":
    # Generate test files
    temp_dir = tempfile.mkdtemp()
    noise_dir = Path(temp_dir) / "noise"
    noise_dir.mkdir()

    print("Generating 20 noise files @ 48000Hz...")
    for i in range(20):
        noise = np.random.randn(30 * 48000).astype(np.float32) * 0.1
        sf.write(noise_dir / f"noise_{i:02d}.wav", noise, 48000)

    input_audio = np.random.randn(5 * 44100).astype(np.float32)

    ray.init(ignore_reinit_error=True)
    cache_actor = AudioCacheActor.remote(max_memory_mb=500)

    print("\nRunning 200 tasks WITHOUT cache...")
    start = time.perf_counter()
    ray.get([process_without_cache.remote(input_audio, str(noise_dir)) for _ in range(200)])
    time_no_cache = time.perf_counter() - start

    print("Running 200 tasks WITH cache...")
    start = time.perf_counter()
    ray.get([process_with_cache.remote(input_audio, str(noise_dir), cache_actor) for _ in range(200)])
    time_with_cache = time.perf_counter() - start

    stats = ray.get(cache_actor.stats.remote())

    print(f"\n{'='*50}")
    print(f"Without cache: {time_no_cache:.2f}s")
    print(f"With cache:    {time_with_cache:.2f}s")
    print(f"Speedup:       {time_no_cache/time_with_cache:.1f}x")
    print(f"Cache:         {stats['entries']} files, {stats['memory_mb']:.0f} MB, {stats['hit_rate']:.0%} hit rate")
    print(f"{'='*50}")

    shutil.rmtree(temp_dir)
    ray.shutdown()
```

Output:

```
Generating 20 noise files @ 48000Hz...

Running 200 tasks WITHOUT cache...
Running 200 tasks WITH cache...

==================================================
Without cache: 4.05s
With cache:    0.37s
Speedup:       11.1x
Cache:         20 files, 101 MB, 73% hit rate
==================================================
```

## Cache Key Format

```python
cache_key = (file_path, sample_rate, mono)
```

All three must match for a cache hit.

## Cache Behavior

- **Stores full files**: On cache miss, the full file is loaded and cached. Slicing for offset/duration happens on retrieval.
- **LRU eviction**: When memory limit is reached, least recently used items are evicted.
- **No pre-loading**: Workers start immediately; cache warms up during processing.

## Backend Interface

Any object with these methods works:

```python
class CacheBackend:
    def get(self, key: tuple) -> np.ndarray | None: ...
    def put(self, key: tuple, audio: np.ndarray) -> None: ...
```

## API

```python
from audiomentations import (
    set_audio_cache_backend,
    get_audio_cache_backend,
    clear_audio_cache_backend,
)
```
