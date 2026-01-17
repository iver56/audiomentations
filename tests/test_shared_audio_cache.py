"""
Tests for the audio cache backend feature in audio_loading_utils.

The cache backend allows a mutable cache with get/put interface to be used
by load_sound_file(). This is useful for sharing audio across multiprocessing
workers (e.g., with a Ray Actor) to avoid redundant I/O and resampling.
"""
import os
import numpy as np
import pytest

from audiomentations.core.audio_loading_utils import (
    load_sound_file,
    set_audio_cache_backend,
    get_audio_cache_backend,
    clear_audio_cache_backend,
)
from demo.demo import DEMO_DIR


class DictCacheBackend:
    """Simple dict-based cache backend for testing."""
    
    def __init__(self, initial_cache=None):
        self.cache = initial_cache or {}
        self.gets = 0
        self.puts = 0
    
    def get(self, key):
        self.gets += 1
        return self.cache.get(key)
    
    def put(self, key, audio):
        self.puts += 1
        self.cache[key] = audio


class TestAudioCacheBackendBasics:
    """Test basic cache backend get/set/clear operations."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_audio_cache_backend()

    def teardown_method(self):
        """Clear cache after each test."""
        clear_audio_cache_backend()

    def test_cache_is_none_by_default(self):
        """Cache backend should be None when not set."""
        assert get_audio_cache_backend() is None

    def test_set_and_get_backend(self):
        """Setting backend should make it retrievable."""
        backend = DictCacheBackend()
        set_audio_cache_backend(backend)
        assert get_audio_cache_backend() is backend

    def test_clear_backend(self):
        """Clearing backend should set it to None."""
        backend = DictCacheBackend()
        set_audio_cache_backend(backend)
        clear_audio_cache_backend()
        assert get_audio_cache_backend() is None

    def test_set_backend_to_none(self):
        """Setting backend to None should disable it."""
        backend = DictCacheBackend()
        set_audio_cache_backend(backend)
        set_audio_cache_backend(None)
        assert get_audio_cache_backend() is None


class TestAudioCacheBackendWithLoadSoundFile:
    """Test that load_sound_file() correctly uses the cache backend."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_audio_cache_backend()

    def teardown_method(self):
        """Clear cache after each test."""
        clear_audio_cache_backend()

    def test_cache_miss_loads_from_disk_and_caches(self):
        """When file is not in cache, should load from disk and cache it."""
        backend = DictCacheBackend()
        set_audio_cache_backend(backend)

        file_path = os.path.join(DEMO_DIR, "acoustic_guitar_0.wav")
        samples, sample_rate = load_sound_file(file_path, sample_rate=16000)

        assert sample_rate == 16000
        assert samples.dtype == np.float32
        assert samples.ndim == 1
        assert len(samples) > 0
        
        # Should have called get (miss) and put
        assert backend.gets == 1
        assert backend.puts == 1
        
        # Cache should now contain the file
        cache_key = (file_path, 16000, True)
        assert cache_key in backend.cache

    def test_cache_hit_returns_cached_data_mono(self):
        """When file is in cache, should return cached data without loading."""
        file_path = os.path.join(DEMO_DIR, "acoustic_guitar_0.wav")
        sample_rate = 16000

        # Create cache with fake audio
        cached_audio = np.ones(1000, dtype=np.float32) * 0.5
        cache_key = (file_path, sample_rate, True)
        backend = DictCacheBackend({cache_key: cached_audio})
        set_audio_cache_backend(backend)

        # Load - should return cached data
        samples, returned_sr = load_sound_file(file_path, sample_rate=sample_rate, mono=True)

        assert returned_sr == sample_rate
        assert samples.dtype == np.float32
        assert samples.ndim == 1
        assert len(samples) == 1000
        assert np.allclose(samples, 0.5)
        
        # Should have called get (hit) but not put
        assert backend.gets == 1
        assert backend.puts == 0

    def test_cache_hit_returns_cached_data_stereo(self):
        """When file is in cache with mono=False, should return stereo cached data."""
        file_path = os.path.join(DEMO_DIR, "stereo_16bit.wav")
        sample_rate = 16000

        # Create fake cached stereo audio
        cached_audio = np.ones((2, 500), dtype=np.float32) * 0.25
        cache_key = (file_path, sample_rate, False)
        backend = DictCacheBackend({cache_key: cached_audio})
        set_audio_cache_backend(backend)

        # Load - should return cached stereo data
        samples, returned_sr = load_sound_file(file_path, sample_rate=sample_rate, mono=False)

        assert returned_sr == sample_rate
        assert samples.dtype == np.float32
        assert samples.ndim == 2
        assert samples.shape == (2, 500)
        assert np.allclose(samples, 0.25)

    def test_sample_rate_mismatch_causes_cache_miss(self):
        """Cache key includes sample_rate, so mismatched rate should miss."""
        file_path = os.path.join(DEMO_DIR, "acoustic_guitar_0.wav")

        # Cache at 44100 Hz
        cached_audio = np.ones(1000, dtype=np.float32) * 0.5
        cache_key = (file_path, 44100, True)
        backend = DictCacheBackend({cache_key: cached_audio})
        set_audio_cache_backend(backend)

        # Request at 16000 Hz - should miss cache and load from disk
        samples, returned_sr = load_sound_file(file_path, sample_rate=16000, mono=True)

        assert returned_sr == 16000
        # Should NOT be our cached data (which was all 0.5)
        assert not np.allclose(samples, 0.5)
        # Should have cached at 16000 Hz
        assert (file_path, 16000, True) in backend.cache

    def test_mono_mismatch_causes_cache_miss(self):
        """Cache key includes mono flag, so mismatched mono should miss."""
        file_path = os.path.join(DEMO_DIR, "stereo_16bit.wav")
        sample_rate = 16000

        # Cache as mono
        cached_audio = np.ones(1000, dtype=np.float32) * 0.5
        cache_key = (file_path, sample_rate, True)
        backend = DictCacheBackend({cache_key: cached_audio})
        set_audio_cache_backend(backend)

        # Request as stereo - should miss cache and load from disk
        samples, returned_sr = load_sound_file(file_path, sample_rate=sample_rate, mono=False)

        assert returned_sr == sample_rate
        # Should be stereo from disk, not our mono cached data
        assert samples.ndim == 2

    def test_cache_returns_copy_not_reference(self):
        """Cached data should be copied to prevent mutation."""
        file_path = "/fake/path.wav"
        sample_rate = 44100

        original_cached = np.ones(100, dtype=np.float32)
        cache_key = (file_path, sample_rate, True)
        backend = DictCacheBackend({cache_key: original_cached})
        set_audio_cache_backend(backend)

        # Get from cache
        samples, _ = load_sound_file(file_path, sample_rate=sample_rate, mono=True)

        # Mutate the returned samples
        samples[:] = 999.0

        # Original cache should be unchanged
        assert np.allclose(original_cached, 1.0)

    def test_offset_slicing_mono(self):
        """Offset parameter should slice cached mono audio correctly."""
        file_path = "/fake/path.wav"
        sample_rate = 1000  # 1000 samples/sec for easy math

        # 2 seconds of audio
        cached_audio = np.arange(2000, dtype=np.float32)
        cache_key = (file_path, sample_rate, True)
        backend = DictCacheBackend({cache_key: cached_audio})
        set_audio_cache_backend(backend)

        # Request with 0.5 second offset
        samples, _ = load_sound_file(file_path, sample_rate=sample_rate, mono=True, offset=0.5)

        # Should start at sample 500
        assert samples[0] == 500.0
        assert len(samples) == 1500  # remaining samples

    def test_duration_slicing_mono(self):
        """Duration parameter should slice cached mono audio correctly."""
        file_path = "/fake/path.wav"
        sample_rate = 1000

        # 2 seconds of audio
        cached_audio = np.arange(2000, dtype=np.float32)
        cache_key = (file_path, sample_rate, True)
        backend = DictCacheBackend({cache_key: cached_audio})
        set_audio_cache_backend(backend)

        # Request 0.5 seconds starting from offset 0.2
        samples, _ = load_sound_file(
            file_path, sample_rate=sample_rate, mono=True, offset=0.2, duration=0.5
        )

        # Should be samples 200-700
        assert len(samples) == 500
        assert samples[0] == 200.0
        assert samples[-1] == 699.0

    def test_offset_slicing_stereo(self):
        """Offset parameter should slice cached stereo audio correctly."""
        file_path = "/fake/path.wav"
        sample_rate = 1000

        # 2 seconds of stereo audio
        cached_audio = np.stack([
            np.arange(2000, dtype=np.float32),
            np.arange(2000, dtype=np.float32) + 10000,
        ])
        cache_key = (file_path, sample_rate, False)
        backend = DictCacheBackend({cache_key: cached_audio})
        set_audio_cache_backend(backend)

        # Request with 0.3 second offset
        samples, _ = load_sound_file(file_path, sample_rate=sample_rate, mono=False, offset=0.3)

        # Should be shape (2, 1700)
        assert samples.shape == (2, 1700)
        assert samples[0, 0] == 300.0
        assert samples[1, 0] == 10300.0

    def test_no_cache_set_loads_from_disk(self):
        """When no cache is set, should load normally from disk."""
        # Ensure cache is None
        clear_audio_cache_backend()

        file_path = os.path.join(DEMO_DIR, "acoustic_guitar_0.wav")
        samples, sample_rate = load_sound_file(file_path, sample_rate=None)

        assert sample_rate == 16000
        assert samples.dtype == np.float32
        assert len(samples) == 140544

    def test_cache_miss_with_offset_caches_full_file(self):
        """When loading with offset, should cache the FULL file, not the slice."""
        backend = DictCacheBackend()
        set_audio_cache_backend(backend)

        file_path = os.path.join(DEMO_DIR, "acoustic_guitar_0.wav")
        sample_rate = 16000
        
        # Load with offset
        samples, _ = load_sound_file(file_path, sample_rate=sample_rate, offset=1.0, duration=1.0)
        
        # Cached version should be the full file
        cache_key = (file_path, sample_rate, True)
        cached = backend.cache[cache_key]
        
        # Full file is longer than 2 seconds
        assert len(cached) > len(samples)
        assert len(cached) == 140544  # Full file length at 16kHz


class TestAudioCacheBackendWithTransforms:
    """Test that transforms work correctly with the cache backend."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_audio_cache_backend()

    def teardown_method(self):
        """Clear cache after each test."""
        clear_audio_cache_backend()

    def test_apply_impulse_response_with_cache(self):
        """ApplyImpulseResponse should work with cached impulse responses."""
        from audiomentations import ApplyImpulseResponse

        ir_dir = os.path.join(DEMO_DIR, "ir")
        sample_rate = 16000

        # Pre-load IRs into cache
        cache = {}
        ir_file = os.path.join(ir_dir, "impulse_response_0.wav")
        mono_ir, _ = load_sound_file(ir_file, sample_rate=sample_rate, mono=True)
        stereo_ir, _ = load_sound_file(ir_file, sample_rate=sample_rate, mono=False)
        cache[(ir_file, sample_rate, True)] = mono_ir
        cache[(ir_file, sample_rate, False)] = stereo_ir

        backend = DictCacheBackend(cache)
        set_audio_cache_backend(backend)

        # Create transform and apply
        transform = ApplyImpulseResponse(ir_path=ir_dir, p=1.0)
        input_audio = np.random.randn(1024).astype(np.float32)

        output = transform(input_audio, sample_rate=sample_rate)

        assert output.dtype == np.float32
        assert output.shape == input_audio.shape

    def test_add_background_noise_with_cache(self):
        """AddBackgroundNoise should work with cached noise files."""
        from audiomentations import AddBackgroundNoise

        noise_dir = os.path.join(DEMO_DIR, "background_noises")
        sample_rate = 16000

        # Pre-load noise into cache
        cache = {}
        noise_file = os.path.join(noise_dir, "hens.ogg")
        noise_audio, _ = load_sound_file(noise_file, sample_rate=sample_rate, mono=True)
        cache[(noise_file, sample_rate, True)] = noise_audio

        backend = DictCacheBackend(cache)
        set_audio_cache_backend(backend)

        # Create transform and apply
        transform = AddBackgroundNoise(sounds_path=noise_dir, p=1.0)
        input_audio = np.random.randn(8000).astype(np.float32)

        output = transform(input_audio, sample_rate=sample_rate)

        assert output.dtype == np.float32
        assert output.shape == input_audio.shape
