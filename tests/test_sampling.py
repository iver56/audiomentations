import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_equal

from audiomentations.core.sampling import WeightedChoiceSampler


class TestWeightedChoiceSampler:
    def test_init_valid_weights(self):
        weights = [0.2, 0.8]
        sampler = WeightedChoiceSampler(weights=weights)
        assert sampler.num_items == 2
        assert_allclose(sampler.cdf, [0.2, 1.0])
        assert sampler.cdf[-1] == 1.0  # Ensure last element is exactly 1.0

    def test_init_valid_weights_needs_normalization(self):
        weights = [1, 4] # Sums to 5
        sampler = WeightedChoiceSampler(weights=weights)
        assert sampler.num_items == 2
        assert_allclose(sampler.cdf, [0.2, 1.0])
        assert sampler.cdf[-1] == 1.0

    def test_init_invalid_weights_none(self):
        with pytest.raises(ValueError, match="If 'weights' is None you must supply a positive 'num_items'."):
            WeightedChoiceSampler(weights=None)

    def test_init_invalid_weights_empty(self):
        with pytest.raises(ValueError, match="Sum of weights must be > 0"):
            WeightedChoiceSampler(weights=[])

    def test_init_invalid_weights_negative(self):
        with pytest.raises(ValueError, match="weights must be non-negative"):
            WeightedChoiceSampler(weights=[0.5, -0.1])

    def test_init_weights_all_zero(self):
        with pytest.raises(ValueError, match="Sum of weights must be > 0"):
            WeightedChoiceSampler(weights=[0.0, 0.0])

    def test_sample_single(self):
        weights = [0.1, 0.2, 0.7]
        sampler = WeightedChoiceSampler(weights=weights)
        sample_index = sampler.sample(size=1)[0]
        assert 0 <= sample_index < 3

    def test_sample_multiple(self):
        weights = [0.1, 0.2, 0.7]
        sampler = WeightedChoiceSampler(weights=weights)
        sample_indices = sampler.sample(size=100)
        assert len(sample_indices) == 100
        assert all(0 <= idx < 3 for idx in sample_indices)

    def test_sample_distribution(self):
        weights = [0.1, 0.3, 0.6]
        sampler = WeightedChoiceSampler(weights=weights)
        num_samples = 20000
        sample_indices = sampler.sample(size=num_samples)

        counts = np.bincount(sample_indices, minlength=len(weights))
        observed_proportions = counts / num_samples

        normalized_weights = np.array(weights) / np.sum(weights)

        # Use a tolerance suitable for statistical tests
        assert_allclose(observed_proportions, normalized_weights, atol=0.02)

    def test_sample_edge_case_one_hot_first(self):
        weights = [1.0, 0.0, 0.0]
        sampler = WeightedChoiceSampler(weights=weights)
        num_samples = 100
        sample_indices = sampler.sample(size=num_samples)
        assert all(idx == 0 for idx in sample_indices)

    def test_sample_edge_case_one_hot_middle(self):
        weights = [0.0, 1.0, 0.0]
        sampler = WeightedChoiceSampler(weights=weights)
        num_samples = 100
        sample_indices = sampler.sample(size=num_samples)
        assert all(idx == 1 for idx in sample_indices)

    def test_sample_edge_case_uniform(self):
        weights = [1.0, 1.0, 1.0]
        sampler = WeightedChoiceSampler(weights=weights)
        num_samples = 30000
        sample_indices = sampler.sample(size=num_samples)

        counts = np.bincount(sample_indices, minlength=len(weights))
        observed_proportions = counts / num_samples

        expected_proportions = np.array([1/3, 1/3, 1/3])

        assert_allclose(observed_proportions, expected_proportions, atol=0.02)

    def test_sample_distribution_uniform(self):
        # Use normalized weights for testing distribution
        weights = np.array([0.5, 0.5])
        sampler = WeightedChoiceSampler(weights=weights)
        num_samples = 10000
        sample_indices = sampler.sample(size=num_samples)
        
        counts = np.bincount(sample_indices, minlength=sampler.num_items)
        observed_proportions = counts / num_samples
        
        assert_allclose(observed_proportions, weights, atol=0.05)

    def test_sample_distribution_non_uniform(self):
         # Use normalized weights for testing distribution
        weights = np.array([0.1, 0.7, 0.2])
        sampler = WeightedChoiceSampler(weights=weights)
        num_samples = 20000
        sample_indices = sampler.sample(size=num_samples)

        counts = np.bincount(sample_indices, minlength=sampler.num_items)
        observed_proportions = counts / num_samples
        
        assert_allclose(observed_proportions, weights, atol=0.05)

    def test_sample_distribution_with_zero_weight(self):
         # Use normalized weights for testing distribution
        weights = np.array([0.5, 0.0, 0.5])
        sampler = WeightedChoiceSampler(weights=weights)
        num_samples = 10000
        sample_indices = sampler.sample(size=num_samples)

        counts = np.bincount(sample_indices, minlength=sampler.num_items)
        observed_proportions = counts / num_samples
        
        assert counts[1] == 0 # Index 1 should never be chosen
        assert_allclose(observed_proportions, weights, atol=0.05)

    def test_sample_single_weight(self):
        weights = [1.0]
        sampler = WeightedChoiceSampler(weights=weights)
        num_samples = 100
        sample_indices = sampler.sample(size=num_samples)
        assert_array_equal(sample_indices, np.zeros(num_samples, dtype=int))
        assert sampler.cdf[-1] == 1.0 
