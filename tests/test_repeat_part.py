import numpy as np
from numpy.testing import assert_array_almost_equal

from audiomentations.augmentations.repeat_part import RepeatPart


class TestRepeatPart:
    # TODO: Test 2D

    def test_replace_one_repeat(self):
        augment = RepeatPart(mode="replace", crossfade=False, p=1.0)
        augment.parameters = {
            "should_apply": True,
            "part_num_samples": 3,
            "repeats": 1,
            "part_start_index": 1,
        }
        augment.freeze_parameters()

        samples = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32
        )
        processed_samples = augment(samples=samples, sample_rate=4000)
        assert_array_almost_equal(
            processed_samples,
            np.array(
                [0.0, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 1.0],
                dtype=np.float32,
            ),
        )
        assert processed_samples.shape == samples.shape
        assert processed_samples.dtype == np.float32

    def test_replace_one_repeat_near_end(self):
        augment = RepeatPart(mode="replace", crossfade=False, p=1.0)
        augment.parameters = {
            "should_apply": True,
            "part_num_samples": 3,
            "repeats": 1,
            "part_start_index": 7,
        }
        augment.freeze_parameters()

        samples = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32
        )
        processed_samples = augment(samples=samples, sample_rate=4000)
        assert_array_almost_equal(
            processed_samples,
            np.array(
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.7],
                dtype=np.float32,
            ),
        )
        assert processed_samples.shape == samples.shape
        assert processed_samples.dtype == np.float32

    def test_replace_two_repeats(self):
        augment = RepeatPart(mode="replace", crossfade=False, p=1.0)
        augment.parameters = {
            "should_apply": True,
            "part_num_samples": 3,
            "repeats": 2,
            "part_start_index": 1,
        }
        augment.freeze_parameters()

        samples = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32
        )
        processed_samples = augment(samples=samples, sample_rate=4000)
        assert_array_almost_equal(
            processed_samples,
            np.array(
                [0.0, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 1.0],
                dtype=np.float32,
            ),
        )
        assert processed_samples.shape == samples.shape
        assert processed_samples.dtype == np.float32

    def test_replace_many_repeats_exceed_input_length(self):
        augment = RepeatPart(mode="replace", crossfade=False, p=1.0)
        augment.parameters = {
            "should_apply": True,
            "part_num_samples": 3,
            "repeats": 9,
            "part_start_index": 1,
        }
        augment.freeze_parameters()

        samples = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32
        )
        processed_samples = augment(samples=samples, sample_rate=4000)
        assert_array_almost_equal(
            processed_samples,
            np.array(
                [0.0, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1],
                dtype=np.float32,
            ),
        )
        assert processed_samples.shape == samples.shape
        assert processed_samples.dtype == np.float32

    def test_insert_one_repeat(self):
        augment = RepeatPart(mode="insert", crossfade=False, p=1.0)
        augment.parameters = {
            "should_apply": True,
            "part_num_samples": 3,
            "repeats": 1,
            "part_start_index": 1,
        }
        augment.freeze_parameters()

        samples = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32
        )
        processed_samples = augment(samples=samples, sample_rate=4000)
        assert_array_almost_equal(
            processed_samples,
            np.array(
                [0.0, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                dtype=np.float32,
            ),
        )
        assert processed_samples.dtype == np.float32

    def test_insert_two_repeats(self):
        augment = RepeatPart(mode="insert", crossfade=False, p=1.0)
        augment.parameters = {
            "should_apply": True,
            "part_num_samples": 3,
            "repeats": 2,
            "part_start_index": 0,
        }
        augment.freeze_parameters()

        samples = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
        processed_samples = augment(samples=samples, sample_rate=4000)
        assert_array_almost_equal(
            processed_samples,
            np.array(
                [0.0, 0.1, 0.2, 0.0, 0.1, 0.2, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                dtype=np.float32,
            ),
        )
        assert processed_samples.dtype == np.float32
