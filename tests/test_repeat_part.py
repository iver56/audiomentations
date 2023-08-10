from copy import deepcopy

import numpy as np
from numpy.testing import assert_array_almost_equal

from audiomentations import Gain, AdjustDuration
from audiomentations.augmentations.repeat_part import RepeatPart


def adapt_ndim(samples, ndim):
    if samples.ndim < ndim:
        samples = samples[np.newaxis, :]
    return samples


class TestRepeatPart:
    # TODO: Test crossfading

    def test_replace_one_repeat(self):
        augment = RepeatPart(mode="replace", crossfade=False, p=1.0)
        augment.parameters = {
            "should_apply": True,
            "part_num_samples": 3,
            "repeats": 1,
            "part_start_index": 1,
        }
        augment.freeze_parameters()

        for ndim in (1, 2):
            samples = np.array(
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                dtype=np.float32,
            )
            samples = adapt_ndim(samples, ndim)
            processed_samples = augment(samples=samples, sample_rate=4000)
            target_samples = np.array(
                [0.0, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 1.0],
                dtype=np.float32,
            )
            target_samples = adapt_ndim(target_samples, ndim)
            assert_array_almost_equal(processed_samples, target_samples)
            assert processed_samples.shape == samples.shape
            assert processed_samples.dtype == np.float32

    def test_replace_one_repeat_transformed(self):
        augment = RepeatPart(
            mode="replace",
            crossfade=False,
            part_transform=Gain(min_gain_db=-6.0, max_gain_db=-6.0, p=1.0),
            p=1.0,
        )
        part_gain_factor = 0.5011872336272722
        augment.part_transform.parameters = {
            "should_apply": True,
            "amplitude_ratio": part_gain_factor,  # -6 dB
        }
        augment.parameters = {
            "should_apply": True,
            "part_num_samples": 3,
            "repeats": 1,
            "part_start_index": 1,
        }
        augment.freeze_parameters()

        for ndim in (1, 2):
            samples = np.array(
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                dtype=np.float32,
            )
            samples = adapt_ndim(samples, ndim)

            processed_samples = augment(samples=samples, sample_rate=4000)

            target_samples = np.array(
                [
                    0.0,
                    0.1,
                    0.2,
                    0.3,
                    0.1 * part_gain_factor,
                    0.2 * part_gain_factor,
                    0.3 * part_gain_factor,
                    0.7,
                    0.8,
                    0.9,
                    1.0,
                ],
                dtype=np.float32,
            )
            target_samples = adapt_ndim(target_samples, ndim)
            assert_array_almost_equal(processed_samples, target_samples)
            assert processed_samples.shape == samples.shape
            assert processed_samples.dtype == np.float32

    def test_freeze_and_unfreeze_part_transform_parameters(self):
        augment = RepeatPart(
            min_part_duration=0.1,
            max_part_duration=0.2,
            part_transform=Gain(p=1.0),
            crossfade=False,
            p=1.0,
        )
        dummy_samples = np.zeros(40, dtype=np.float32)
        augment(dummy_samples, 40)
        params1 = deepcopy(augment.part_transform.parameters)
        augment.freeze_parameters()
        augment(dummy_samples, 40)
        params2 = deepcopy(augment.part_transform.parameters)
        assert params1 == params2

        augment.unfreeze_parameters()
        augment(dummy_samples, 40)
        params3 = deepcopy(augment.part_transform.parameters)
        assert params3 != params2

    def test_replace_one_repeat_near_end(self):
        augment = RepeatPart(mode="replace", crossfade=False, p=1.0)
        augment.parameters = {
            "should_apply": True,
            "part_num_samples": 3,
            "repeats": 1,
            "part_start_index": 7,
        }
        augment.freeze_parameters()

        for ndim in (1, 2):
            samples = np.array(
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                dtype=np.float32,
            )
            samples = adapt_ndim(samples, ndim)
            processed_samples = augment(samples=samples, sample_rate=4000)
            target_samples = np.array(
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.7],
                dtype=np.float32,
            )
            target_samples = adapt_ndim(target_samples, ndim)
            assert_array_almost_equal(processed_samples, target_samples)
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

        for ndim in (1, 2):
            samples = np.array(
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                dtype=np.float32,
            )
            samples = adapt_ndim(samples, ndim)
            processed_samples = augment(samples=samples, sample_rate=4000)
            target_samples = np.array(
                [0.0, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 1.0],
                dtype=np.float32,
            )
            target_samples = adapt_ndim(target_samples, ndim)
            assert_array_almost_equal(processed_samples, target_samples)
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

        for ndim in (1, 2):
            samples = np.array(
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                dtype=np.float32,
            )
            samples = adapt_ndim(samples, ndim)
            processed_samples = augment(samples=samples, sample_rate=4000)
            target_samples = np.array(
                [0.0, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1],
                dtype=np.float32,
            )
            target_samples = adapt_ndim(target_samples, ndim)
            assert_array_almost_equal(processed_samples, target_samples)
            assert processed_samples.shape == samples.shape
            assert processed_samples.dtype == np.float32

    def test_replace_many_transformed_repeats_exceed_input_length(self):
        augment = RepeatPart(
            mode="replace",
            crossfade=False,
            part_transform=Gain(min_gain_db=-6.0, max_gain_db=-6.0, p=1.0),
            p=1.0,
        )
        part_gain_factor = 0.5011872336272722
        augment.part_transform.parameters = {
            "should_apply": True,
            "amplitude_ratio": part_gain_factor,  # -6 dB
        }
        augment.parameters = {
            "should_apply": True,
            "part_num_samples": 3,
            "repeats": 9,
            "part_start_index": 1,
        }
        augment.freeze_parameters()

        for ndim in (1, 2):
            samples = np.array(
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                dtype=np.float32,
            )
            samples = adapt_ndim(samples, ndim)
            processed_samples = augment(samples=samples, sample_rate=4000)
            target_samples = np.array(
                [0.0, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1],
                dtype=np.float32,
            )
            target_samples[4:] *= part_gain_factor
            target_samples = adapt_ndim(target_samples, ndim)
            assert_array_almost_equal(processed_samples, target_samples)
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

    def test_insert_two_elongated_repeats(self):
        augment = RepeatPart(
            mode="insert",
            crossfade=False,
            part_transform=AdjustDuration(duration_samples=6, p=1.0),
            p=1.0,
        )
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
                [
                    0.0,
                    0.1,
                    0.2,
                    0.0,  # first repetition starts here
                    0.1,
                    0.2,
                    0.0,  # start of zero padding
                    0.0,
                    0.0,  # end of zero padding
                    0.0,  # second repetition starts here
                    0.1,
                    0.2,
                    0.0,  # start of zero padding
                    0.0,
                    0.0,  # end of zero padding
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                ],
                dtype=np.float32,
            ),
        )
        assert processed_samples.dtype == np.float32
