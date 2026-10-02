import random

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from audiomentations import Compose, Gain, OneOf, Reverse, Shift, SomeOf


@pytest.fixture(autouse=True)
def restore_random_state():
    state = random.getstate()
    numpy_state = np.random.get_state()
    yield
    random.setstate(state)
    np.random.set_state(numpy_state)


@pytest.fixture
def samples():
    return np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)


def make_shuffled_compose(p=1.0):
    return Compose(
        [
            Reverse(p=1.0),
            Shift(
                min_shift=1,
                max_shift=1,
                shift_unit="samples",
                fade_duration=0.0,
                p=1.0,
            ),
        ],
        p=p,
        shuffle=True,
    )


@pytest.mark.parametrize("freeze_before_call", [False, True])
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
def test_freeze_preserves_apply_decision(samples, p, seed, freeze_before_call):
    augmenter = Compose([Gain(min_gain_db=-6, max_gain_db=-6, p=1.0)], p=p)
    if freeze_before_call:
        augmenter.freeze_parameters()
    random.seed(seed)
    expected = augmenter(samples=samples, sample_rate=16000)
    should_apply = p == 1.0 or (p == 0.5 and seed == 1)
    assert_allclose(expected, samples * (10 ** (-6 / 20) if should_apply else 1))
    augmenter.freeze_parameters()

    for repeat_seed in range(10):
        random.seed(repeat_seed)
        assert_array_equal(augmenter(samples, 16000), expected)
        assert_array_equal(augmenter(samples * 0.5, 16000), expected * 0.5)


@pytest.mark.parametrize("freeze_before_call", [False, True])
@pytest.mark.parametrize("seed", [0, 1])
def test_freeze_preserves_shuffle_order(samples, seed, freeze_before_call):
    augmenter = make_shuffled_compose()
    original_transforms = augmenter.transforms.copy()
    if freeze_before_call:
        augmenter.freeze_parameters()
    random.seed(seed)
    expected = augmenter(samples, 16000)
    possible_outputs = [np.roll(samples[::-1], 1), np.roll(samples, 1)[::-1]]
    assert any(np.array_equal(expected, output) for output in possible_outputs)
    assert not np.array_equal(*possible_outputs)
    augmenter.freeze_parameters()

    for repeat_seed in range(10):
        random.seed(repeat_seed)
        assert_array_equal(augmenter(samples, 16000), expected)
    assert augmenter.transforms == original_transforms


def test_unfreeze_resumes_apply_and_shuffle_randomization(samples):
    augmenter = make_shuffled_compose(p=0.5)
    random.seed(1)
    expected = augmenter(samples, 16000)
    augmenter.freeze_parameters()
    assert_array_equal(augmenter(samples, 16000), expected)
    augmenter.unfreeze_parameters()

    outputs = set()
    for seed in range(20):
        random.seed(seed)
        outputs.add(tuple(augmenter(samples, 16000)))
    assert outputs == {
        tuple(samples),
        tuple(np.roll(samples[::-1], 1)),
        tuple(np.roll(samples, 1)[::-1]),
    }


def test_freeze_only_composition_leaves_child_random(samples):
    child = Gain(min_gain_db=-12, max_gain_db=-6, p=1.0)
    augmenter = Compose([child], p=0.5)
    random.seed(1)
    augmenter(samples, 16000)
    augmenter.freeze_parameters(apply_to_children=False)

    outputs = set()
    for seed in range(10):
        random.seed(seed)
        output = augmenter(samples, 16000)
        assert not np.array_equal(output, samples)
        outputs.add(tuple(output))
    assert len(outputs) > 1
    assert not child.are_parameters_frozen


def test_unfreeze_only_composition_keeps_child_frozen(samples):
    child = Gain(min_gain_db=-12, max_gain_db=-6, p=1.0)
    augmenter = Compose([child], p=0.5)
    random.seed(1)
    expected = augmenter(samples, 16000)
    augmenter.freeze_parameters()
    augmenter.unfreeze_parameters(apply_to_children=False)

    outputs = set()
    for seed in range(10):
        random.seed(seed)
        outputs.add(tuple(augmenter(samples, 16000)))
    assert outputs == {tuple(samples), tuple(expected)}
    assert child.are_parameters_frozen


@pytest.mark.parametrize("apply_to_children", [False, True])
def test_explicit_randomization_updates_frozen_composition(samples, apply_to_children):
    child = Gain(min_gain_db=-12, max_gain_db=-6, p=1.0)
    augmenter = Compose([child], p=0.5)
    random.seed(1)
    previous = augmenter(samples, 16000)
    augmenter.freeze_parameters()

    # The two seeds select both applying and skipping, also when the child draws first.
    outputs = set()
    for seed in (0, 1):
        random.seed(seed)
        augmenter.randomize_parameters(
            samples, 16000, apply_to_children=apply_to_children
        )
        expected = augmenter(samples, 16000)
        outputs.add(tuple(expected))
        if not apply_to_children and not np.array_equal(expected, samples):
            assert_array_equal(expected, previous)
        for repeat_seed in range(5):
            random.seed(repeat_seed)
            assert_array_equal(augmenter(samples, 16000), expected)
    assert tuple(samples) in outputs
    assert len(outputs) == 2


@pytest.mark.parametrize("outer_type", [Compose, OneOf, SomeOf])
def test_freeze_nested_composition(samples, outer_type):
    inner = make_shuffled_compose(p=0.5)
    if outer_type is SomeOf:
        augmenter = SomeOf(1, [inner])
    else:
        augmenter = outer_type([inner])

    # Cover both the skipped and applied inner composition before freezing.
    outputs = set()
    for seed in range(10):
        augmenter.unfreeze_parameters()
        random.seed(seed)
        expected = augmenter(samples, 16000)
        outputs.add(tuple(expected))
        augmenter.freeze_parameters()
        for repeat_seed in range(5):
            random.seed(repeat_seed)
            assert_array_equal(augmenter(samples, 16000), expected)
    assert len(outputs) == 3


def test_freeze_nested_compose_before_first_call(samples):
    augmenter = Compose([make_shuffled_compose(p=0.5)])
    augmenter.freeze_parameters()
    random.seed(4)
    expected = augmenter(samples, 16000)
    assert not np.array_equal(expected, samples)
    for seed in range(10):
        random.seed(seed)
        assert_array_equal(augmenter(samples, 16000), expected)
