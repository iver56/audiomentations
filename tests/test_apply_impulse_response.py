import os
import pickle

import numpy as np
import pytest

from audiomentations import ApplyImpulseResponse
from audiomentations.core.composition import Compose
from demo.demo import DEMO_DIR


class TestImpulseResponse:
    def test_apply_impulse_response(self):
        sample_len = 1024
        samples_in = np.random.normal(0, 1, size=sample_len).astype(np.float32)
        sample_rate = 16000

        add_ir_transform = ApplyImpulseResponse(
            ir_path=os.path.join(DEMO_DIR, "ir"), p=1.0
        )

        # Check that misc_file.txt is not one of the IR file candidates, as it's not audio
        assert len(add_ir_transform.ir_files) == 1

        augmenter = Compose([add_ir_transform])

        assert len(samples_in) == sample_len
        samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)

        # Check parameters
        assert augmenter.transforms[0].parameters["should_apply"]
        assert augmenter.transforms[0].parameters["ir_file_path"] == os.path.join(
            DEMO_DIR, "ir", "impulse_response_0.wav"
        )

        assert samples_out.dtype == np.float32
        assert len(samples_out) > len(samples_in)

    def test_apply_impulse_response_multi_channel(self):
        sample_len = 1024
        samples_in = np.random.normal(0, 1, size=sample_len).astype(np.float32)
        sample_rate = 16000

        samples_in = np.expand_dims(samples_in, axis=0)
        samples_in = np.tile(samples_in, (2, 1))

        add_ir_transform = ApplyImpulseResponse(
            ir_path=os.path.join(DEMO_DIR, "ir"), p=1.0
        )

        # Check that misc_file.txt is not one of the IR file candidates, as it's not audio
        assert len(add_ir_transform.ir_files) == 1

        augmenter = Compose([add_ir_transform])

        assert samples_in.shape[1] == sample_len
        samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)

        # Check parameters
        assert augmenter.transforms[0].parameters["should_apply"]
        assert augmenter.transforms[0].parameters["ir_file_path"] == os.path.join(
            DEMO_DIR, "ir", "impulse_response_0.wav"
        )

        assert samples_out.dtype == np.float32
        assert samples_out.shape[1] > len(samples_in)

    def test_leave_length_unchanged(self):
        sample_len = 1024
        samples_in = np.random.normal(0, 1, size=sample_len).astype(np.float32)
        sample_rate = 16000

        add_ir_transform = ApplyImpulseResponse(
            ir_path=os.path.join(DEMO_DIR, "ir"), p=1.0, leave_length_unchanged=True
        )

        samples_out = add_ir_transform(samples=samples_in, sample_rate=sample_rate)

        assert samples_out.dtype == np.float32
        assert len(samples_out) == len(samples_in)

    def test_picklability(self):
        add_ir_transform = ApplyImpulseResponse(
            ir_path=os.path.join(DEMO_DIR, "ir"), p=1.0
        )
        pickled = pickle.dumps(add_ir_transform)
        unpickled = pickle.loads(pickled)
        assert add_ir_transform.ir_files == unpickled.ir_files
