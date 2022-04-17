from unittest import TestCase

import jax.numpy as jnp
import numpy as np
import pytest

import jax_metrics as jm


class LossTest(TestCase):
    def test_basic(self):
        class MAE(jm.Loss):
            def call(self, target, preds):
                return jnp.abs(target - preds)

        target = jnp.array([1.0, 2.0, 3.0])
        preds = jnp.array([2.0, 3.0, 4.0])

        mae = MAE()

        sample_loss = mae.call(target, preds)
        loss = mae(target=target, preds=preds)

        assert jnp.alltrue(sample_loss == jnp.array([1.0, 1.0, 1.0]))
        assert loss == 1

    def test_slice(self):
        class MAE(jm.Loss):
            def call(self, target, preds):
                return jnp.abs(target - preds)

        target = dict(a=jnp.array([1.0, 2.0, 3.0]))
        preds = dict(a=jnp.array([2.0, 3.0, 4.0]))

        mae = MAE().index_into(target="a", preds="a")

        # raises because it doesn't use kwargs
        with pytest.raises(BaseException):
            sample_loss = mae(target, preds)

        loss = mae(target=target, preds=preds)

        assert loss == 1

        # test using call, no reduction is performed
        sample_loss = mae.call(target=target, preds=preds)
        assert sample_loss.shape == (3,)
