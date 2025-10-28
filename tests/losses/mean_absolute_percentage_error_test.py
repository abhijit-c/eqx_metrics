from unittest import TestCase

import jax
import jax.numpy as jnp
import numpy as np

import jax_metrics as jm
from jax_metrics import types


class MeanAbsolutePercentageErrorTest(TestCase):
    #
    def test_basic(self):
        target = jnp.array([[1.0, 1.0], [0.9, 0.0]])
        preds = jnp.array([[1.0, 1.0], [1.0, 0.0]])

        # Using 'auto'/'sum_over_batch_size' reduction type.
        mape = jm.losses.MeanAbsolutePercentageError()
        result = mape(target=target, preds=preds)
        assert np.isclose(result, 2.78, rtol=0.01)

        # Calling with 'sample_weight'.
        assert np.isclose(
            mape(target=target, preds=preds, sample_weight=jnp.array([0.1, 0.9])),
            2.5,
            rtol=0.01,
        )

        # Using 'sum' reduction type.
        mape = jm.losses.MeanAbsolutePercentageError(reduction=jm.losses.Reduction.SUM)

        assert np.isclose(mape(target=target, preds=preds), 5.6, rtol=0.01)

        # Using 'none' reduction type.
        mape = jm.losses.MeanAbsolutePercentageError(reduction=jm.losses.Reduction.NONE)

        result = mape(target=target, preds=preds)
        assert jnp.all(np.isclose(result, [0.0, 5.6], rtol=0.01))

    #
    def test_function(self):
        rng = jax.random.PRNGKey(42)
        target = jax.random.randint(rng, shape=(2, 3), minval=0, maxval=2)
        preds = jax.random.uniform(rng, shape=(2, 3))
        target = target.astype(preds.dtype)
        loss = jm.losses.mean_absolute_percentage_error(target, preds)
        assert loss.shape == (2,)
        assert jnp.array_equal(
            loss,
            100
            * jnp.mean(
                jnp.abs((preds - target) / jnp.maximum(jnp.abs(target), types.EPSILON)),
                axis=-1,
            ),
        )
