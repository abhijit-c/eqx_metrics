import hypothesis as hp
import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import strategies as st

import jax_metrics as jm


class TestMSE:
    def test_mse_basic(self):
        target = np.random.randn(8, 20, 20)
        preds = np.random.randn(8, 20, 20)

        mse_tx = jm.metrics.MeanSquareError()
        mse_tx_value, mse_tx = mse_tx(target=target, preds=preds)

        # Calculate expected MSE using NumPy
        expected_mse = np.mean((preds - target) ** 2)
        assert np.isclose(np.array(mse_tx_value), expected_mse)

    def test_accumulative_mse(self):
        mse_tx = jm.metrics.MeanSquareError()

        all_targets = []
        all_preds = []

        for batch in range(2):
            target = np.random.randn(8, 5, 5)
            preds = np.random.randn(8, 5, 5)

            mse_tx = mse_tx.update(target=target, preds=preds)
            all_targets.append(target)
            all_preds.append(preds)

        # Calculate expected MSE across all batches
        all_targets = np.concatenate(all_targets, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        expected_mse = np.mean((all_preds - all_targets) ** 2)

        assert np.isclose(
            np.array(mse_tx.compute()),
            expected_mse,
        )

    def test_mse_short(self):
        target = np.random.randn(8, 20, 20)
        preds = np.random.randn(8, 20, 20)

        mse_tx_long = (
            jm.metrics.MeanSquareError().update(target=target, preds=preds).compute()
        )
        mse_tx_short = jm.metrics.MSE().update(target=target, preds=preds).compute()
        assert np.isclose(np.array(mse_tx_long), np.array(mse_tx_short))

    @hp.given(
        use_sample_weight=st.booleans(),
    )
    @hp.settings(deadline=None, max_examples=10)
    def test_mse_weights(self, use_sample_weight):
        target = np.random.randn(8, 20, 20)
        preds = np.random.randn(8, 20, 20)

        params = {"target": target, "preds": preds}
        mse_tx = jm.metrics.MeanSquareError()

        if use_sample_weight:
            sample_weight = np.random.choice([0, 1], 8)
            while sample_weight.sum() == 0:
                sample_weight = np.random.choice([0, 1], 8)
            params.update({"sample_weight": sample_weight})

        mse_tx_value, mse_tx = mse_tx(**params)

        if use_sample_weight:
            target, preds = target[sample_weight == 1], preds[sample_weight == 1]

        mse_tx = jm.metrics.MeanSquareError()
        mse_tx_no_sample_weight, mse_tx = mse_tx(**params)

        assert np.isclose(mse_tx_value, mse_tx_no_sample_weight)

    @hp.given(
        use_sample_weight=st.booleans(),
    )
    @hp.settings(deadline=None, max_examples=10)
    def test_mse_weights_values_dim(self, use_sample_weight):
        target = np.random.randn(8, 20, 20)
        preds = np.random.randn(8, 20, 20)

        params = {"target": target, "preds": preds}
        if use_sample_weight:
            sample_weight = np.random.choice([0, 1], 8 * 20).reshape((8, 20))
            params.update({"sample_weight": sample_weight})

        mse_tx, _ = jm.metrics.MeanSquareError()(**params)

        assert isinstance(mse_tx, jax.Array)
