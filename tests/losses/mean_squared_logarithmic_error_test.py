import jax
import jax.numpy as jnp
import numpy as np

import jax_metrics as jm
from jax_metrics import types

# import debugpy

# print("Waiting for debugger...")
# debugpy.listen(5679)
# debugpy.wait_for_client()


def test_basic():
    target = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    preds = jnp.array([[1.0, 1.0], [1.0, 0.0]])

    # Using 'auto'/'sum_over_batch_size' reduction type.
    msle = jm.losses.MeanSquaredLogarithmicError()

    assert msle(target=target, preds=preds) == 0.24022643

    # Calling with 'sample_weight'.
    assert (
        msle(target=target, preds=preds, sample_weight=jnp.array([0.7, 0.3]))
        == 0.12011322
    )

    # Using 'sum' reduction type.
    msle = jm.losses.MeanSquaredLogarithmicError(reduction=jm.losses.Reduction.SUM)

    assert msle(target=target, preds=preds) == 0.48045287

    # Using 'none' reduction type.
    msle = jm.losses.MeanSquaredLogarithmicError(reduction=jm.losses.Reduction.NONE)

    assert jnp.equal(
        msle(target=target, preds=preds), jnp.array([0.24022643, 0.24022643])
    ).all()


def test_function():
    rng = jax.random.PRNGKey(42)

    target = jax.random.randint(rng, shape=(2, 3), minval=0, maxval=2)
    preds = jax.random.uniform(rng, shape=(2, 3))

    loss = jm.losses.mean_squared_logarithmic_error(target, preds)

    assert loss.shape == (2,)

    first_log = jnp.log(jnp.maximum(target, types.EPSILON) + 1.0)
    second_log = jnp.log(jnp.maximum(preds, types.EPSILON) + 1.0)
    assert jnp.array_equal(loss, jnp.mean(jnp.square(first_log - second_log), axis=-1))


if __name__ == "__main__":
    test_basic()
    test_function()
