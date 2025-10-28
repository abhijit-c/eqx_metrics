import jax
import jax.numpy as jnp
import numpy as np

import jax_metrics as jm

# import debugpy

# print("Waiting for debugger...")
# debugpy.listen(5679)
# debugpy.wait_for_client()


def test_basic():
    target = jnp.array([[0, 1], [0, 0]])
    preds = jnp.array([[0.6, 0.4], [0.4, 0.6]])

    # Using 'auto'/'sum_over_batch_size' reduction type.
    huber_loss = jm.losses.Huber()
    assert huber_loss(target=target, preds=preds) == 0.155

    # Calling with 'sample_weight'.
    assert (
        huber_loss(target=target, preds=preds, sample_weight=jnp.array([0.8, 0.2]))
        == 0.08500001
    )

    # Using 'sum' reduction type.
    huber_loss = jm.losses.Huber(reduction=jm.losses.Reduction.SUM)
    assert huber_loss(target=target, preds=preds) == 0.31

    # Using 'none' reduction type.
    huber_loss = jm.losses.Huber(reduction=jm.losses.Reduction.NONE)

    assert jnp.equal(
        huber_loss(target=target, preds=preds), jnp.array([0.18, 0.13000001])
    ).all()


def test_function():
    rng = jax.random.PRNGKey(42)

    target = jax.random.randint(rng, shape=(2, 3), minval=0, maxval=2)
    preds = jax.random.uniform(rng, shape=(2, 3))

    loss = jm.losses.huber(target, preds, delta=1.0)
    assert loss.shape == (2,)

    preds = preds.astype(float)
    target = target.astype(float)
    delta = 1.0
    error = jnp.subtract(preds, target)
    abs_error = jnp.abs(error)
    quadratic = jnp.minimum(abs_error, delta)
    linear = jnp.subtract(abs_error, quadratic)
    assert jnp.array_equal(
        loss,
        jnp.mean(
            jnp.add(
                jnp.multiply(0.5, jnp.multiply(quadratic, quadratic)),
                jnp.multiply(delta, linear),
            ),
            axis=-1,
        ),
    )


if __name__ == "__main__":
    test_basic()
    test_function()
