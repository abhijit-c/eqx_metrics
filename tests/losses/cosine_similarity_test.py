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
    target = jnp.array([[0.0, 1.0], [1.0, 1.0]])
    preds = jnp.array([[1.0, 0.0], [1.0, 1.0]])

    # Using 'auto'/'sum_over_batch_size' reduction type.
    cosine_loss = jm.losses.CosineSimilarity(axis=1)
    assert cosine_loss(target=target, preds=preds) == -0.49999997

    # Calling with 'sample_weight'.
    assert (
        cosine_loss(target=target, preds=preds, sample_weight=jnp.array([0.8, 0.2]))
        == -0.099999994
    )

    # Using 'sum' reduction type.
    cosine_loss = jm.losses.CosineSimilarity(axis=1, reduction=jm.losses.Reduction.SUM)
    assert cosine_loss(target=target, preds=preds) == -0.99999994

    # Using 'none' reduction type.
    cosine_loss = jm.losses.CosineSimilarity(axis=1, reduction=jm.losses.Reduction.NONE)

    assert jnp.equal(
        cosine_loss(target=target, preds=preds), jnp.array([-0.0, -0.99999994])
    ).all()


def test_function():
    rng = jax.random.PRNGKey(42)

    target = jax.random.randint(rng, shape=(2, 3), minval=0, maxval=2)
    preds = jax.random.uniform(rng, shape=(2, 3))

    loss = jm.losses.cosine_similarity(target, preds, axis=1)
    assert loss.shape == (2,)

    target = target / jnp.maximum(
        jnp.linalg.norm(target, axis=1, keepdims=True), jnp.sqrt(types.EPSILON)
    )
    preds = preds / jnp.maximum(
        jnp.linalg.norm(preds, axis=1, keepdims=True), jnp.sqrt(types.EPSILON)
    )
    assert jnp.array_equal(loss, -jnp.sum(target * preds, axis=1))


if __name__ == "__main__":
    test_basic()
    test_function()
