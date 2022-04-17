import inspect

import jax
import jax.numpy as jnp
import pytest

import metrix as mtx
from metrix.metrics.accuracy import Accuracy
from metrix.metrics.utils import DataType


class TestAccuracy:
    def test_jit(self):
        N = 0

        @jax.jit
        def f(m, target, preds):
            nonlocal N
            N += 1
            return m.update(target=target, preds=preds)

        metric = Accuracy(num_classes=10).reset()
        target = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])[None, None, None, :]
        preds = jnp.array([0, 1, 2, 3, 0, 5, 6, 7, 0, 9])[None, None, None, :]

        metric = f(metric, target, preds)
        assert N == 1
        assert metric.compute() == 0.8

        metric = f(metric, target, preds)
        assert N == 1
        assert metric.compute() == 0.8

    def test_logits_preds(self):
        N = 0

        @jax.jit
        def f(m, target, preds):
            nonlocal N
            N += 1
            return m.update(target=target, preds=preds)

        metric = Accuracy().reset()
        target = jnp.array([0, 0, 1, 1, 1])
        preds = jnp.array(
            [
                [10.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [40.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
            ]
        )

        metric = f(metric, target, preds)
        assert N == 1
        assert metric.compute() == 0.8

        metric = f(metric, target, preds)
        # assert N == 1
        assert metric.compute() == 0.8
