import unittest

import numpy as np

from tesseract_nr.backend import (
    ArrayBackend,
    ParallelContext,
    apply_constant_2x2_exponential,
    exchange_theory3_halos,
)


class BackendTests(unittest.TestCase):
    def test_numpy_backend_round_trip(self):
        backend = ArrayBackend("numpy")
        value = backend.asarray([1, 2, 3])
        np.testing.assert_array_equal(backend.to_host(value), [1, 2, 3])
        self.assertFalse(backend.accelerated)

    def test_serial_partition_and_reduction(self):
        context = ParallelContext(enabled=False)
        partition = context.partition((11, 7, 5), axis=1)
        self.assertEqual(partition.local_shape, (11, 7, 5))
        self.assertEqual((partition.start, partition.stop), (0, 7))
        self.assertEqual(context.deterministic_sum(2.5), 2.5)

    def test_serial_halo_exchange_is_noop(self):
        context = ParallelContext(enabled=False)
        partition = context.partition((4,), 0)
        data = np.arange(8.0)
        before = data.copy()
        context.exchange_halos(data, partition, ghost_width=2)
        np.testing.assert_array_equal(data, before)

    def test_accelerator_ready_2x2_exponential_kernel(self):
        matrix = np.diag([-2.0, -0.5])
        values = np.stack((np.ones((3, 5)), 2.0 * np.ones((3, 5))))
        timestep = np.linspace(0.0, 0.4, 5)[None, :]
        result = apply_constant_2x2_exponential(matrix, values, timestep)
        np.testing.assert_allclose(
            result[0], np.broadcast_to(np.exp(-2.0 * timestep), (3, 5)), atol=2e-15
        )
        np.testing.assert_allclose(
            result[1],
            np.broadcast_to(2.0 * np.exp(-0.5 * timestep), (3, 5)),
            atol=2e-15,
        )

    def test_theory3_halo_bundle_uses_complete_field_contract(self):
        context = ParallelContext(enabled=False)
        partition = context.partition((8,), 0)
        fields = {
            "a": np.arange(36.0).reshape(3, 12),
            "longitudinal_A": np.arange(12.0),
            "target_current": np.arange(36.0).reshape(3, 12),
        }
        before = {name: value.copy() for name, value in fields.items()}
        exchange_theory3_halos(context, partition, 2, fields)
        for name in fields:
            np.testing.assert_array_equal(fields[name], before[name])


if __name__ == "__main__":
    unittest.main()
