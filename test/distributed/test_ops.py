import torch
import torch.distributed as dist
import torch.distributed.ops as ops

from torch.testing._internal.common_utils import TestCase, run_tests
from typing import Callable, Tuple
from unittest.mock import patch


class OpsTest(TestCase):
    def setUp(self) -> None:
        self._patch_get_rank = patch("torch.distributed.ops.dist.get_rank")
        self._patch_get_size = patch("torch.distributed.ops.dist.get_world_size")

        self._mock_get_rank_fn = self._patch_get_rank.start()
        self._mock_get_size_fn = self._patch_get_size.start()

        self._patch_all_reduce = patch("torch.distributed.ops.dist.all_reduce")
        self._patch_all_gather = patch("torch.distributed.ops.dist.all_gather")

        self._mock_all_reduce_fn = self._patch_all_reduce.start()
        self._mock_all_gather_fn = self._patch_all_gather.start()

        self._mock_all_reduce_fn.side_effect = self._mock_all_reduce
        self._mock_all_gather_fn.side_effect = self._mock_all_gather

    def tearDown(self) -> None:
        self._patch_get_rank.stop()
        self._patch_get_size.stop()

        self._patch_all_reduce.stop()
        self._patch_all_gather.stop()

    def _setup(self, rank: int, world_size: int, shape: Tuple[int, ...]) -> None:
        self._rank = rank  # Simulated rank.
        self._world_size = world_size  # Simulated world size.

        self._mock_get_rank_fn.return_value = rank
        self._mock_get_size_fn.return_value = world_size

        gen = torch.manual_seed(1)

        # Holds the inputs per rank.
        self._x = [
            torch.rand(shape, generator=gen, requires_grad=True) * 4.0 for _ in range(world_size)
        ]

        # Holds the intermediate activations per rank.
        self._activations = [torch.nn.functional.elu(x) for x in self._x]

        # Indicates whether we are in the forward or backward pass of the test.
        self._backprop = False

    def _mock_all_reduce(self, output, op, _) -> None:
        if self._backprop:
            inputs = [output.detach().clone()] * self._world_size
        else:
            inputs = self._activations

        with torch.no_grad():
            if op == dist.ReduceOp.SUM:
                output.zero_()
                for i in inputs:
                    output.add_(i)
            elif op == dist.ReduceOp.PRODUCT:
                output.fill_(1.0)
                for i in inputs:
                    output.mul_(i)
            else:
                output.zero_()

    def _mock_all_gather(self, outputs, input_, _) -> None:
        if self._backprop:
            inputs = [input_] * self._world_size
        else:
            inputs = self._activations

        with torch.no_grad():
            for o, i in zip(outputs, inputs):
                o.copy_(i)

    def _run_test(self, set_groundtruth: Callable, op: Callable) -> None:
        for world_size in range(1, 5):
            for rank in range(world_size):
                for shape in [(1,), (2, 3), (4, 4), (6, 4)]:
                    with self.subTest(rank=rank, world_size=world_size, shape=shape):
                        self._setup(rank, world_size, shape)

                        set_groundtruth()

                        self._run_and_assert_op(op)

    def _set_groundtruth_for_add(self) -> None:
        self._expected_output = torch.zeros_like(self._activations[0])
        for a in self._activations:
            self._expected_output.add_(a)

        self._compute_expected_derivatives()

    def _set_groundtruth_for_mul(self) -> None:
        self._expected_output = torch.ones_like(self._activations[0])
        for a in self._activations:
            self._expected_output.mul_(a)

        self._compute_expected_derivatives()

    def _set_groundtruth_for_min(self) -> None:
        self._expected_output = self._activations[0]
        for a in self._activations:
            self._expected_output = torch.minimum(self._expected_output, a)

        self._compute_expected_derivatives()

    def _set_groundtruth_for_max(self) -> None:
        self._expected_output = self._activations[0]
        for a in self._activations:
            self._expected_output = torch.maximum(self._expected_output, a)

        self._compute_expected_derivatives()

    def _compute_expected_derivatives(self) -> None:
        x = self._x[self._rank]

        # Simulate as if backprop (e.g. backward()) was perfomed by all ranks.
        all_output = self._world_size * self._expected_output

        jacobian = torch.autograd.grad(
            all_output, x, grad_outputs=torch.ones_like(all_output), create_graph=True
        )

        self._expected_jacobian = jacobian[0]

        # We need to retain the autograd graph in order to run backprop a second
        # time for assertion.
        hessian = torch.autograd.grad(
            jacobian[0], x, grad_outputs=torch.ones_like(jacobian[0]), retain_graph=True
        )

        self._expected_hessian = hessian[0]

    def _run_and_assert_op(self, op: Callable) -> None:
        output = op(self._activations[self._rank])

        self._assert_op(output)

    def _assert_op(self, output: torch.Tensor) -> None:
        self.assertEqual(output, self._expected_output)

        try:
            self._backprop = True

            self._assert_derivatives(output)
        finally:
            self._backprop = False

    def _assert_derivatives(self, output: torch.Tensor) -> None:
        x = self._x[self._rank]

        # Assert Jacobian
        jacobian = torch.autograd.grad(
            output, x, grad_outputs=torch.ones_like(output), create_graph=True
        )

        self.assertEqual(jacobian[0], self._expected_jacobian)

        # Assert Hessian.
        hessian = torch.autograd.grad(jacobian[0], x, grad_outputs=torch.ones_like(jacobian[0]))

        self.assertEqual(hessian[0], self._expected_hessian)

    def test_add_computes_output_and_derivatives_correctly(self) -> None:
        self._run_test(self._set_groundtruth_for_add, ops.add)

    def test_mul_computes_output_and_derivatives_correctly(self) -> None:
        self._run_test(self._set_groundtruth_for_mul, ops.mul)

    def test_min_computes_output_and_derivatives_correctly(self) -> None:
        self._run_test(self._set_groundtruth_for_min, ops.min)

    def test_max_computes_output_and_derivatives_correctly(self) -> None:
        self._run_test(self._set_groundtruth_for_max, ops.max)
