from abc import ABC, abstractmethod
import torch
from torch.autograd import Function
from typing import Any, List, Optional, Tuple
import torch.distributed as dist


def add(
    input_: torch.Tensor, group: Optional[dist.ProcessGroup] = dist.group.WORLD
) -> torch.Tensor:
    return _DistributedAdd.apply(input_, group)


class _DistributedAdd(Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, input_: torch.Tensor, group: dist.ProcessGroup
    ) -> torch.Tensor:
        ctx.group = group

        output = input_.clone()

        dist.all_reduce(output, dist.ReduceOp.SUM, group)

        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:  # type: ignore[override]
        grad_input = add(grad_output, ctx.group)

        return (grad_input, None)


def mul(
    input_: torch.Tensor, group: Optional[dist.ProcessGroup] = dist.group.WORLD
) -> torch.Tensor:
    return _DistributedMul.apply(input_, group)


multiply = mul


class _DistributedMul(Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, input_: torch.Tensor, group: dist.ProcessGroup
    ) -> torch.Tensor:
        ctx.group = group

        output = input_.clone()

        dist.all_reduce(output, dist.ReduceOp.PRODUCT, group)

        ctx.prod_of_others = output / input_

        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:  # type: ignore[override]
        total_grad_output = add(grad_output, ctx.group)

        grad_input = total_grad_output * ctx.prod_of_others

        return (grad_input, None)


def minimum(
    input_: torch.Tensor, group: Optional[dist.ProcessGroup] = dist.group.WORLD
) -> torch.Tensor:
    return _DistributedMixMax.apply(input_, group, dist.ReduceOp.MIN)


min = minimum


def maximum(
    input_: torch.Tensor, group: Optional[dist.ProcessGroup] = dist.group.WORLD
) -> torch.Tensor:
    return _DistributedMixMax.apply(input_, group, dist.ReduceOp.MAX)


max = maximum


class _DistributedMixMax(Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, input_: torch.Tensor, group: dist.ProcessGroup, op: dist.ReduceOp
    ) -> torch.Tensor:
        ctx.group = group

        group_size = dist.get_world_size(group)

        # Gather all inputs to compute the minimum or maximum.
        inputs = [torch.zeros_like(input_) for _ in range(group_size)]

        dist.all_gather(inputs, input_, group)

        # Compute the minimum or maximum.
        output = inputs[0].clone()
        for inp in inputs[1:]:
            if op == dist.ReduceOp.MIN:
                torch.minimum(output, inp, out=output)
            else:
                torch.maximum(output, inp, out=output)

        # We use the inputs of other ranks to compute the gradients in the
        # backward pass.
        rank = dist.get_rank(group)

        del inputs[rank]

        ctx.other_inputs = inputs

        ctx.save_for_backward(input_, output)

        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:  # type: ignore[override]
        input_: torch.Tensor
        output: torch.Tensor

        input_, output = ctx.saved_tensors

        # Collect the gradients of all ranks and accumulate them.
        grad_input = add(grad_output, ctx.group)

        # If there are other ranks that have the same minima or maxima, we have
        # to scale down the corresponding gradients.
        scale = torch.ones_like(grad_input)
        for other_inp in ctx.other_inputs:
            scale += torch.where(input_ == other_inp, 1, 0)

        grad_input /= scale

        # Zero out all gradients for which our input is not the minimum or
        # maximum.
        grad_input.masked_fill_(input_ != output, 0)

        return (grad_input, None, None)
