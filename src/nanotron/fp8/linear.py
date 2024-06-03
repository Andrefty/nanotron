from dataclasses import dataclass
from typing import Optional, Tuple, Union, cast

import torch
import transformer_engine as te  # noqa
from torch import nn

from nanotron import constants
from nanotron.fp8.constants import FP8LM_RECIPE, QTYPE_TO_DTYPE
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.kernel import correct_fp8_matmul_kernel, fp8_matmul_kernel
from nanotron.fp8.meta import FP8Meta
from nanotron.fp8.parameter import FP8Parameter
from nanotron.fp8.tensor import FP8Tensor

# import torch.nn.functional as F


@dataclass
class FP8LinearMeta:
    """FP8 metadata for FP8Linear."""

    input: Optional[FP8Meta] = None
    weight: Optional[FP8Meta] = None
    input_grad: Optional[FP8Meta] = None
    weight_grad: Optional[FP8Meta] = None
    # output_grad: Optional[FP8Meta] = None


class FP8Linear(nn.Linear):
    # TODO(xrsrke): qtype isn't the data types of the weight and bias
    # but the accumulation precision dtype
    # chanege it to accum_dtype
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        accum_qtype: DTypes = FP8LM_RECIPE.linear.accum_dtype,
        # NOTE: placeholder for dtype in torch's nn.Linear
        # TODO(xrsrke): remove this shit
        **kwargs,
    ):
        """
        Args:
            qtype (DTypes, optional): This is accumulation precision dtype
        """

        assert device != torch.device("cpu"), "FP8Linear only supports CUDA tensors"
        assert accum_qtype in DTypes

        super().__init__(in_features, out_features, bias, device, QTYPE_TO_DTYPE[accum_qtype])
        # TODO(xrsrke): don't fixed dtype, take it from the FP8 recipe
        # DTypes.FP8E4M3
        quant_w = FP8Parameter(
            self.weight.data, dtype=FP8LM_RECIPE.linear.weight.dtype, interval=FP8LM_RECIPE.linear.weight.interval
        )
        assert quant_w.dtype in [torch.uint8, torch.int8], f"got {self.weight.data.dtype}"
        self.weight = quant_w
        # assert self.weight.data.orig_data.abs().max() == quant_w.fp8_meta.amax

        assert self.weight.data.dtype in [torch.uint8, torch.int8], f"got {self.weight.data.dtype}"
        self.metadatas = FP8LinearMeta()
        self.accum_qtype = accum_qtype

    def forward(self, input: Union[FP8Tensor, torch.Tensor]) -> torch.Tensor:
        import nanotron.fp8.functional as F

        # # NOTE: only do fp8 kernel if both input and weight are on CUDA device
        # if input.device == torch.device("cpu") or self.weight.device == torch.device("cpu"):
        #     # TODO(xrsrke): adjust the accumation precision
        #     return F.linear(input, self.weight, self.bias)

        return F.linear(input, self.weight.data, self.bias, self.accum_qtype, self.metadatas)

    def __repr__(self) -> str:
        return f"FP8{super().__repr__()}"


class _FP8Matmul(torch.autograd.Function):
    @staticmethod
    @torch.no_grad()
    def forward(
        ctx,
        input: Union[FP8Tensor, torch.Tensor],
        weight: FP8Tensor,
        output: torch.Tensor,
        phony: torch.Tensor,
        metadatas: FP8LinearMeta,
        accum_qtype: DTypes,
        name: Optional[str] = None  # TODO(xrsrke): remove this shit after debugging
        # is_weight_transposed: bool = False,
    ) -> torch.Tensor:
        assert not isinstance(input, FP8Tensor)

        # NOTE: pad input shape to disibile by 16
        # input = F.pad(input, (0, 0, 0, 16 - input.shape[1] % 16))

        if metadatas.input is None:
            fp8_input = FP8Tensor(
                input,
                dtype=FP8LM_RECIPE.linear.input.dtype,
                interval=FP8LM_RECIPE.linear.input.interval,
                # is_delayed_scaling=FP8LM_RECIPE.linear.input.is_delayed_scaling,
            )
            metadatas.input = fp8_input.fp8_meta
        else:
            fp8_input = FP8Tensor.from_metadata(input, metadatas.input)

        ctx.accum_qtype = accum_qtype
        ctx.metadatas = metadatas
        # ctx.is_weight_transposed = is_weight_transposed
        ctx.save_for_backward(fp8_input, weight)
        ctx.name = name

        # if is_weight_transposed is False:
        #     weight_temp = weight.transpose_fp8()
        # else:
        #     weight_temp = weight

        output = fp8_matmul_kernel(
            # NOTE: that works
            mat_a=weight,
            transpose_a=True,
            mat_b=fp8_input,
            transpose_b=False,
            output=output,
            use_split_accumulator=FP8LM_RECIPE.linear.split_accumulator.output,
            accum_qtype=accum_qtype,
        )

        # output = _fp8_matmul_kernel_2(
        #     # NOTE: that works
        #     mat_a=input,
        #     transpose_a=False,
        #     mat_b=weight.data,
        #     transpose_b=True,
        #     use_split_accumulator=False,
        #     accum_qtype=accum_qtype,
        # )

        return output, phony

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_output: torch.Tensor, grad_phony: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        """
        ∂L/∂X = ∂L/∂Y @ Wᵀ
        ∂L/∂W = Xᵀ @ ∂L/∂Y
        Reference: https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html
        """
        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        # assert isinstance(grad_output, torch.Tensor)
        assert grad_output.__class__ == torch.Tensor
        assert grad_output.dtype == torch.float16

        if ctx.name == "mlp.down_proj":
            assert 1 == 1

        if ctx.name == "lm_head":
            assert 1 == 1

        fp8_input, fp8_weight = ctx.saved_tensors
        accum_qtype = ctx.accum_qtype

        fp8_input = cast(FP8Tensor, fp8_input)
        fp8_weight = cast(FP8Tensor, fp8_weight)
        grad_output = grad_output.contiguous()

        ctx.metadatas = cast(FP8LinearMeta, ctx.metadatas)
        if ctx.metadatas.input_grad is None:
            # try:
            #     fp8_grad_output = FP8Tensor(
            #         grad_output,
            #         dtype=FP8LM_RECIPE.linear.input_grad.dtype,
            #         interval=FP8LM_RECIPE.linear.input_grad.interval,
            #     )
            # except:
            #     assert 1 == 1
            fp8_grad_output = FP8Tensor(
                grad_output,
                dtype=FP8LM_RECIPE.linear.input_grad.dtype,
                interval=FP8LM_RECIPE.linear.input_grad.interval,
            )

            ctx.metadatas.input_grad = fp8_grad_output.fp8_meta
        else:
            fp8_grad_output = FP8Tensor.from_metadata(grad_output, ctx.metadatas.input_grad)

        # TODO(xrsrke): extract use_split_accumulator from FP8Recipe
        # grad_input = fp8_matmul_kernel(
        #     mat_a=grad_output,
        #     transpose_a=True,
        #     mat_b=weight,
        #     transpose_b=True,
        #     use_split_accumulator=True,
        #     accum_qtype=accum_qtype,
        #     is_backward=True
        # )
        # grad_weight = fp8_matmul_kernel(
        #     mat_a=input,
        #     transpose_a=False,
        #     mat_b=grad_output,
        #     transpose_b=False,
        #     use_split_accumulator=True,
        #     accum_qtype=accum_qtype,
        #     is_backward=True
        # )

        # fp8_weight_transposed = tex.fp8_transpose(fp8_weight, fp8_weight.fp8_meta.te_dtype)
        # fp8_weight_transposed.fp8_meta = fp8_weight.fp8_meta
        # if ctx.is_weight_transposed is False:
        transposed_fp8_weight = fp8_weight.transpose_fp8()

        grad_input_temp = torch.zeros(
            fp8_grad_output.shape[0], transposed_fp8_weight.shape[0], device="cuda", dtype=QTYPE_TO_DTYPE[accum_qtype]
        )
        grad_input = fp8_matmul_kernel(
            mat_a=transposed_fp8_weight,
            transpose_a=True,
            mat_b=fp8_grad_output,
            transpose_b=False,
            output=grad_input_temp,
            use_split_accumulator=FP8LM_RECIPE.linear.split_accumulator.input_grad,
            accum_qtype=accum_qtype,
            is_backward=True,
        )

        # fp8_grad_output_transposed = tex.fp8_transpose(fp8_grad_output, fp8_grad_output.fp8_meta.te_dtype)
        # fp8_grad_output_transposed.fp8_meta = fp8_grad_output.fp8_meta
        # fp8_input_tranposed = tex.fp8_transpose(fp8_input, fp8_input.fp8_meta.te_dtype)
        # fp8_input_tranposed.fp8_meta = fp8_input.fp8_meta

        transposed_fp8_grad_output = fp8_grad_output.transpose_fp8()
        transposed_fp8_input = fp8_input.transpose_fp8()

        grad_weight_temp = torch.zeros(
            transposed_fp8_input.shape[0],
            transposed_fp8_grad_output.shape[0],
            device="cuda",
            dtype=QTYPE_TO_DTYPE[accum_qtype],
        )
        grad_weight = fp8_matmul_kernel(
            mat_a=transposed_fp8_input,
            transpose_a=True,
            mat_b=transposed_fp8_grad_output,
            transpose_b=False,
            output=grad_weight_temp,
            use_split_accumulator=FP8LM_RECIPE.linear.split_accumulator.weight_grad,
            accum_qtype=accum_qtype,
            is_backward=True,
        )
        # grad_weight = grad_weight.contiguous()
        # grad_weight_shape = grad_weight.shape
        # grad_weight = grad_weight.view(-1).view(grad_weight_shape)

        # NOTE: ∂L/∂W = Xᵀ @ ∂L/∂Y
        # [m, n] x [n, p] = [m, p]
        # so here we already transposed the input
        # grad_weight_temp = torch.zeros(
        #     transposed_fp8_input.shape[0],
        #     transposed_fp8_grad_output.shape[0],
        #     device="cuda",
        #     dtype=QTYPE_TO_DTYPE[accum_qtype],
        # )

        # grad_weight = correct_fp8_matmul_kernel(
        #     mat_a=fp8_input,
        #     transpose_a=True,
        #     mat_b=fp8_grad_output,
        #     transpose_b=False,
        #     # output=grad_weight_temp,
        #     use_split_accumulator=FP8LM_RECIPE.linear.split_accumulator.weight_grad,
        #     accum_qtype=accum_qtype,
        #     # is_backward=True
        # )

        # grad_input = _fp8_matmul_kernel_2(
        #     mat_a=grad_output,
        #     transpose_a=True,
        #     mat_b=weight,
        #     transpose_b=True,
        #     use_split_accumulator=True,
        #     accum_qtype=accum_qtype,
        # )
        # grad_weight = _fp8_matmul_kernel_2(
        #     mat_a=input,
        #     transpose_a=False,
        #     mat_b=grad_output,
        #     transpose_b=False,
        #     use_split_accumulator=True,
        #     accum_qtype=accum_qtype,
        # )

        # grad_input = _fp8_matmul_kernel(
        #     mat_a=grad_output,
        #     transpose_a=False,
        #     mat_b=weight,
        #     transpose_b=True,
        #     use_split_accumulator=True,
        #     accum_qtype=accum_qtype,
        # )
        # grad_weight = _fp8_matmul_kernel(
        #     mat_a=input,
        #     transpose_a=True,
        #     mat_b=grad_output,
        #     transpose_b=False,
        #     use_split_accumulator=True,
        #     accum_qtype=accum_qtype,
        # )

        assert grad_input.dtype == QTYPE_TO_DTYPE[accum_qtype]
        assert grad_weight.dtype == QTYPE_TO_DTYPE[accum_qtype]
        # TODO(xrsrke): maintain a persistence metadata across training

        # if fp8_weight.shape != grad_weight.shape:
        #     grad_weight = grad_weight.T

        grad_weight = grad_weight.T

        if ctx.metadatas.weight_grad is None:
            # NOTE: this is weird, i only add this when work with TP
            # didn't encount the mismatch shape problem with non-tp
            fp8_weight_grad = FP8Tensor(
                grad_weight,
                dtype=FP8LM_RECIPE.linear.weight_grad.dtype,
                interval=FP8LM_RECIPE.linear.weight_grad.interval,
            )
            ctx.metadatas.weight_grad = fp8_weight_grad.fp8_meta
        else:
            fp8_weight_grad = FP8Tensor.from_metadata(grad_weight, ctx.metadatas.weight_grad)

        # NOTE: grad_input.shape = [512, 512]
        # NOTE: fp8_grad_output.shape = [512, 128]
        # ∂L/∂W = Xᵀ @ ∂L/∂Y
        # [512, 512] @  [512, 128] = [512, 128]

        # try:
        #     fp8_weight.grad = fp8_weight_grad
        # except RuntimeError as e:
        #     assert 1 == 1

        fp8_weight.grad = fp8_weight_grad
        # TODO(xrsrke): currently some after setting .grad, can't retrieve it
        # so use ._temp_grad, remove this later on
        fp8_weight._temp_grad = fp8_weight_grad

        # try:
        #     # NOTE: sanity check
        #     assert isinstance(fp8_weight.grad, FP8Tensor)
        # except AssertionError as e:
        #     assert 1 == 1

        assert isinstance(fp8_weight.grad, FP8Tensor)

        # assert isinstance(grad_input, torch.Tensor)
        assert grad_output.__class__ == torch.Tensor
        assert grad_input.dtype == torch.float16

        return grad_input, None, None, None, None, None, None


class _FP8MatmulWithFixedShapeMismatch(torch.autograd.Function):
    @staticmethod
    @torch.no_grad()
    def forward(
        ctx,
        input: Union[FP8Tensor, torch.Tensor],
        weight: FP8Tensor,
        output: torch.Tensor,
        phony: torch.Tensor,
        metadatas: FP8LinearMeta,
        accum_qtype: DTypes,
        name: Optional[str] = None  # TODO(xrsrke): remove this shit after debugging
        # is_weight_transposed: bool = False,
    ) -> torch.Tensor:
        assert not isinstance(input, FP8Tensor)

        # NOTE: pad input shape to disibile by 16
        # input = F.pad(input, (0, 0, 0, 16 - input.shape[1] % 16))

        if metadatas.input is None:
            fp8_input = FP8Tensor(
                input,
                dtype=FP8LM_RECIPE.linear.input.dtype,
                interval=FP8LM_RECIPE.linear.input.interval,
            )
            metadatas.input = fp8_input.fp8_meta
        else:
            fp8_input = FP8Tensor.from_metadata(input, metadatas.input)

        ctx.accum_qtype = accum_qtype
        ctx.metadatas = metadatas
        # NOTE: the weight is in correct shape, [out_features, in_features]
        ctx.save_for_backward(fp8_input, weight)
        ctx.name = name

        output = correct_fp8_matmul_kernel(
            # NOTE: that works
            # mat_a=weight,
            # transpose_a=True,
            # mat_b=fp8_input,
            # transpose_b=False,
            mat_a=fp8_input,
            transpose_a=True,
            mat_b=weight,
            transpose_b=False,
            output=output,
            use_split_accumulator=FP8LM_RECIPE.linear.split_accumulator.output,
            accum_qtype=accum_qtype,
        )

        return output, phony

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_output: torch.Tensor, grad_phony: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        """
        ∂L/∂X = ∂L/∂Y @ Wᵀ
        ∂L/∂W = Xᵀ @ ∂L/∂Y
        Reference: https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html
        """
        import pydevd

        pydevd.settrace(suspend=False, trace_only_current_thread=True)
        # assert isinstance(grad_output, torch.Tensor)

        constants.DEBUG_FP8_GRAD_OUTPUT = grad_output

        assert grad_output.__class__ == torch.Tensor
        assert grad_output.dtype == torch.float16

        if ctx.name == "mlp.down_proj":
            assert 1 == 1

        if ctx.name == "lm_head":
            assert 1 == 1

        fp8_input, fp8_weight = ctx.saved_tensors
        accum_qtype = ctx.accum_qtype

        fp8_input = cast(FP8Tensor, fp8_input)
        fp8_weight = cast(FP8Tensor, fp8_weight)
        grad_output = grad_output.contiguous()

        ctx.metadatas = cast(FP8LinearMeta, ctx.metadatas)
        if ctx.metadatas.input_grad is None:
            fp8_grad_output = FP8Tensor(
                grad_output,
                dtype=FP8LM_RECIPE.linear.input_grad.dtype,
                interval=FP8LM_RECIPE.linear.input_grad.interval,
            )

            ctx.metadatas.input_grad = fp8_grad_output.fp8_meta
        else:
            fp8_grad_output = FP8Tensor.from_metadata(grad_output, ctx.metadatas.input_grad)

        # transposed_fp8_weight = fp8_weight.transpose_fp8()

        # grad_input_temp = torch.zeros(
        #     fp8_grad_output.shape[0], transposed_fp8_weight.shape[0], device="cuda", dtype=QTYPE_TO_DTYPE[accum_qtype]
        # )
        grad_input_temp = torch.zeros(
            32,
            16,
            device="cuda",
            dtype=QTYPE_TO_DTYPE[accum_qtype],
        )
        # grad_input = fp8_matmul_kernel(
        #     mat_a=transposed_fp8_weight,
        #     transpose_a=True,
        #     mat_b=fp8_grad_output,
        #     transpose_b=False,
        #     output=grad_input_temp,
        #     use_split_accumulator=FP8LM_RECIPE.linear.split_accumulator.input_grad,
        #     accum_qtype=accum_qtype,
        #     is_backward=True,
        # )

        grad_input = correct_fp8_matmul_kernel(
            mat_a=fp8_grad_output,
            transpose_a=False,
            mat_b=fp8_weight,
            transpose_b=True,
            output=grad_input_temp,
            use_split_accumulator=FP8LM_RECIPE.linear.split_accumulator.input_grad,
            accum_qtype=accum_qtype,
            # is_backward=True,
        )
        constants.DEBUG_FP8_GRAD_INPUT = grad_input

        transposed_fp8_grad_output = fp8_grad_output.transpose_fp8()
        fp8_input.transpose_fp8()

        # grad_weight_temp = torch.zeros(
        #     transposed_fp8_input.shape[0],
        #     transposed_fp8_grad_output.shape[0],
        #     device="cuda",
        #     dtype=QTYPE_TO_DTYPE[accum_qtype],
        # )

        grad_weight_temp = torch.zeros(
            32,
            16,
            device="cuda",
            dtype=QTYPE_TO_DTYPE[accum_qtype],
        )

        grad_weight = correct_fp8_matmul_kernel(
            mat_a=fp8_input,
            transpose_a=True,
            mat_b=transposed_fp8_grad_output,
            transpose_b=False,
            output=grad_weight_temp,
            use_split_accumulator=FP8LM_RECIPE.linear.split_accumulator.weight_grad,
            accum_qtype=accum_qtype,
            # is_backward=True,
        )
        constants.DEBUG_FP8_GRAD_WEIGHT = grad_weight

        assert grad_input.dtype == QTYPE_TO_DTYPE[accum_qtype]
        assert grad_weight.dtype == QTYPE_TO_DTYPE[accum_qtype]
        # TODO(xrsrke): maintain a persistence metadata across training

        # if fp8_weight.shape != grad_weight.shape:
        #     grad_weight = grad_weight.T

        grad_weight = grad_weight.T

        if ctx.metadatas.weight_grad is None:
            # NOTE: this is weird, i only add this when work with TP
            # didn't encount the mismatch shape problem with non-tp
            fp8_weight_grad = FP8Tensor(
                grad_weight,
                dtype=FP8LM_RECIPE.linear.weight_grad.dtype,
                interval=FP8LM_RECIPE.linear.weight_grad.interval,
            )
            ctx.metadatas.weight_grad = fp8_weight_grad.fp8_meta
        else:
            fp8_weight_grad = FP8Tensor.from_metadata(grad_weight, ctx.metadatas.weight_grad)

        fp8_weight.grad = fp8_weight_grad
        # TODO(xrsrke): currently some after setting .grad, can't retrieve it
        # so use ._temp_grad, remove this later on
        fp8_weight._temp_grad = fp8_weight_grad

        assert isinstance(fp8_weight.grad, FP8Tensor)

        assert grad_output.__class__ == torch.Tensor
        assert grad_input.dtype == torch.float16

        return grad_input, None, None, None, None, None, None
