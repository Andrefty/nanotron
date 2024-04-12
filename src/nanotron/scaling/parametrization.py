import math
from abc import abstractmethod
from enum import Enum, auto
from typing import Dict

# from nanotron.config import ModelArgs, OptimizerArgs
from torch import nn
from torch.nn import init


class ParametrizationMethod(Enum):
    STANDARD = auto()
    SPECTRAL_MUP = auto()


class Parametrizator:
    def __init__(self, config: "ModelArgs"):
        self.config = config

    def parametrize(self, param_name: str, module: nn.Module):
        if not isinstance(module, tuple(self.MODULE_TO_PARAMETRIZE.keys())):
            raise Exception(f"Parameter {param_name} was not initialized")

        return self.MODULE_TO_PARAMETRIZE[type(module)](param_name, module)


class StandardParametrizator(Parametrizator):
    def __init__(self, config: "ModelArgs"):
        super().__init__(config)
        from nanotron.nn.layer_norm import TritonRMSNorm
        from nanotron.parallel.tensor_parallel.nn import (
            TensorParallelColumnLinear,
            TensorParallelEmbedding,
            TensorParallelRowLinear,
        )

        self.MODULE_TO_PARAMETRIZE = {
            TensorParallelColumnLinear: self._parametrize_column_linear,
            TensorParallelRowLinear: self._parametrize_row_linear,
            TritonRMSNorm: self._parametrize_layer_norm,
            TensorParallelEmbedding: self._parametrize_embedding,
        }

        self.std = config.init_method.std
        self.num_layers = config.model_config.num_hidden_layers

    def _parametrize_column_linear(self, param_name: str, module: nn.Module):
        assert param_name in ["weight", "bias"]

        if "weight" == param_name:
            init.normal_(module.weight, mean=0.0, std=self.std)
        elif "bias" == param_name:
            module.bias.zero_()

    def _parametrize_row_linear(self, param_name: str, module: nn.Module):
        assert param_name in ["weight", "bias"]

        if "weight" == param_name:
            std = self.std / math.sqrt(2 * self.num_layers)
            init.normal_(module.weight, mean=0.0, std=std)
        elif "bias" == param_name:
            module.bias.zero_()

    def _parametrize_layer_norm(self, param_name: str, module: nn.Module):
        assert param_name in ["weight", "bias"]

        if "weight" == param_name:
            # TODO @thomasw21: Sometimes we actually want 0
            module.weight.fill_(1)
        elif "bias" == param_name:
            module.bias.zero_()

    def _parametrize_embedding(self, param_name: str, module: nn.Module):
        assert param_name in ["weight"]

        if "weight" == param_name:
            init.normal_(module.weight, mean=0.0, std=self.std)


class SpectralMupParametrizator(Parametrizator):
    """
    A Spectral Condition for Feature Learning.
    https://arxiv.org/abs/2310.17813
    """

    def __init__(self, config: "ModelArgs"):
        super().__init__(config)
        from nanotron.nn.layer_norm import TritonRMSNorm
        from nanotron.parallel.tensor_parallel.nn import (
            TensorParallelColumnLinear,
            TensorParallelEmbedding,
            TensorParallelRowLinear,
        )

        self.MODULE_TO_PARAMETRIZE = {
            TensorParallelColumnLinear: self._parametrize_mup_weight,
            TensorParallelRowLinear: self._parametrize_mup_weight,
            TritonRMSNorm: self._parametrize_layer_norm,
            TensorParallelEmbedding: self._parametrize_embedding,
        }
        self.std = 1.0

    @staticmethod
    def _compute_spectral_std(std: float, fan_in: int, fan_out: int):
        return (std / math.sqrt(fan_in)) * min(1, math.sqrt(fan_out / fan_in))

    def _parametrize_mup_weight(self, param_name: str, module: nn.Module):
        from nanotron.parallel.tensor_parallel.nn import (
            TensorParallelColumnLinear,
            TensorParallelRowLinear,
        )

        assert param_name in ["weight", "bias"]

        data = module.weight if param_name == "weight" else module.bias
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(data)
        world_size = module.world_size

        if isinstance(module, TensorParallelColumnLinear):
            fan_out = fan_out * world_size
        elif isinstance(module, TensorParallelRowLinear):
            fan_in = fan_in * world_size
        else:
            raise ValueError(f"Unknown module {module}")

        vocab_size = self.config.model_config.vocab_size
        if fan_in == vocab_size or fan_out == vocab_size:
            return self._parametrize_embedding(param_name, module)

        std = SpectralMupParametrizator._compute_spectral_std(std=self.std, fan_in=fan_in, fan_out=fan_out)
        init.normal_(data, mean=0.0, std=std)

    def _parametrize_layer_norm(self, param_name: str, module: nn.Module):
        assert param_name in ["weight", "bias"]

        # NOTE: you're free to change the initialization of layer norm
        # as it's not a part of µTransfer
        if "weight" == param_name:
            module.weight.fill_(1)
        elif "bias" == param_name:
            module.bias.zero_()

    def _parametrize_embedding(self, param_name: str, module: nn.Module):
        assert param_name in ["weight"]

        # NOTE: you're free to change the initialization of input embedding/lm head
        if "weight" == param_name:
            init.normal_(module.weight, mean=0.0, std=self.std)


class LearningRateForParametrizator:
    def __init__(self, names_to_modules: Dict[str, nn.Module], config: "OptimizerArgs"):
        self.config = config
        self.names_to_modules = names_to_modules

    @abstractmethod
    def get_lr(self, param_name: str, module: nn.Module):
        raise NotImplementedError


class LearningRateForSP(LearningRateForParametrizator):
    def get_lr(self, param_name: str, param: nn.Module):
        return self.config.learning_rate_scheduler.learning_rate


class LearningRateForSpectralMup(LearningRateForParametrizator):
    def __init__(self, names_to_modules: Dict[str, nn.Module], config: "OptimizerArgs"):
        super().__init__(names_to_modules, config)

        from nanotron.nn.layer_norm import TritonRMSNorm
        from nanotron.parallel.tensor_parallel.nn import (
            TensorParallelColumnLinear,
            TensorParallelEmbedding,
            TensorParallelRowLinear,
        )

        self.MODULE_TO_PARAMETRIZE = {
            TensorParallelColumnLinear: self._get_mup_lr,
            TensorParallelRowLinear: self._get_mup_lr,
            TritonRMSNorm: self._get_global_lr,
            TensorParallelEmbedding: self._get_global_lr,
        }

    def _get_mup_lr(self, param: nn.Parameter, module: nn.Module):
        lr = self.config.learning_rate_scheduler.learning_rate
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(param)
        world_size = module.world_size

        if isinstance(module, TensorParallelColumnLinear):
            fan_out = fan_out * world_size
        elif isinstance(module, TensorParallelRowLinear):
            fan_in = fan_in * world_size
        else:
            raise ValueError(f"Unknown module {module}")

        return lr * (fan_out / fan_in)

    def _get_global_lr(self, param: nn.Parameter, module: nn.Module) -> float:
        return self.config.learning_rate_scheduler.learning_rate

    def get_lr(self, param_name: str, param: nn.Parameter) -> float:
        module = self.names_to_modules[param_name]
        return self.MODULE_TO_PARAMETRIZE[type(module)](param, module)