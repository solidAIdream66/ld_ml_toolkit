from torch import nn

from ..configs import ModelConfig
from .vgg_style import MyVGGNet
from .res_style import MyResidualNet

_REGISTRY = {
    "vgg_style": MyVGGNet,
    "res_style": MyResidualNet,
}


def ConvClassifier(config: ModelConfig) -> nn.Module:
    """Factory: select and instantiate the model specified by config.model."""
    name = str(config.model).strip().lower()
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown model {config.model!r}. "
            f"Available: {list(_REGISTRY)}"
        )
    return cls(config)


__all__ = ["ConvClassifier", "MyVGGNet", "MyResidualNet"]
