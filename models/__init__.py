from torch import nn
from torchvision import models


from ..configs import ModelConfig
from .vgg_style import MyVGGNet
from .res_style import MyResidualNet

_CUSTOM_MODELS = {
    "vgg_style": MyVGGNet,
    "res_style": MyResidualNet,
}


def _replace_classifier_head(model: nn.Module, num_classes: int) -> None:
    """Replace the classification head for common torchvision classifier layouts."""
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return

    if hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Linear):
            model.classifier = nn.Linear(classifier.in_features, num_classes)
            return
        if isinstance(classifier, nn.Sequential):
            for idx in range(len(classifier) - 1, -1, -1):
                if isinstance(classifier[idx], nn.Linear):
                    classifier[idx] = nn.Linear(classifier[idx].in_features, num_classes)
                    return

    raise ValueError(
        f"Cannot replace classifier head for model type {type(model).__name__!r}. "
        "Expected an 'fc' or 'classifier' linear head."
    )


def ConvClassifier(config: ModelConfig) -> nn.Module:
    """Factory: instantiate the model from config.model dict.

    Expected shape:
    {"name": "resnet18", "pretrained": True}
    """
    if not isinstance(config.model, dict):
        raise ValueError(
            f"config.model must be a dict with a 'name' key, got: {config.model!r}"
        )

    name = str(config.model.get("name", "")).strip().lower()
    if not name:
        raise ValueError("config.model['name'] must be a non-empty string")
    model_kwargs = {k: v for k, v in config.model.items() if k != "name"}

    custom_cls = _CUSTOM_MODELS.get(name)
    if custom_cls is not None:
        # Lift per-model kwargs onto config for custom constructors.
        for key, value in model_kwargs.items():
            setattr(config, key, value)
        return custom_cls(config)

    try:
        fine_tune_start = int(model_kwargs.pop("fine_tune_start", 0))
        weights = model_kwargs.pop("weights", None)
        # Remove custom-model-only keys if they leaked from merged config payloads.
        model_kwargs.pop("channels", None)
        model_kwargs.pop("blocks_per_stage", None)

        try:
            model = models.get_model(name, weights=weights, **model_kwargs)
            loaded_pretrained = bool(weights)
        except RuntimeError:
            # If the local pretrained cache is incompatible/corrupted, continue from scratch.
            model = models.get_model(name, weights=None, **model_kwargs)
            loaded_pretrained = False

        print(f"Instantiated model {name!r} with pretrained={loaded_pretrained} and fine_tune_start={fine_tune_start}")

        if loaded_pretrained:
            for param in model.parameters():
                param.requires_grad = False

        if loaded_pretrained and fine_tune_start <= 1 and hasattr(model, "layer1"):
            for param in model.layer1.parameters():
                param.requires_grad = True

        if loaded_pretrained and fine_tune_start <= 2 and hasattr(model, "layer2"):
            for param in model.layer2.parameters():
                param.requires_grad = True

        if loaded_pretrained and fine_tune_start <= 3 and hasattr(model, "layer3"):
            for param in model.layer3.parameters():
                param.requires_grad = True

        if loaded_pretrained and fine_tune_start <= 4 and hasattr(model, "layer4"):
            for param in model.layer4.parameters():
                param.requires_grad = True

        _replace_classifier_head(model, config.num_classes)

        return model
    except Exception as exc:
        raise ValueError(
            f"Unknown or unsupported model {name!r} with kwargs {model_kwargs!r}"
        ) from exc


__all__ = ["ConvClassifier", "MyVGGNet", "MyResidualNet"]
