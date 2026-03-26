#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import os
import json
from typing import Any

import torch.nn as nn
from timm.models import create_model

from .mci import GlobalPool2D


def load_model_config(
        model_name: str,
) -> Any:
    # Strip suffixes to model name
    model_name = "_".join(model_name.split("_")[0:2])

    # Config files
    root_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(root_dir, "configs")
    model_cfg_file = os.path.join(configs_dir, model_name + ".json")

    # Get config from yaml file
    if not os.path.exists(model_cfg_file):
        raise ValueError(f"Unsupported model name: {model_name}")
    model_cfg = json.load(open(model_cfg_file, "r"))

    return model_cfg


class MCi(nn.Module):
    """
    This class implements `MCi Models <https://arxiv.org/pdf/2311.17049.pdf>`_
    """

    def __init__(self, model_name: str, *args, **kwargs) -> None:
        super().__init__()
        self.projection_dim = None
        if "projection_dim" in kwargs:
            self.projection_dim = kwargs.get("projection_dim")

        # Create model
        self.model = create_model(model_name, projection_dim=self.projection_dim)
        print(f"Loaded MobileCLIP model: {model_name}")
        # Build out projection head.
        if self.projection_dim is not None:
            if hasattr(self.model, "head"):
                self.model.head = MCi._update_image_classifier(
                    image_classifier=self.model.head, projection_dim=self.projection_dim
                )

    def forward(self, x: Any, *args, **kwargs) -> Any:
        """A forward function of the model."""
        x = self.model(x, *args, **kwargs)
        return x

    @staticmethod
    def _get_in_feature_dimension(image_classifier: nn.Module) -> int:
        """Return the input feature dimension to the image classification head."""
        in_features = None
        if isinstance(image_classifier, nn.Sequential):
            # Classifier that uses nn.Sequential usually has global pooling and
            # multiple linear layers. Find the first linear layer and get its
            # in_features
            for layer in image_classifier:
                if isinstance(layer, nn.Linear):
                    in_features = layer.in_features
                    break
        elif isinstance(image_classifier, nn.Linear):
            in_features = image_classifier.in_features

        if in_features is None:
            raise NotImplementedError(
                f"Cannot get input feature dimension of {image_classifier}."
            )
        return in_features

    @staticmethod
    def _update_image_classifier(
        image_classifier: nn.Module, projection_dim: int, *args, **kwargs
    ) -> nn.Module:
        in_features = MCi._get_in_feature_dimension(image_classifier)
        new_img_classifier = GlobalPool2D(in_dim=in_features, out_dim=projection_dim)
        return new_img_classifier