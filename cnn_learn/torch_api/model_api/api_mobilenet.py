# mobilenet
from typing import Union, Optional

from torchvision.models.mobilenetv2 import MobileNetV2
from torchvision.models.mobilenetv3 import MobileNetV3
from torchvision.models import (
    mobilenet_v2, MobileNet_V2_Weights,
    mobilenet_v3_small, mobilenet_v3_large,
    MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights,
)

def get_mobilenet(
    version: str = "v3_small",
    weight_name: Optional[str] = None
) -> Union[MobileNetV2, MobileNetV3]:
    """
    Build a MobileNet model from torchvision with optional pretrained weights.

    Parameters
    ----------
    version : str, default="v3_small"
        The MobileNet version to use. Supported values are:
        - "v2"
        - "v3_small"
        - "v3_large"
    weight_name : str or None, default=None
        The pretrained weight configuration. Supported values are:
        - None: random initialization
        - "v1": IMAGENET1K_V1 weights (when available)
        - "v2": IMAGENET1K_V2 weights (when available)
        - "default": DEFAULT weights (recommended)

    Returns
    -------
    Union[MobileNetV2, MobileNetV3]
        A MobileNet instance corresponding to the specified version and weights.

    Raises
    ------
    ValueError
        If `version` is not supported.
    ValueError
        If `weight_name` is invalid for the given version.

    Examples
    --------
    >>> model = get_mobilenet("v2", "v1")
    >>> x = torch.randn(1, 3, 224, 224)
    >>> y = model(x)
    >>> y.shape
    torch.Size([1, 1000])
    """

    # Supported versions and weight options
    versions = ["v2", "v3_small", "v3_large"]
    weight_names = [None, "v1", "v2", "default"]

    # Sanity checks
    if version not in versions:
        raise ValueError(f"version must be one of {versions}")
    if weight_name not in weight_names:
        raise ValueError(f"weight_name must be one of {weight_names}")

    # Select model constructor and corresponding weight enum
    if version == "v2":
        model_fn = mobilenet_v2
        weight_enum = MobileNet_V2_Weights
    elif version == "v3_small":
        model_fn = mobilenet_v3_small
        weight_enum = MobileNet_V3_Small_Weights
    elif version == "v3_large":
        model_fn = mobilenet_v3_large
        weight_enum = MobileNet_V3_Large_Weights

    # Map weight_name to actual enum
    if weight_name is None:
        weights = None
    elif weight_name == "v1":
        # Only MobileNetV2 has v1 weights
        if version == "v2":
            weights = weight_enum.IMAGENET1K_V1
        else:
            raise ValueError(f"{version} does not support v1 weights")
    elif weight_name == "v2":
        # MobileNetV3 Small does not have v2 weights
        if version == "v3_small":
            raise ValueError("MobileNet_v3_small does not support ImageNet1K v2 weights")
        weights = weight_enum.IMAGENET1K_V2
    elif weight_name == "default":
        weights = weight_enum.DEFAULT

    # Return model with selected weights
    return model_fn(weights=weights)