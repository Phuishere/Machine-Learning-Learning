# resnet
from typing import Optional
from torchvision.models import (
    ResNet, resnet18, resnet34, resnet50, resnet101, resnet152,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
)

def get_resnet(
    layer_num: int = 50,
    weight_name: Optional[str] = None
) -> ResNet:
    """
    Build a ResNet model from torchvision with optional pretrained weights.

    Parameters
    ----------
    layer_num : int, default=50
        The depth of the ResNet architecture to use. Supported values are
        {18, 34, 50, 101, 152}.
    weight_name : str or None, default=None
        The pretrained weight configuration. Supported values are:
        - None: random initialization
        - "v1": IMAGENET1K_V1 weights (top-1 acc ≈ 76.13%)
        - "v2": IMAGENET1K_V2 weights (top-1 acc ≈ 80.86%)
        - "default": DEFAULT weights (recommended)

    Returns
    -------
    torchvision.models.ResNet
        A ResNet instance corresponding to the specified layer depth and weights.

    Raises
    ------
    ValueError
        If `layer_num` is not one of the supported values.
    ValueError
        If `weight_name` is not one of {None, "v1", "v2", "default"}.

    Examples
    --------
    >>> model = get_resnet(50, "v2")
    >>> x = torch.randn(1, 3, 224, 224)
    >>> y = model(x)
    >>> y.shape
    torch.Size([1, 1000])
    """
    # List of values
    layer_nums = [18, 34, 50, 101, 152]
    weight_names = [None, "v1", "v2", "default"]

    # Sanity check
    if layer_num not in layer_nums:
        raise ValueError(f"Error: layer_num must be one of {layer_nums}")
    if weight_name not in weight_names:
        raise ValueError(f"Error: weight_name must be one of {weight_names}")
    
    # Get model
    if layer_num == 18:
        model_fn = resnet18
        weight_enum = ResNet18_Weights
    elif layer_num == 34:
        model_fn = resnet34
        weight_enum = ResNet34_Weights
    elif layer_num == 50:
        model_fn = resnet50
        weight_enum = ResNet50_Weights
    elif layer_num == 101:
        model_fn = resnet101
        weight_enum = ResNet101_Weights
    elif layer_num == 152:
        model_fn = resnet152
        weight_enum = ResNet152_Weights
    
    # Get weight
    if weight_name is None:
        weights = None
    elif weight_name.lower() == "v1":
        weights = weight_enum.IMAGENET1K_V1 # Old weights with accuracy 76.130%
    elif weight_name.lower() == "v2":
        weights = weight_enum.IMAGENET1K_V2 # New weights with accuracy 80.858%
    elif weight_name.lower() == "default":
        weights = weight_enum.DEFAULT # Best

    return model_fn(weights = weights)