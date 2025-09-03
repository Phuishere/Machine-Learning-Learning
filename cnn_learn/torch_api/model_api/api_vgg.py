# mobilenet
from typing import Optional

from torchvision.models import (
    VGG, vgg11, VGG11_Weights,
    vgg13, VGG13_Weights,
    vgg16, VGG16_Weights,
    vgg19, VGG19_Weights,
)

def get_vgg(
    layer_num: int = 16,
    weight_name: Optional[str] = None
) -> VGG:
    """
    Create a VGG model with optional pretrained ImageNet weights.

    This is a thin factory wrapper around torchvision's VGG constructors
    (vgg11, vgg13, vgg16, vgg19) that maps a simple `layer_num` and a short
    `weight_name` to the appropriate torchvision weights enum and then calls
    the constructor with `weights=...`.

    Parameters
    ----------
    layer_num : int, optional (default=16)
        VGG variant to construct. Supported values:
        {11, 13, 16, 19}. This selects the corresponding constructor
        (for example, `16` -> `vgg16`).
    weight_name : str or None, optional (default=None)
        Which pretrained weight to load. Case-insensitive. Supported values:
        - None : return a model with random initialization.
        - "v1" : use `IMAGENET1K_V1` weights (if provided by the installed torchvision).
        - "default" : use the `DEFAULT` weights (recommended when available).

    Returns
    -------
    torchvision.models.VGG
        Instantiated VGG model (one of vgg11/vgg13/vgg16/vgg19) with the
        selected weights applied (or randomly initialized if weights is None).

    Raises
    ------
    ValueError
        If `layer_num` is not one of {11, 13, 16, 19} or if `weight_name` is
        not one of {None, "v1", "default"}.

    Notes
    -----
    - This function relies on the `VGG*_Weights` enums (from torchvision).
      The actual enum members available depend on the torchvision version you
      have installed. If a requested enum member is not present, the call
      to the underlying constructor may raise an AttributeError/ValueError.
    - The function calls the torchvision constructor as `model_fn(weights=...)`.
      Older torchvision releases used `pretrained=True/False` instead of a
      `weights` enum; this wrapper assumes a modern torchvision API. If you
      need to support older torchvision versions, modify the call-site to
      handle `pretrained` fallbacks.
    - `weight_name` is interpreted case-insensitively.

    Example
    -------
    >>> model = get_vgg(16, "default")
    >>> x = torch.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.shape
    torch.Size([1, 1000])
    """
    # List of values
    layer_nums = [11, 13, 16, 19]
    weight_names = [None, "v1", "default"]

    # Sanity check
    if layer_num not in layer_nums:
        raise ValueError(f"Error: layer_num must be one of {layer_nums}")
    if weight_name not in weight_names:
        raise ValueError(f"Error: weight_name must be one of {weight_names}")
    
    # Get model
    if layer_num == 11:
        model_fn = vgg11
        weight_enum = VGG11_Weights
    elif layer_num == 13:
        model_fn = vgg13
        weight_enum = VGG13_Weights
    elif layer_num == 16:
        model_fn = vgg16
        weight_enum = VGG16_Weights
    elif layer_num == 19:
        model_fn = vgg19
        weight_enum = VGG19_Weights
    
    # Get weight
    if weight_name is None:
        weights = None
    elif weight_name.lower() == "v1":
        weights = weight_enum.IMAGENET1K_V1
    elif weight_name.lower() == "default":
        weights = weight_enum.DEFAULT # Best

    return model_fn(weights = weights)