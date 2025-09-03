import os

from torch.utils.data import DataLoader
from torchvision import datasets

# Assuming your images are in '/dataset/train'; '/dataset/test'; '/dataset/val'
def get_loader_from_dir(
    base_dir: str = "./dataset",
    train_transform = None,
    test_transform = None,
    collate_fn = None,
    batch_size: int = 32,
    get_val: bool = True,
    verbose: bool = True,
) -> tuple[DataLoader]:
    """
    Create PyTorch DataLoaders for train/validation/test sets from a directory structure.

    This function expects the dataset to follow an ImageNet-style folder layout:
        base_dir/
            train/
                class1/xxx.png
                class2/xxy.png
                ...
            test/
                class1/yyy.png
                class2/yyz.png
                ...
            val/    (optional, only if get_val=True)
                class1/zzz.png
                class2/zza.png
                ...

    Parameters
    ----------
    base_dir : str, default="./dataset"
        Root directory containing "train", "test", and optionally "val" subdirectories.
    train_transform : callable, optional
        Transform to apply to training images (e.g., torchvision.transforms).
    test_transform : callable, optional
        Transform to apply to validation/test images.
    collate_fn : callable, optional
        Custom collate function for the DataLoader (e.g., for variable-sized inputs).
    batch_size : int, default=32
        Number of samples per batch to load.
    get_val : bool, default=True
        If True, also load the validation set from `base_dir/val`.
        If False, only return train and test loaders.
    verbose : bool, default=True
        If True, print dataset statistics such as class names and dataset sizes.

    Returns
    -------
    tuple of DataLoader
        - If get_val=True: (train_loader, valid_loader, test_loader)
        - If get_val=False: (train_loader, test_loader)

    Raises
    ------
    FileNotFoundError
        If required directories ("train", "test", or "val" when get_val=True) do not exist.
    """
    # Get directories
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    if get_val:
        val_dir = os.path.join(base_dir, "val")

    # Get dataset
    train_dataset = datasets.ImageFolder(train_dir, transform = train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform = test_transform)
    if get_val:
        val_dataset = datasets.ImageFolder(val_dir, transform = test_transform)

    # Print out result
    if verbose:
        print("Classes:", train_dataset.classes)
        if get_val:
            print("Sizes (train/valid/test):", len(train_dataset), len(val_dataset), len(test_dataset))
        else:
            print("Sizes (train/test):", len(train_dataset), len(test_dataset))

    # Get loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    if get_val:
        valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Return
    if get_val:
        return train_loader, valid_loader, test_loader
    else:
        return train_loader, test_loader