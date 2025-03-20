from torch.utils.data import DataLoader


def get_first_batch(dataloader: DataLoader):
    # Get an iterator from the DataLoader
    iterator = iter(dataloader)

    # Get the first batch
    try:
        first_batch = next(iterator)
        return first_batch
    except StopIteration:
        print("The DataLoader is empty!")
        return None


def scale_lr(base_lr: float, num_gpus: int):
    return base_lr * num_gpus
