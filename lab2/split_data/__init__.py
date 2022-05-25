from torchtext.legacy.data import BucketIterator


def _len_sort_key(x):
    return len(x.src)


def split_dataset(dataset):
    train_data, valid_data, test_data = dataset.split(split_ratio=[0.8, 0.15, 0.05])
    return train_data, valid_data, test_data


def get_iterators(train_data, valid_data, test_data, device, batch_size):
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batch_size,
        device=device,
        sort_key=_len_sort_key
    )
    return train_iterator, valid_iterator, test_iterator
