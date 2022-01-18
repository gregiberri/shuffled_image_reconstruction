from torch.utils.data import DataLoader

from data.datasets.mnist_dataloader import MNISTDataloader


def get_dataloader(data_config, mode):
    dataset_mode = mode if mode != 'topk_gathering' else 'train'
    # get the iterator object
    if data_config.name == 'MNIST':
        dataset = MNISTDataloader(data_config.params, dataset_mode)
    else:
        raise ValueError(f'Wrong dataset name: {data_config.name}')

    # batch size 1 for validation
    batch_size = int(data_config.params.batch_size) if 'train' in mode else data_config.params.eval_batch_size

    # make the torch dataloader object
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=data_config.params.workers,
                        drop_last=True,
                        shuffle='train' in mode,
                        pin_memory=data_config.params.load_into_memory)

    return loader
