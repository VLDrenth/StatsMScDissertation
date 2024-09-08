from batchbald_redux import repeated_mnist, active_learning
from torch.utils import data
from ddu_dirty_mnist import DirtyMNIST
import torch
import numpy as np
from torchvision import datasets, transforms

def create_dataloaders(config, **kwargs):
    # loading data
    if config.dataset == 'mnist':
        train_dataset, test_dataset = repeated_mnist.create_MNIST_dataset()
    elif config.dataset == 'repeated_mnist':
        train_dataset, test_dataset = create_small_repeated_MNIST_dataset(num_repetitions=config.num_repeats)
    elif config.dataset == 'dirty_mnist':
        train_dataset, test_dataset = create_dirty_MNIST_dataset()
    elif config.dataset == 'fashion_mnist':
        train_dataset, test_dataset = create_fashion_MNIST_dataset()
    elif config.dataset == 'imagenet':
        train_dataset, test_dataset = create_embeddings_dataset()
    else:
        raise ValueError(f'Unknown dataset {config.dataset}')
    
    # Create data loaders
    train_loader, test_loader, pool_loader, active_learning_data = create_dataloaders_AL(train_dataset, test_dataset, config)

    return train_loader, test_loader, pool_loader, active_learning_data

def create_dirty_MNIST_dataset():
    train_dataset = DirtyMNIST("./data", train=True, download=True)
    test_dataset = DirtyMNIST("./data", train=False, download=True)
    return train_dataset, test_dataset

def create_embeddings_dataset():
    '''
    Create torch datasets to be used in a torch Dataloader for training and testing
    '''

    # load embeddings and labels
    embeddings_train = np.load("data/imagenet-embeddings/embeddings_vitb4_300ep_train.npy")
    labels_train = np.load("data/imagenet-embeddings/superclass_labels_train.npy")

    embeddings_val = np.load("data/imagenet-embeddings/embeddings_vitb4_300ep_val.npy")
    labels_val = np.load("data/imagenet-embeddings/superclass_labels_val.npy")

    # convert to torch tensors
    embeddings_train = torch.tensor(embeddings_train)
    labels_train = torch.tensor(labels_train)

    embeddings_val = torch.tensor(embeddings_val)
    labels_val = torch.tensor(labels_val)

    # change labels to long
    labels_train = labels_train.long()
    labels_val = labels_val.long()

    # create datasets with embeddings and labels as
    train_dataset = torch.utils.data.TensorDataset(embeddings_train, labels_train)
    val_dataset = torch.utils.data.TensorDataset(embeddings_val, labels_val)

    train_dataset.targets = labels_train
    val_dataset.targets = labels_val

    # take all validation data of classes 0-9 and add 10% of data from class 10
    val_indices = []
    for i in range(10):
        val_indices += np.where(labels_val == i)[0].tolist()
    val_indices += np.where(labels_val == 10)[0].tolist()[:int(0.1*len(np.where(labels_val == 10)[0]))]
    val_dataset = data.Subset(val_dataset, val_indices)

    return train_dataset, val_dataset

def create_fashion_MNIST_dataset(data_dir="./data"):
    # Set up transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # Fashion MNIST mean and std
    ])

    # Load train dataset
    train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)

    # Load test dataset
    test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)

    return train_dataset, test_dataset

def create_small_repeated_MNIST_dataset(*, num_repetitions: int = 10, add_noise: bool = True):
    # Based on repeated_mnist.create_repeated_MNIST_dataset from batchbald_redux

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)

    # take random subset 10% of the data
    indices = active_learning.get_balanced_sample_indices(repeated_mnist.get_targets(train_dataset),
                                                          num_classes=10,
                                                           n_per_digit= (1/(10 * num_repetitions)) * len(train_dataset))
    train_dataset = data.Subset(train_dataset, indices)

    if num_repetitions > 1:
        train_dataset = data.ConcatDataset([train_dataset] * num_repetitions)

    if add_noise:
        dataset_noise = torch.empty((len(train_dataset), 28, 28), dtype=torch.float32).normal_(0.0, 0.1)

        def apply_noise(idx, sample):
            data, target = sample
            return data + dataset_noise[idx], target

        train_dataset = repeated_mnist.TransformedDataset(train_dataset, transformer=apply_noise)

    test_dataset = datasets.MNIST("data", train=False, transform=transform)

    return train_dataset, test_dataset

def create_dataloaders_AL(train_dataset, test_dataset, config, **kwargs):
    if config.dataset == 'dirty_mnist':
        initial_samples = active_learning.get_balanced_sample_indices(
            repeated_mnist.get_targets(data.Subset(train_dataset, range(60000))),
            num_classes=config.num_classes,
            n_per_digit=config.num_initial_samples / config.num_classes
        )
    else:
        # Get indices of initial samples
        initial_samples = active_learning.get_balanced_sample_indices(
            repeated_mnist.get_targets(train_dataset),
            num_classes=config.num_classes,
            n_per_digit=config.num_initial_samples / config.num_classes
        )

    # Create data loaders
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        **kwargs
    )

    active_learning_data = active_learning.ActiveLearningData(train_dataset)
    active_learning_data.acquire(initial_samples)
    active_learning_data.extract_dataset_from_pool(config.extract_pool)

    train_loader = torch.utils.data.DataLoader(
        active_learning_data.training_dataset,
        sampler=active_learning.RandomFixedLengthSampler(
            active_learning_data.training_dataset,
            config.training_iterations
        ),
        batch_size=config.train_batch_size,
        **kwargs
    )

    pool_loader = torch.utils.data.DataLoader(
        active_learning_data.pool_dataset,
        batch_size=config.scoring_batch_size,
        shuffle=False,
        **kwargs
    )

    return train_loader, test_loader, pool_loader, active_learning_data