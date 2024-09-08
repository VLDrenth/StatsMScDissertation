from dataclasses import dataclass


# set configurations for redundant MNIST experiment
@dataclass
class ActiveLearningConfigRedundant:
    subset_of_weights: str = 'last_layer'
    hessian_structure: str = 'kron'
    backend: str = 'AsdlGGN'
    temperature: float = 1.0
    max_training_samples: int = 100
    acquisition_batch_size: int = 10
    al_method: str = 'bald'
    test_batch_size: int = 512
    num_classes: int = 10
    num_initial_samples: int = 20
    training_iterations: int = 4096 * 6
    scoring_batch_size: int = 64
    train_batch_size: int = 64
    extract_pool: int = 9 * 60000
    num_repeats: int = 10
    samples_per_digit: int = 100

@dataclass
class ActiveLearningConfig:
    subset_of_weights: str = 'last_layer'
    hessian_structure: str = 'kron'
    backend: str = 'AsdlGGN'
    temperature: float = 1
    max_training_samples: int = 500
    acquisition_batch_size: int = 100
    al_method: str = 'badge'
    test_batch_size: int = 512
    num_classes: int = 10
    num_initial_samples: int = 50
    training_iterations: int = 4096 * 6
    scoring_batch_size: int = 64
    train_batch_size: int = 64
    extract_pool: int = 0 
    dataset: str = 'mnist'

def get_config(min_samples, max_samples, acquisition_batch_size, method, dataset):
    '''
    Returns dataclass object with active learning configuration
    --------------------------------
    min_samples: int - number of initial samples
    max_samples: int - maximum number of training samples
    acquisition_batch_size: int - number of samples to acquire per iteration
    method: str - method to use for batch selection
    dataset: str - dataset to uses

    '''
    if dataset == 'repeated_mnist':
        config = ActiveLearningConfigRedundant()
    elif dataset == 'imagenet':
        config = ActiveLearningConfig()
        config.extract_pool = 322979 * (1/3)
    else:
        config = ActiveLearningConfig()

    config.num_initial_samples = min_samples
    config.max_training_samples = max_samples
    config.al_method = method
    config.dataset = dataset
    config.acquisition_batch_size = acquisition_batch_size

    experiment_name = f"{method}_{dataset}_{min_samples}_to_{max_samples}_B={acquisition_batch_size}"
    return config, experiment_name

    
