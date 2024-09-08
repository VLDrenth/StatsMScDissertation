import torch
import torch.nn as nn
from tqdm.auto import tqdm
import laplace
import logging
import time

from dataclasses import dataclass
from batchbald_redux import repeated_mnist
from .laplace_batch import get_laplace_batch
from laplace.curvature import AsdlGGN


def run_active_learning(train_loader, test_loader, pool_loader, active_learning_data, model_constructor, config, device):
    if config.backend == 'AsdlGGN':
        backend = AsdlGGN

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Run experiment
    test_accs = []
    test_loss = []
    added_indices = []
    added_labels = []
    times = []

    pbar = tqdm(initial=len(active_learning_data.training_dataset), total=config.max_training_samples, desc="Training Set Size")
    loss_fn = nn.NLLLoss()

    while True:
        model = model_constructor()
        optimizer = torch.optim.Adam(model.parameters())

        model.train()

        # Train
        for data, target in tqdm(train_loader, desc="Training", leave=False):
            data = data.to(device=device)
            target = target.to(device=device)

            optimizer.zero_grad()
            
            logits = model(data)
            if config.al_method == 'badge':
                logits = logits[0]
            prediction = torch.log_softmax(logits.squeeze(1), dim=1)

            loss = loss_fn(prediction, target)

            loss.backward()
            optimizer.step()

        # Test
        loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Testing", leave=False):
                data = data.to(device=device)
                target = target.to(device=device)

                logits = model(data)
                if config.al_method == 'badge':
                    logits = logits[0]

                prediction = torch.log_softmax(logits.squeeze(1), dim=1)
                loss += loss_fn(prediction, target)

                prediction = prediction.argmax(dim=1)
                correct += prediction.eq(target.view_as(prediction)).sum().item()

        loss /= len(test_loader.dataset)
        test_loss.append(loss)

        percentage_correct = 100.0 * correct / len(test_loader.dataset)
        test_accs.append(percentage_correct)
        logger.info('Training set size: %d, Test set accuracy: %.2f, Test set loss: %.4f', 
                    len(active_learning_data.training_dataset),
                    percentage_correct,
                    loss)
        

        if len(active_learning_data.training_dataset) >= config.max_training_samples:
            break

        # Acquire new batch from pool samples using entropy acquisition function
        N = len(active_learning_data.pool_dataset)
        
        la = laplace.Laplace(
                            model,
                            likelihood="classification",
                            subset_of_weights=config.subset_of_weights,
                            hessian_structure=config.hessian_structure,
                            backend=backend,
                            temperature=config.temperature
                        )
        # time the acquisition function and the laplace approximation
        start_time = time.perf_counter()

        if config.al_method not in  ['random', 'badge']:
            
            # Fit Laplace approximation
            la.fit(train_loader)

            # Empirical Bayes for prior precision
            la.optimize_prior_precision(method='marglik', pred_type='glm', link_approx='probit')
            
            # Get acquisition batch
            candidate_batch = get_laplace_batch(model=la, pool_loader=pool_loader,
                                                    acquisition_batch_size=config.acquisition_batch_size,
                                                    device=device, 
                                                    method=config.al_method)
        else:
            candidate_batch = get_laplace_batch(model=model, pool_loader=pool_loader,
                                                    acquisition_batch_size=config.acquisition_batch_size,
                                                    device=device, 
                                                    method=config.al_method)
            
        total_time = time.perf_counter() - start_time
        times.append(total_time)

        targets = repeated_mnist.get_targets(active_learning_data.pool_dataset)
        dataset_indices = active_learning_data.get_dataset_indices(candidate_batch.indices)

        print("Dataset indices: ", dataset_indices)
        print("Scores: ", candidate_batch.scores)
        print("Labels: ", targets[candidate_batch.indices])

        active_learning_data.acquire(candidate_batch.indices)
        added_indices.append(dataset_indices)
        added_labels.append(targets[candidate_batch.indices])
        pbar.update(len(dataset_indices))

    test_loss = [loss.cpu() for loss in test_loss]
    added_labels = [labels.cpu() for labels in added_labels]

    return {'test_accs': test_accs, 'test_loss': test_loss, 'added_indices': added_indices, 'added_labels': added_labels, 'times': times}