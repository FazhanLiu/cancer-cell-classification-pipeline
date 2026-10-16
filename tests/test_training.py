import copy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cancer_pipeline.artifacts import load_checkpoint, new_run
from cancer_pipeline.data import make_loaders, prepare_manifest
from cancer_pipeline.engine import fit, train_one_epoch
from cancer_pipeline.losses import FocalLoss
from cancer_pipeline.utils import set_seed


@pytest.mark.parametrize('loss_kind', ['ce', 'weighted_smooth_ce', 'focal'])
def test_gradient_accumulation_matches_effective_batches(loss_kind):
    set_seed(9)
    x, y = torch.randn(11, 3), torch.tensor([0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0])
    dataset = TensorDataset(x, y, torch.arange(len(y)))
    first = nn.Linear(3, 2)
    second = copy.deepcopy(first)
    criterion = {'ce': nn.CrossEntropyLoss(), 'weighted_smooth_ce': nn.CrossEntropyLoss(weight=torch.tensor([0.3, 2.1]), label_smoothing=0.1), 'focal': FocalLoss()}[loss_kind]
    for model, batch, accumulation in ((first, 3, 3), (second, 9, 1)):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        train_one_epoch(model, DataLoader(dataset, batch_size=batch), criterion, optimizer,
                        torch.amp.GradScaler('cuda', enabled=False), torch.device('cpu'), accumulation, False)
    for a, b in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize('weighted_sampling', [False, True])
def test_epoch_resume_matches_continuous_training(experiment, weighted_sampling):
    _, config = experiment
    prepare_manifest(config)
    config['training_mode'] = 'standard'
    config['loss'] = {'name': 'cross_entropy'}

    def setup():
        set_seed(42)
        weights = torch.arange(1, 29, dtype=torch.float) if weighted_sampling else None
        loaders, _, provenance = make_loaders(config, train_weights=weights)
        model = nn.Sequential(nn.Flatten(), nn.Linear(3*32*32, 2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['train']['lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
        return model, loaders, provenance, optimizer, scheduler

    model, loaders, provenance, optimizer, scheduler = setup()
    full = new_run(config)
    fit(model, loaders, nn.CrossEntropyLoss(), optimizer, scheduler, torch.device('cpu'), config, full, provenance)
    model, loaders, provenance, optimizer, scheduler = setup()
    partial = new_run(config)
    fit(model, loaders, nn.CrossEntropyLoss(), optimizer, scheduler, torch.device('cpu'), config, partial, provenance, stop_after_epoch=1)
    # Construction consumes RNG, then checkpoint restoration must recover it.
    model, loaders, provenance, optimizer, scheduler = setup()
    resumed = new_run(config)
    fit(model, loaders, nn.CrossEntropyLoss(), optimizer, scheduler, torch.device('cpu'), config, resumed, provenance, resume=partial / 'last.pt')
    original, restored = load_checkpoint(full / 'last.pt'), load_checkpoint(resumed / 'last.pt')
    for key in original['model']:
        torch.testing.assert_close(original['model'][key], restored['model'][key], atol=0, rtol=0)
    assert original['history'] == restored['history']
    assert original['scheduler'] == restored['scheduler']


def test_resume_rejects_legacy(experiment):
    _, config = experiment
    prepare_manifest(config)
    loaders, _, provenance = make_loaders(config)
    model = nn.Linear(2, 2)
    output = new_run(config)
    path = output / 'old.pt'
    torch.save({'model': model.state_dict()}, path)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    with pytest.raises(ValueError, match='Legacy checkpoint cannot resume'):
        fit(model, loaders, nn.CrossEntropyLoss(), optimizer, scheduler, torch.device('cpu'), config, output, provenance, resume=path)
