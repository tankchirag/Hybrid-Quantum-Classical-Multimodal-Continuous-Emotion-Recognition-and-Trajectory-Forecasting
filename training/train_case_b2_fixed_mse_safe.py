"""Independent B2 no-history experiment: MSE-led training, full-validation model selection.

Copy to training/train_case_b2_fixed_mse.py. From repository root:
 python -m training.train_case_b2_fixed_mse --epochs 3 --train-batches 100
 python -m training.train_case_b2_fixed_mse --epochs 35

Does not edit original trainers, loss functions, configs, checkpoints, or earlier runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from training.train_case_b2_fixed_no_history import FixedNoHistoryB2, build_ablation_config
from training.base_training_script import build_dataloaders, to_namespace
from training.trainer import Trainer
from training.optimizer import build_optimizer
from training.scheduler import build_scheduler, step_scheduler
from utils.seed import set_seed


class MSEMultiTaskLoss(nn.Module):
    """Sum of VA-channel MSE for current recognition and all future horizons."""
    def __init__(self, rec_weight=1., fore_weight=1.):
        super().__init__()
        self.rec_weight, self.fore_weight = rec_weight, fore_weight

    def forward(self, outputs, recognition_targets, forecast_targets):
        device = next(iter(outputs['recognition']['continuous'].values())).device
        rec = torch.zeros((), device=device)
        fore = torch.zeros((), device=device)
        for key, pred in outputs['recognition']['continuous'].items():
            target = recognition_targets['continuous'][key]
            if pred.shape != target.shape:
                raise ValueError(f'recognition/{key}: {pred.shape} vs {target.shape}')
            rec = rec + F.mse_loss(pred, target)
        for key, pred in outputs['forecasting']['continuous'].items():
            target = forecast_targets['continuous'][key]
            if pred.shape != target.shape:
                raise ValueError(f'forecast/{key}: {pred.shape} vs {target.shape}')
            fore = fore + F.mse_loss(pred, target)
        return {'total_loss': self.rec_weight*rec + self.fore_weight*fore,
                'recognition_loss': rec, 'forecasting_loss': fore}


class RegressionMoments:
    """Aggregate metrics over all examples, not individual low-variance batches."""
    def __init__(self):
        self.n = 0
        self.sy = self.sp = self.syy = self.spp = self.syp = self.sae = self.sse = 0.

    def add(self, pred, truth):
        p = pred.detach().reshape(-1).double()
        y = truth.detach().reshape(-1).double()
        self.n += y.numel()
        self.sy += y.sum().item()
        self.sp += p.sum().item()
        self.syy += (y*y).sum().item()
        self.spp += (p*p).sum().item()
        self.syp += (y*p).sum().item()
        self.sae += (y-p).abs().sum().item()
        self.sse += ((y-p)**2).sum().item()

    def metrics(self):
        if not self.n:
            raise RuntimeError('Empty validation aggregation')
        my, mp = self.sy/self.n, self.sp/self.n
        vy = max(0., self.syy/self.n-my*my)
        vp = max(0., self.spp/self.n-mp*mp)
        cov = self.syp/self.n-my*mp
        import math
        return {'n': self.n, 'ccc': 2*cov/(vy+vp+(my-mp)**2+1e-8),
                'pcc': cov/math.sqrt(vy*vp) if vy>1e-12 and vp>1e-12 else None,
                'mse': self.sse/self.n, 'rmse': math.sqrt(self.sse/self.n),
                'mae': self.sae/self.n, 'target_std': math.sqrt(vy),
                'prediction_std': math.sqrt(vp)}


class GlobalMSETrainer(Trainer):
    def train_one_epoch(self, train_loader, epoch, total_epochs):
        """Stop immediately at first nonfinite input, output, loss, or gradient."""
        self.model.train()
        sums = {'loss': 0., 'recognition_loss': 0., 'forecasting_loss': 0.}
        count = 0
        for idx, batch in enumerate(train_loader):
            if self.max_train_batches is not None and idx >= self.max_train_batches:
                break
            batch = self._to_device(batch)
            # Check the actual targets used by the objective.
            for target_key in ('recognition_targets', 'forecast_targets'):
                for name, value in batch[target_key]['continuous'].items():
                    if not torch.isfinite(value).all():
                        raise FloatingPointError(f'Epoch {epoch}, batch {idx}, nonfinite {target_key}/{name}')
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(batch)
            for task in ('recognition', 'forecasting'):
                for name, value in outputs[task]['continuous'].items():
                    if not torch.isfinite(value).all():
                        raise FloatingPointError(f'Epoch {epoch}, batch {idx}, nonfinite output {task}/{name}')
            losses = self.loss_fn(
                outputs=outputs,
                recognition_targets=batch['recognition_targets'],
                forecast_targets=batch['forecast_targets'],
            )
            loss = losses['total_loss']
            if not torch.isfinite(loss).all():
                raise FloatingPointError(f'Epoch {epoch}, batch {idx}, nonfinite loss')
            loss.backward()
            max_grad = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip or float('inf'),
                error_if_nonfinite=True,
            )
            if not torch.isfinite(max_grad):
                raise FloatingPointError(f'Epoch {epoch}, batch {idx}, invalid gradient norm')
            self.optimizer.step()
            for name, param in self.model.named_parameters():
                if not torch.isfinite(param).all():
                    raise FloatingPointError(f'Epoch {epoch}, batch {idx}, nonfinite parameter AFTER optimizer step: {name}')
            for key in sums:
                sums[key] += float(losses['total_loss' if key == 'loss' else key].detach().item())
            count += 1
            if count % 250 == 0:
                print(f'TRAIN epoch={epoch} batches={count} mean_loss={sums["loss"]/count:.6f} '
                      f'grad_norm={max_grad.item():.6g}', flush=True)
        if count == 0:
            raise RuntimeError('No training batches')
        return {key: val/count for key, val in sums.items()}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = MSEMultiTaskLoss()

    @torch.no_grad()
    def eval_one_epoch(self, val_loader, epoch, total_epochs):
        self.model.eval()
        stats = {task: {name: RegressionMoments() for name in ('valence', 'arousal')}
                 for task in ('recognition', 'forecasting')}
        batches = 0
        for batch_idx, batch in enumerate(val_loader):
            if self.max_val_batches is not None and batch_idx >= self.max_val_batches:
                break
            batch = self._to_device(batch)
            outputs = self.model(batch)
            for task in ('recognition', 'forecasting'):
                for name, value in outputs[task]['continuous'].items():
                    if not torch.isfinite(value).all():
                        raise FloatingPointError(f'Validation epoch {epoch}, batch {batch_idx}, nonfinite {task}/{name}')
            for task, target_name in (('recognition', 'recognition_targets'),
                                      ('forecasting', 'forecast_targets')):
                for name in ('valence', 'arousal'):
                    pred = outputs[task]['continuous'][name]
                    truth = batch[target_name]['continuous'][name]
                    if pred.shape != truth.shape:
                        raise ValueError(f'{task}/{name} shape mismatch: {pred.shape} != {truth.shape}')
                    stats[task][name].add(pred, truth)
            batches += 1
        if not batches:
            raise RuntimeError('No validation batches')
        metrics = {task: {name: entry.metrics() for name, entry in labels.items()}
                   for task, labels in stats.items()}
        rec_mse = sum(entry['mse'] for entry in metrics['recognition'].values())
        fore_mse = sum(entry['mse'] for entry in metrics['forecasting'].values())
        score = rec_mse + fore_mse
        if not __import__("math").isfinite(score):
            raise FloatingPointError(f"Validation epoch {epoch}: nonfinite global MSE")
        step_scheduler(self.scheduler, validation_loss=score)
        if score < self.best_val_loss:
            self.best_val_loss, self.wait = score, 0
            if self.save_path:
                self.save_checkpoint(path=self.save_path, epoch=epoch,
                                     train_loss=None, val_loss=score)
        else:
            self.wait += 1
        print(f'FULL VALIDATION epoch={epoch}: n={metrics["recognition"]["valence"]["n"]} '
              f'global_MSE={score:.6f} rec_MSE={rec_mse:.6f} fore_MSE={fore_mse:.6f}', flush=True)
        for task in metrics:
            for label, m in metrics[task].items():
                print(f' {task}/{label}: CCC={m["ccc"]:.4f}, RMSE={m["rmse"]:.4f}, '
                      f'pred_std={m["prediction_std"]:.4f}', flush=True)
        return {'loss': score, 'recognition_loss': rec_mse,
                'forecasting_loss': fore_mse, 'metrics': metrics,
                'validation_batches': batches}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--train-batches', type=int, default=0,
                        help='0 = entire training split')
    parser.add_argument('--val-batches', type=int, default=0,
                        help='0 = entire validation split; keep zero for scientific runs')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--run-base', default='runs/case_b2_fixed_mse_stability')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--clip', type=float, default=0.5)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.train_batches < 0 or args.val_batches < 0:
        parser.error('Invalid epoch, batch or batch-size argument')
    cfg = build_ablation_config()
    cfg['training'].update({'epochs': args.epochs, 'batch_size': args.batch_size,
                            'max_train_batches': args.train_batches or None,
                            'max_val_batches': args.val_batches or None})
    cfg['experiment']['name'] = 'case_b2_fixed_mse_global_val'
    cfg['experiment']['description'] = 'No-history B2, fixed fusion optimizer, MSE train, global-MSE checkpoint selection'
    if args.lr <= 0 or args.clip <= 0:
        parser.error('Learning rate and clipping must be positive')
    cfg['optimizer']['lr'] = args.lr
    cfg['training']['gradient_clip'] = args.clip
    cfg['loss']['method'] = 'mse_only'
    cfg['validation_selection'] = 'global_sum_of_channel_mse_all_windows'
    from datetime import datetime, timezone
    run = Path(args.run_base) / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True, exist_ok=False)
    (run/'config_used.json').write_text(json.dumps(cfg, indent=2)+'\n')
    ns = to_namespace(cfg)
    set_seed(seed=ns.runtime.seed, deterministic=ns.runtime.deterministic)
    device = torch.device('cuda' if torch.cuda.is_available() and ns.runtime.device in ('auto','cuda')
                          else 'cpu')
    print(f'Run={run} Device={device}', flush=True)
    train_loader, val_loader = build_dataloaders(ns, device)
    # Critical causal contract: current recognition label is last observed VA.
    probe = next(iter(train_loader))
    for name in ('valence', 'arousal'):
        expected = probe['continuous_labels'][name][:, -1]
        observed = probe['recognition_targets']['continuous'][name]
        if observed.shape != expected.shape or not torch.allclose(observed, expected, atol=1e-6):
            raise RuntimeError(f'Recognition target for {name} is not last-observed VA. Stop: check CASE adapter.')
    print('PASS: current-time recognition labels match final observed VA; no-history input is disabled', flush=True)
    model = FixedNoHistoryB2(ns)
    optimizer = build_optimizer(model, ns)
    registered = {id(p) for group in optimizer.param_groups for p in group['params']}
    missing = [name for name, p in model.named_parameters() if p.requires_grad and id(p) not in registered]
    if missing:
        raise RuntimeError(f'Optimizer missing trainable parameters: {missing}')
    trainer = GlobalMSETrainer(model=model, optimizer=optimizer,
                               scheduler=build_scheduler(optimizer, ns), device=device,
                               patience=ns.training.patience,
                               save_path=str(run/'best.pt'), use_tqdm=True,
                               gradient_clip=ns.training.gradient_clip,
                               max_train_batches=ns.training.max_train_batches,
                               max_val_batches=ns.training.max_val_batches)
    print(f'STABILITY GUARDS ENABLED: lr={args.lr} gradient_clip={args.clip}', flush=True)
    history = trainer.fit(train_loader=train_loader, val_loader=val_loader, epochs=args.epochs)
    (run/'history_final.json').write_text(json.dumps(history, indent=2)+'\n')
    print('SUCCESS run:', run, flush=True)


if __name__ == '__main__':
    main()
