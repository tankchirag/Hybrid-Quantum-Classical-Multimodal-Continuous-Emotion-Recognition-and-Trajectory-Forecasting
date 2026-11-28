"""Isolated, subject-disjoint CASE four-method CV runner.

Does NOT replace the repository's original entry points, models or checkpoints.
Commands:
 python -m training.train_case_four_method_cv --method b1 --fold 1
 python -m training.train_case_four_method_cv --method b1 --fold 1 --smoke
 python -m training.train_case_four_method_cv --method b1 --fold 1 --train

Preflight defaults to no dataset load/model construction. Smoke uses one GPU batch;
train uses all development train/validation windows and independent run directory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_configuration(method, info, args):
    from utils.config import load_config
    from copy import deepcopy
    if method == 'b2':
        from training.train_case_b2_fixed_no_history import build_ablation_config
        cfg = build_ablation_config()
    else:
        cfg = deepcopy(load_config(f'configs/case_{method}.yaml'))
    if cfg['dataset']['name'].lower() != 'case':
        raise RuntimeError('Expected CASE dataset')
    cfg['dataset']['root_dir'] = info['fold_root']
    cfg['history']['enabled'] = False
    cfg['history_encoder']['enabled'] = False
    cfg['training'].update({'batch_size': args.batch_size, 'epochs': args.epochs,
                            'max_train_batches': None, 'max_val_batches': None,
                            'gradient_clip': args.clip})
    cfg['model']['dropout'] = 0.1  # common regularization across fusion variants
    cfg['optimizer']['name'] = 'adamw'
    cfg['optimizer']['lr'] = args.lr
    cfg['optimizer']['weight_decay'] = args.weight_decay
    cfg['scheduler']['name'] = 'reduce_on_plateau'
    cfg['scheduler']['patience'] = 3
    cfg['runtime']['seed'] = args.seed
    cfg['runtime']['device'] = 'cuda'
    cfg['loss']['method'] = 'mse_only'
    cfg.setdefault('experiment', {})['name'] = f'case_four_method_{method}_fold{args.fold}'
    cfg['validation_selection'] = 'global_sum_of_channel_mse_all_windows'
    if cfg['forecasting']['horizon'] != 20 or cfg['forecasting'].get('forecast_horizon') != 20:
        raise RuntimeError('Expected 20 future points in every model')
    if cfg['history']['history_length'] != 120:
        raise RuntimeError('Expected history of 120 samples')
    if set(k for k, v in cfg['modalities'].items() if v) != {
        'ecg','bvp','gsr','rsp','skt','emg_zygo','emg_coru','emg_trap'}:
        raise RuntimeError('Modalities differ from the verified eight CASE channels')
    if cfg['continuous_labels'] != ['valence', 'arousal']:
        raise RuntimeError('Continuous targets do not match verified B2 task')
    return cfg


def make_model(method, ns):
    if method == 'b1':
        from models.wrappers.baseline_b1 import BaselineB1
        return BaselineB1(ns)
    if method == 'b2':
        from training.train_case_b2_fixed_no_history import FixedNoHistoryB2
        return FixedNoHistoryB2(ns)
    if method == 'b3':
        from models.wrappers.baseline_b3 import BaselineB3
        return BaselineB3(ns)
    if method == 'qsal':
        from models.wrappers.qsal import QSAL
        return QSAL(ns)
    raise ValueError(method)


def check_registration(model, optimizer):
    registered = {id(p) for g in optimizer.param_groups for p in g['params']}
    missing = [name for name, p in model.named_parameters()
               if p.requires_grad and id(p) not in registered]
    if missing:
        raise RuntimeError(f'Unregistered trainable model parameters: {missing[:15]}')
    return len(registered)


def check_model_no_history(model):
    if getattr(model, 'history_enabled', None) is not False:
        raise RuntimeError('Ground-truth VA history remains ENABLED')
    if getattr(model, 'history_encoder', None) is not None:
        raise RuntimeError('History encoder unexpectedly exists')
    if getattr(model, 'history_injection', None) is not None:
        raise RuntimeError('History injection unexpectedly exists')


def validate_batch(batch, outputs, torch):
    if 'modalities' not in batch or not batch['modalities']:
        raise RuntimeError('No physiological modalities')
    for label in ('valence', 'arousal'):
        latest = batch['continuous_labels'][label][:, -1]
        current = batch['recognition_targets']['continuous'][label]
        if current.shape != latest.shape or not torch.allclose(current, latest, atol=1e-6):
            raise RuntimeError(f'{label} current target is not last observed VA')
        for task, target in [('recognition', 'recognition_targets'),
                             ('forecasting', 'forecast_targets')]:
            pred = outputs[task]['continuous'][label]
            truth = batch[target]['continuous'][label]
            if pred.shape != truth.shape:
                raise RuntimeError(f'{task}/{label} shape mismatch: {pred.shape} != {truth.shape}')
            if not torch.isfinite(pred).all() or not torch.isfinite(truth).all():
                raise RuntimeError(f'Nonfinite {task}/{label}')
        target = batch['forecast_targets']['continuous'][label]
        if target.ndim != 2 or target.shape[1] != 20:
            raise RuntimeError(f'Expected exactly 20-step future forecast target for {label}, got {tuple(target.shape)}')


def run(args, info):
    import torch
    from training.base_training_script import build_dataloaders, to_namespace
    from training.optimizer import build_optimizer
    from training.scheduler import build_scheduler
    from training.train_case_b2_fixed_mse_safe import GlobalMSETrainer, MSEMultiTaskLoss
    from utils.seed import set_seed

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available; submit via GPU SLURM partition')
    cfg = build_configuration(args.method, info, args)
    ns = to_namespace(cfg)
    set_seed(seed=ns.runtime.seed, deterministic=ns.runtime.deterministic)
    device = torch.device('cuda')
    # The same CASE fold, sample count and collator for all methods.
    train_loader, val_loader = build_dataloaders(ns, device)
    if len(train_loader.dataset) != 28560 or len(val_loader.dataset) != 7140:
        raise RuntimeError('Incorrect number of training/validation windows')
    model = make_model(args.method, ns).to(device)
    check_model_no_history(model)
    optimizer = build_optimizer(model, ns)
    initial_param_names = set(dict(model.named_parameters()))
    initial_registered = check_registration(model, optimizer)
    batch = next(iter(train_loader))
    # Model forward uses physiology only; disabled label-history modules enforced above.
    def to_device(obj):
        if torch.is_tensor(obj): return obj.to(device)
        if isinstance(obj, dict): return {k: to_device(v) for k, v in obj.items()}
        if isinstance(obj, list): return [to_device(v) for v in obj]
        if isinstance(obj, tuple): return tuple(to_device(v) for v in obj)
        return obj
    sample = to_device(batch)
    if args.smoke:
        # Quantum cost is often substantial: exercise a single example, not an entire batch.
        def one(obj):
            if torch.is_tensor(obj) and obj.ndim: return obj[:1]
            if isinstance(obj, dict): return {k: one(v) for k, v in obj.items()}
            if isinstance(obj, list): return obj[:1]
            return obj
        sample = one(sample)
    model.train()
    outputs = model(sample)
    validate_batch(sample, outputs, torch)
    if initial_param_names != set(dict(model.named_parameters())):
        raise RuntimeError('Model creates parameters in forward AFTER optimizer construction')
    check_registration(model, optimizer)
    losses = MSEMultiTaskLoss()(outputs, sample['recognition_targets'], sample['forecast_targets'])
    if not torch.isfinite(losses['total_loss']):
        raise RuntimeError('Nonfinite objective')
    optimizer.zero_grad(set_to_none=True)
    losses['total_loss'].backward()
    gradients = [(n, p.grad) for n, p in model.named_parameters() if p.requires_grad]
    if not any(g is not None for _,g in gradients):
        raise RuntimeError('No trainable gradients')
    bad_grad = [n for n, g in gradients if g is not None and not torch.isfinite(g).all()]
    if bad_grad: raise RuntimeError(f'Nonfinite gradients: {bad_grad[:10]}')
    # Check the fusion-specific trainable tensors receive a gradient.
    fusion_grads = [n for n, g in gradients if n.startswith('fusion.') and g is not None and g.abs().sum().item() > 0]
    if not fusion_grads:
        raise RuntimeError('Fusion mechanism has no nonzero gradient')
    print(f'FORWARD/BACKWARD PASS: method={args.method}, fold={args.fold}, '
          f'loss={losses["total_loss"].item():.6f}, registered={initial_registered}, '
          f'fusion_grad_tensors={len(fusion_grads)}', flush=True)
    if args.smoke:
        report = {'method': args.method, 'fold': args.fold, 'status': 'SMOKE_PASS',
                  'validation_participants': info['validation_subject_ids'],
                  'registered_parameter_groups': initial_registered,
                  'nonzero_fusion_gradient_tensors': fusion_grads,
                  'smoke_loss': float(losses['total_loss'].detach().item()),
                  'full_validation_run': False, 'test_set_accessed': False}
        output = Path(args.output_dir)/'smoke'/f'{args.method}_fold_{args.fold:02d}.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2)+'\n')
        print('SMOKE PASS:',output,flush=True)
        return
    # No optimizer step on the probe; training begins at the verified initialized weights.
    model.zero_grad(set_to_none=True)
    run_dir = (Path(args.output_dir)/args.method/f'fold_{args.fold:02d}'/
               datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir/'config_used.json').write_text(json.dumps(cfg, indent=2)+'\n')
    protocol = {'method':args.method,'fold_info':info,'fold_manifest':str(args.fold_manifest.resolve()),
                'manifest_sha256':digest(args.fold_manifest), 'seed':args.seed,
                'model_class':type(model).__name__, 'no_ground_truth_history':True,
                'validation_selection':'global_sum_of_channel_mse_all_windows',
                'loss':'sum of four channel MSE terms','test_set_accessed':False,
                'source_hashes':{p:digest(p) for p in [
                   'training/train_case_four_method_cv.py',
                   'training/train_case_b2_fixed_mse_safe.py',
                   'training/train_case_b2_fixed_no_history.py',
                   f'models/wrappers/{"baseline_"+args.method if args.method!="qsal" else "qsal"}.py']}}
    (run_dir/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    trainer = GlobalMSETrainer(model=model, optimizer=optimizer,
        scheduler=build_scheduler(optimizer,ns), device=device,
        patience=ns.training.patience, save_path=str(run_dir/'best.pt'),
        use_tqdm=True, gradient_clip=ns.training.gradient_clip,
        max_train_batches=None, max_val_batches=None)
    history = trainer.fit(train_loader=train_loader,val_loader=val_loader,epochs=args.epochs)
    (run_dir/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    if not (run_dir/'best.pt').is_file(): raise RuntimeError('Missing best checkpoint')
    best = min(history,key=lambda x:x['val']['loss'])
    if best['val'].get('validation_batches', 0) < 893:
        raise RuntimeError('Validation appears truncated')
    metrics = best['val']['metrics']
    summary={'method':args.method,'fold':args.fold,'best_epoch':best['epoch'],
             'best_val_combined_mse':best['val']['loss'], 'best_val_metrics':metrics,
             'run_dir':str(run_dir),'training_subject_ids':info['training_subject_ids'],
             'validation_subject_ids':info['validation_subject_ids'],'test_set_accessed':False}
    (run_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('TRAIN SUCCESS:',args.method, 'fold', args.fold,'best MSE',best['val']['loss'],
          'run',run_dir,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['b1','b2','b3','qsal'],required=True)
    p.add_argument('--fold',choices=list(range(1,6)),required=True,type=int)
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--smoke',action='store_true')
    modes.add_argument('--train',action='store_true')
    p.add_argument('--fold-manifest',type=Path,default=Path('runs/case_b2_cv_folds/20261126/fold_manifest.json'))
    p.add_argument('--output-dir',type=Path,default=Path('runs/case_four_method_cv'))
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--clip',type=float,default=0.5)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.epochs<1 or args.batch_size<1 or args.lr<=0 or args.weight_decay<0 or args.clip<=0 or not all(map(math.isfinite,[args.lr,args.weight_decay,args.clip])):
        p.error('Invalid optimization settings')
    from training.train_case_b2_grouped_cv import verify_fold
    info=verify_fold(args.fold_manifest,args.fold)
    cfg=build_configuration(args.method,info,args)
    print('PREFLIGHT PASS:',json.dumps({'method':args.method, **info,
          'loss':'mse_only','history_enabled':False,'epochs':args.epochs,
          'batch_size':args.batch_size,'lr':args.lr,'weight_decay':args.weight_decay,
          'training_batches_limit':None,'validation_batches_limit':None},sort_keys=True),flush=True)
    if not args.smoke and not args.train:
        print('READ-ONLY: no GPU allocation, model construction or training',flush=True)
        return
    run(args,info)

if __name__ == '__main__': main()
