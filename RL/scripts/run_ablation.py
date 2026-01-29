#!/usr/bin/env python
"""Ablation Study Runner for MARL Topology Optimization.

This script runs a series of ablation experiments to verify the necessity of specific modules:
- Baseline: All modules enabled
- No_GNN: Disable GNN in Actor (use linear projection instead)
- No_RNN: Disable LSTM in Actor (use feedforward instead)
- No_Safety: Disable connectivity safety check in MergerSafety
- Or_Policy: Use 'min' cross-policy instead of 'and' for edge merging

Usage:
    python scripts/run_ablation.py --cfg configs/config.yaml
    python scripts/run_ablation.py --cfg configs/config.yaml --experiments Baseline No_GNN
    python scripts/run_ablation.py --cfg configs/config.yaml --episodes 500 --parallel 2
"""

from __future__ import annotations

import os
import sys
import copy
import argparse
import datetime
from typing import Dict, List, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure project root on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import yaml
except ImportError:
    yaml = None


# ============================================================================
# Ablation Experiment Definitions
# ============================================================================

ABLATION_EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    'Baseline': {
        # All modules enabled (default)
        'description': 'Baseline with all modules enabled',
        'overrides': {
            'ablation': {
                'enable_gnn': True,
                'enable_rnn': True,
                'enable_safety_check': True,
                'force_flat_action': False,
            }
        }
    },
    'No_GNN': {
        # Disable GNN, use linear projection + mean pooling
        'description': 'Disable GNN in Actor (use linear projection)',
        'overrides': {
            'ablation': {
                'enable_gnn': False,
                'enable_rnn': True,
                'enable_safety_check': True,
                'force_flat_action': False,
            }
        }
    },
    'No_RNN': {
        # Disable LSTM, use feedforward projection
        'description': 'Disable LSTM in Actor (use feedforward)',
        'overrides': {
            'ablation': {
                'enable_gnn': True,
                'enable_rnn': False,
                'enable_safety_check': True,
                'force_flat_action': False,
            }
        }
    },
    'No_Safety': {
        # Disable connectivity safety check in deletion
        'description': 'Disable connectivity check in MergerSafety',
        'overrides': {
            'ablation': {
                'enable_gnn': True,
                'enable_rnn': True,
                'enable_safety_check': False,
                'force_flat_action': False,
            }
        }
    },
    'Or_Policy': {
        # Use 'min' cross-policy instead of 'and'
        # 'and' policy: requires >= 2 DIFFERENT agents to propose same cross-edge
        # 'min' policy: requires >= 2 proposals (can be from same agent), uses min score
        'description': 'Use min cross-policy (relaxed multi-agent agreement)',
        'overrides': {
            'ablation': {
                'enable_gnn': True,
                'enable_rnn': True,
                'enable_safety_check': True,
                'force_flat_action': False,
            },
            'merge': {
                'cross_policy': 'min',
            }
        }
    },
}


def load_base_config(cfg_path: str) -> Dict:
    """Load base configuration from YAML file."""
    if not cfg_path or not os.path.exists(cfg_path):
        raise FileNotFoundError(f'Config file not found: {cfg_path}')
    if yaml is None:
        raise ImportError('PyYAML is required. Install with: pip install pyyaml')
    with open(cfg_path, 'r', encoding='utf-8-sig') as f:
        cfg = yaml.safe_load(f)
    return cfg if isinstance(cfg, dict) else {}


def apply_overrides(cfg: Dict, overrides: Dict) -> Dict:
    """Apply nested overrides to config dictionary."""
    cfg = copy.deepcopy(cfg)
    for section, values in overrides.items():
        if isinstance(values, dict):
            if section not in cfg or not isinstance(cfg[section], dict):
                cfg[section] = {}
            cfg[section].update(values)
        else:
            cfg[section] = values
    return cfg


def run_single_experiment(
    experiment_name: str,
    base_cfg_path: str,
    experiment_def: Dict,
    episodes: Optional[int] = None,
    max_steps: Optional[int] = None,
    output_dir: str = 'result_save',
    no_viz: bool = False,
) -> Dict[str, Any]:
    """Run a single ablation experiment.
    
    Args:
        experiment_name: Name of the experiment (e.g., 'Baseline', 'No_GNN')
        base_cfg_path: Path to base config YAML
        experiment_def: Experiment definition with 'overrides' and 'description'
        episodes: Override number of episodes (optional)
        max_steps: Override max steps per episode (optional)
        output_dir: Base output directory
        no_viz: Disable visualization
        
    Returns:
        Dictionary with experiment results
    """
    # Import here to avoid circular imports and allow parallel execution
    from scripts.train import load_cfg, run
    
    print(f"\n{'='*60}")
    print(f"[Ablation] Starting experiment: {experiment_name}")
    print(f"[Ablation] Description: {experiment_def.get('description', 'N/A')}")
    print(f"{'='*60}\n")
    
    # Load and configure
    cfg = load_cfg(base_cfg_path)
    cfg = apply_overrides(cfg, experiment_def.get('overrides', {}))
    
    # Apply CLI overrides
    run_overrides: Dict[str, Any] = {'training': {}, 'viz': {}}
    
    # Set run name with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"ablation_{experiment_name}_{timestamp}"
    run_overrides['training']['run_name'] = run_name
    
    if episodes is not None:
        run_overrides['training']['episodes'] = episodes
    if max_steps is not None:
        run_overrides['training']['max_steps_per_episode'] = max_steps
    if no_viz:
        run_overrides['viz']['enable'] = False
    
    # Merge with experiment overrides
    final_overrides = apply_overrides(experiment_def.get('overrides', {}), run_overrides)
    
    try:
        success = run(base_cfg_path, final_overrides)
        result = {
            'experiment': experiment_name,
            'run_name': run_name,
            'status': 'success' if success else 'failed',
            'output_dir': os.path.join(output_dir, run_name),
        }
    except Exception as e:
        result = {
            'experiment': experiment_name,
            'run_name': run_name,
            'status': 'error',
            'error': str(e),
        }
        print(f"[Ablation] ERROR in {experiment_name}: {e}")
    
    print(f"\n[Ablation] Completed: {experiment_name} -> {result['status']}")
    return result


def run_ablation_study(
    cfg_path: str,
    experiments: Optional[List[str]] = None,
    episodes: Optional[int] = None,
    max_steps: Optional[int] = None,
    output_dir: str = 'result_save',
    no_viz: bool = False,
    parallel: int = 1,
) -> List[Dict[str, Any]]:
    """Run ablation study with specified experiments.
    
    Args:
        cfg_path: Path to base config YAML
        experiments: List of experiment names to run (None = all)
        episodes: Override number of episodes
        max_steps: Override max steps per episode
        output_dir: Base output directory
        no_viz: Disable visualization
        parallel: Number of parallel experiments (1 = sequential)
        
    Returns:
        List of experiment results
    """
    # Determine which experiments to run
    if experiments is None or len(experiments) == 0:
        experiments = list(ABLATION_EXPERIMENTS.keys())
    
    # Validate experiment names
    invalid = [e for e in experiments if e not in ABLATION_EXPERIMENTS]
    if invalid:
        print(f"[Ablation] WARNING: Unknown experiments will be skipped: {invalid}")
        experiments = [e for e in experiments if e in ABLATION_EXPERIMENTS]
    
    if not experiments:
        print("[Ablation] No valid experiments to run.")
        return []
    
    print(f"\n{'#'*60}")
    print(f"# ABLATION STUDY")
    print(f"# Base config: {cfg_path}")
    print(f"# Experiments: {', '.join(experiments)}")
    print(f"# Episodes: {episodes or 'default'}")
    print(f"# Parallel: {parallel}")
    print(f"{'#'*60}\n")
    
    results: List[Dict[str, Any]] = []
    
    if parallel <= 1:
        # Sequential execution
        for exp_name in experiments:
            exp_def = ABLATION_EXPERIMENTS[exp_name]
            result = run_single_experiment(
                experiment_name=exp_name,
                base_cfg_path=cfg_path,
                experiment_def=exp_def,
                episodes=episodes,
                max_steps=max_steps,
                output_dir=output_dir,
                no_viz=no_viz,
            )
            results.append(result)
    else:
        # Parallel execution using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = {}
            for exp_name in experiments:
                exp_def = ABLATION_EXPERIMENTS[exp_name]
                future = executor.submit(
                    run_single_experiment,
                    experiment_name=exp_name,
                    base_cfg_path=cfg_path,
                    experiment_def=exp_def,
                    episodes=episodes,
                    max_steps=max_steps,
                    output_dir=output_dir,
                    no_viz=no_viz,
                )
                futures[future] = exp_name
            
            for future in as_completed(futures):
                exp_name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        'experiment': exp_name,
                        'status': 'error',
                        'error': str(e),
                    })
    
    # Summary
    print(f"\n{'#'*60}")
    print(f"# ABLATION STUDY SUMMARY")
    print(f"{'#'*60}")
    for r in results:
        status_icon = '✓' if r['status'] == 'success' else '✗'
        print(f"  {status_icon} {r['experiment']}: {r['status']}")
        if r.get('output_dir'):
            print(f"      -> {r['output_dir']}")
        if r.get('error'):
            print(f"      ERROR: {r['error']}")
    print()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run ablation study for MARL topology optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available experiments:
  Baseline   - All modules enabled (default configuration)
  No_GNN     - Disable GNN in Actor (use linear projection instead)
  No_RNN     - Disable LSTM in Actor (use feedforward instead)
  No_Safety  - Disable connectivity check in MergerSafety
  Or_Policy  - Use 'min' cross-policy (relaxed multi-agent agreement)

Examples:
  python scripts/run_ablation.py --cfg configs/config.yaml
  python scripts/run_ablation.py --cfg configs/config.yaml --experiments Baseline No_GNN
  python scripts/run_ablation.py --cfg configs/config.yaml --episodes 500 --no-viz
        """
    )
    parser.add_argument('--cfg', type=str, default=None,
                        help='Path to base config YAML (required unless --list)')
    parser.add_argument('--experiments', type=str, nargs='*', default=None,
                        help='List of experiments to run (default: all)')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of training episodes (default: 100)')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Override max steps per episode')
    parser.add_argument('--output-dir', type=str, default='result_save',
                        help='Base output directory (default: result_save)')
    parser.add_argument('--no-viz', action='store_true',
                        help='Disable visualization')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel experiments (default: 1)')
    parser.add_argument('--list', action='store_true',
                        help='List available experiments and exit')
    
    args = parser.parse_args()
    
    # --list doesn't require --cfg
    if args.list:
        print("\nAvailable ablation experiments:")
        print("-" * 50)
        for name, exp_def in ABLATION_EXPERIMENTS.items():
            print(f"  {name:12s} - {exp_def.get('description', 'N/A')}")
        print()
        return
    
    # --cfg is required for running experiments
    if not args.cfg:
        parser.error("--cfg is required when running experiments")
    
    results = run_ablation_study(
        cfg_path=args.cfg,
        experiments=args.experiments,
        episodes=args.episodes,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        no_viz=args.no_viz,
        parallel=args.parallel,
    )
    
    # Exit with error code if any experiment failed
    success_count = sum(1 for r in results if r['status'] == 'success')
    if success_count < len(results):
        sys.exit(1)


if __name__ == '__main__':
    main()
