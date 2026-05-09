from pathlib import Path
from typing import Any, Dict, List, Optional

from .configs import ExperimentConfig
from .experiment import Experiment
from .experiment_tracking import (
    setup_log_directory,
)


def run_experiments(
    experiment_settings_list: List[Dict[str, Any]],
    base_experiment_settings: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    all_results: List[Dict[str, Any]] = []
    base_settings = base_experiment_settings or {}

    for settings in experiment_settings_list:
        merged_settings = ExperimentConfig.deep_merge_dict(base_settings, settings)
        experiment_config = ExperimentConfig.from_settings(merged_settings)
        experiment_config.paths, current_version_name = setup_log_directory(experiment_config.paths)

        exp_name = experiment_config.get("extras.exp_name", current_version_name)
        print("=" * 88)
        print(f"Running experiment: {exp_name}")
        print(f"Logging at: {experiment_config.paths.log_dir}")
        print(f"Model checkpoint at: {experiment_config.paths.checkpoint_dir}")

        experiment = Experiment(experiment_config)
        
        result = experiment.run(
            version_name=current_version_name,
            experiments_xlsx=experiment_config.paths.experiments_file,
            exp_name=exp_name,
        )
        all_results.append(result)

    print("=" * 88)
    print("Batch completed")
    for idx, row in enumerate(all_results, start=1):
        print(
            f"{idx}. {row['exp_name']} | version={row['version_name']} | "
            f"best_valid_acc={row['best_valid_acc']:.4f} | best_valid_loss={row['best_valid_loss']:.4f}"
        )

    return all_results
