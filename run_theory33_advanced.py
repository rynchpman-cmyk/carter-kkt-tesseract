#!/usr/bin/env python3
"""Train, qualify, freeze, replay, and audit advanced Theory 3.3 variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from theory33_advanced import (
    CovariantPSDMobility,
    MobilityTheory33,
    build_composed_frozen_model,
    compare_multistep_estimators,
    create_frozen_artifact,
    explicit_margin_audit,
    fit_phase_threshold_with_uncertainty,
    load_transport_tensors,
    mobility_covariance_audit,
    qualify_residual_scale,
    replay_frozen_artifact,
    train_drift_network_from_data,
    train_mobility_network,
    train_serializable_kappa,
    write_frozen_artifact,
)
from theory33_kinetic_data import KineticDataConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--data-directory", type=Path, default=Path("data")
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("theory33_frozen_variants.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("theory33_advanced_results.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = KineticDataConfig()
    metadata_path = (
        args.data_directory / "theory33_kinetic_metadata.json"
    )
    if not metadata_path.is_file():
        subprocess.run(
            [
                sys.executable,
                "generate_theory33_kinetic_data.py",
                "--output-directory",
                str(args.data_directory),
            ],
            check=True,
        )
    data_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    data = load_transport_tensors(
        args.data_directory / "theory33_kinetic_transport.csv",
        config,
    )
    controller, controller_metrics = train_serializable_kappa(
        epochs=450 if args.quick else 800,
        samples_per_branch=257 if args.quick else 1024,
    )
    mobility, mobility_fit = train_mobility_network(
        data, epochs=600 if args.quick else 1200
    )
    drift, drift_fit = train_drift_network_from_data(
        data,
        config=config,
        epochs=400 if args.quick else 900,
    )
    phase, phase_fit = fit_phase_threshold_with_uncertainty(
        args.data_directory / "theory33_noisy_phase_trajectories.csv",
        config=config,
        bootstrap_samples=40 if args.quick else 200,
    )
    covariance = mobility_covariance_audit(mobility)

    mobility_model = MobilityTheory33(
        CovariantPSDMobility(mobility),
        data_config=config,
        learn_dynamics=False,
        boundary_certificates=False,
    )
    mobility_model.kappa_controller = controller
    mobility_audit = explicit_margin_audit(
        mobility_model,
        points=33 if args.quick else 65,
        characteristic_points=5 if args.quick else 9,
    )

    energy_scale, continuation, final_audit = qualify_residual_scale(
        drift,
        phase,
        data_config=config,
        mobility_network=mobility,
        kappa_controller=controller,
        bisection_steps=4 if args.quick else 8,
        audit_points=33 if args.quick else 65,
        characteristic_points=5 if args.quick else 9,
    )
    final_model = build_composed_frozen_model(
        controller=controller,
        mobility_network=mobility,
        drift_network=drift,
        phase=phase,
        energy_scale=energy_scale,
        data_config=config,
    )
    event_bptt = compare_multistep_estimators(
        final_model,
        horizons=(1, 4) if args.quick else (1, 4, 8, 16),
        grid_points=257 if args.quick else 1537,
        randomized_samples=128 if args.quick else 2048,
    )
    training_metrics = {
        "data": data_metadata,
        "controller": controller_metrics,
        "mobility_fit": mobility_fit,
        "drift_fit": drift_fit,
        "phase_fit": phase_fit,
        "mobility_covariance": covariance,
        "mobility_audit": mobility_audit,
        "continuation": {
            "selected_energy_scale": energy_scale,
            "attempts": continuation,
            "final_audit": final_audit,
        },
        "event_bptt": event_bptt,
    }
    artifact = create_frozen_artifact(
        data_directory=args.data_directory,
        controller=controller,
        mobility_network=mobility,
        drift_network=drift,
        phase=phase,
        energy_scale=energy_scale,
        metrics=training_metrics,
        data_config=config,
    )
    write_frozen_artifact(args.artifact, artifact)
    replay = replay_frozen_artifact(
        args.artifact,
        audit_points=33 if args.quick else 65,
        characteristic_points=5 if args.quick else 9,
    )
    passed = (
        controller_metrics["maximum_error"] < 2.0e-3
        and mobility_fit["test"]["rmse"] < 2.0e-3
        and mobility_fit["test"]["minimum_eigenvalue"] > 0.0
        and drift_fit["test"]["rmse"] < 2.0e-3
        and drift_fit["minimum_second_derivative"] > 0.0
        and phase_fit["test"]["accuracy"] > 0.90
        and (
            phase_fit["uncertainty"]["base"]["upper_95"]
            > phase_fit["uncertainty"]["base"]["lower_95"]
        )
        and bool(covariance["passed"])
        and bool(mobility_audit["qualified"])
        and bool(final_audit["qualified"])
        and bool(event_bptt["jump_aware_best_every_horizon"])
        and bool(replay["passed"])
    )
    results = {
        "format": "theory33-advanced-qualification",
        "version": 1,
        "quick": args.quick,
        "passed": passed,
        "artifact": str(args.artifact),
        "artifact_digest": artifact["variant_digest"],
        "selected_energy_scale": energy_scale,
        "training": training_metrics,
        "replay": replay,
    }
    args.output.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2))
    if not passed:
        raise RuntimeError("advanced Theory 3.3 qualification failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
