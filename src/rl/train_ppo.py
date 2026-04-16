import wandb
import argparse

def main(args):
    wandb.init(
        project="hcsla-csm",
        entity="ishaansharma102005-thapar-university",
        name=f"stage1-flat-v{args.v_desired}",
        config={
            "v_desired": args.v_desired,
            "stage": 1,
            "terrain": "flat",
            "w_velocity": 2.0,
            "w_stability": 3.0,
            "w_smooth": 0.5,
            "w_energy": 0.1,
            "w_survival": 1.0,
            "w_contact": 0.8,
        }
    )

    # TODO: replace with actual training loop
    for iteration in range(1000):
        # Placeholder — swap these with real values from legged_gym
        r_total = 0.0
        r_velocity = 0.0
        r_stability = 0.0
        fall_rate = 0.0
        tracking_error = 0.0
        d_S_mean = 0.0

        wandb.log({
            "reward/total": r_total,
            "reward/velocity": r_velocity,
            "reward/stability": r_stability,
            "metrics/fall_rate": fall_rate,
            "metrics/tracking_error_ms": tracking_error,
            "metrics/d_S_mean": d_S_mean,
            "train/iteration": iteration,
        })

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v_desired", type=float, default=0.5)
    args = parser.parse_args()
    main(args)