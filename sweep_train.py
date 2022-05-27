import argparse
import sys
from subprocess import run

from sweeps.sweep_utils import get_args_from_sweep

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-name", required=True, help="Sweep name")
    parser.add_argument("--model-num", type=int, required=True, help="Model number")
    args = parser.parse_args()

    # Get model config corresponding to model number, according to sweep number.
    # This is a list of strings like 'param=value'
    command_args = get_args_from_sweep(args.sweep_name, args.model_num)

    # Set run dir and add it to config
    run_dir = f"outputs/sweeps/sweep_{args.sweep_name}/{args.model_num}"
    command_args.append(f"hydra.run.dir={run_dir}")

    # Run
    result = run(["python", "train_object_discovery.py"] + command_args)
    sys.exit(result.returncode)
