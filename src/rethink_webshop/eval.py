"""Formal shared140 native-vLLM evaluation; identical sampling to RL validation."""
import argparse
from .run import main as run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Project-relative HF model directory')
    parser.add_argument('--output', required=True, help='New project-relative evaluation directory')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    argv = ['--actor', args.model, '--run-dir', args.output, '--eval-only']
    if args.prepare_only:
        argv.append('--prepare-only')
    run(argv)


if __name__ == '__main__':
    main()
