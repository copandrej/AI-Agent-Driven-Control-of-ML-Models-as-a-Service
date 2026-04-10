"""MLOps Agent — launcher. Selects architecture via --arch flag."""
import argparse

ARCHS = {"flow": "src.agent", "dual": "src.dual", "simple": "src.simple", "minimal": "src.minimal"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=ARCHS, default="flow", help="Agent architecture (default: flow)")
    args, _ = parser.parse_known_args()

    import importlib
    importlib.import_module(ARCHS[args.arch]).main()
