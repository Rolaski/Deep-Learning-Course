"""Lab 01 console menu; use --task 1..7 or --task all to skip the menu."""

import argparse
import importlib
import json
import os
from pathlib import Path


TASKS = {
    "1": ("Tensory i wektoryzacja", "task01"),
    "2": ("Gradienty i autograd", "task02"),
    "3": ("Regresja liniowa", "task03"),
    "4": ("Learning rate", "task04"),
    "5": ("Mini-batche i SGD", "task05"),
    "6": ("Wielomiany i regularyzacja", "task06"),
    "7": ("Dodatkowe: Rosenbrock", "extra"),
}
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def choose_task() -> str | None:
    print("\nLab 01 — wybierz zadanie:", flush=True)
    for number, (title, _) in TASKS.items():
        print(f"{number}. {title}")
    print("8. Całe laboratorium\n0. Wyjście", flush=True)
    while True:
        try:
            choice = input("Wybór [0–8]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if choice == "0":
            return None
        if choice == "8":
            return "all"
        if choice in TASKS:
            return choice
        print("Wpisz numer od 0 do 8.")


def run_tasks(selection, output, fast=False, show=False, environment=None):
    if environment is None:
        print("Przygotowanie środowiska GPU...", flush=True)
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        from deep_learning_course.environment import check_environment

        environment = check_environment()

    import keras
    import torch

    numbers = TASKS if selection == "all" else [selection]
    results = {}
    for number in numbers:
        title, module_name = TASKS[number]
        print(f"\nZadanie {number}: {title}", flush=True)
        keras.utils.set_random_seed(42)
        torch.manual_seed(42)
        module = importlib.import_module(f"deep_learning_course.lab01.{module_name}")
        task_output = output / module_name
        task_output.mkdir(parents=True, exist_ok=True)
        result = module.run(task_output, fast=fast, show=show)
        results[module_name] = result
        document = {"task": module_name, "fast": fast, "environment": environment, "results": result}
        (task_output / "results.json").write_text(
            json.dumps(document, indent=2, allow_nan=False), encoding="utf-8"
        )
        for key, value in result.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.6g}")
        print(f"Wyniki: {task_output}", flush=True)
    if selection == "all":
        output.mkdir(parents=True, exist_ok=True)
        results.update(environment=environment, fast=fast)
        (output / "results.json").write_text(
            json.dumps(results, indent=2, allow_nan=False), encoding="utf-8"
        )
    return environment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[*TASKS, "all"], help="Run once without the menu.")
    parser.add_argument("--fast", action="store_true", default=os.environ.get("DL_FAST") == "1")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/lab01")
    args = parser.parse_args(argv)
    if args.task is not None:
        run_tasks(args.task, args.output, fast=args.fast, show=args.show)
        return
    environment = None
    while (selection := choose_task()) is not None:
        environment = run_tasks(
            selection, args.output, fast=args.fast, show=args.show, environment=environment
        )


if __name__ == "__main__":
    main()
