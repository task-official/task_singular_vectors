"""Historical eight-task CLI; no separate inference or normalization code."""
from src.compat.cli import single_task_main

if __name__ == "__main__":
    single_task_main(benchmark="fft8")
