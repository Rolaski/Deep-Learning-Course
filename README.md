# Deep Learning Course


Python 3.12, Ubuntu on WSL2, NVIDIA GPU, PyCharm Pro.

Run in Ubuntu from the project directory:

```bash
sudo apt install python3.12-venv python3-tk
python3 -m venv .local/venv
.local/venv/bin/python -m pip install -r requirements.txt
.local/venv/bin/python -m pip install --no-deps -e .
.local/venv/bin/python -m deep_learning_course.environment
```

In PyCharm, select **Add Interpreter → On WSL → Ubuntu → Existing** and
choose `/mnt/d/Github/Deep-Learning-Course/.local/venv/bin/python`.

Code: `src/deep_learning_course/`. Original materials: `Lessons/` (local).
Datasets: `data/`. Generated results: `outputs/`.

Lab 01 console menu (tasks 1–6, extra task, all tasks, exit):

```bash
.local/venv/bin/python -m deep_learning_course.lab01
.local/venv/bin/python -m unittest discover -s tests -v
```

Use `--fast` for reduced data/epochs or `--show` to display plots.
Use `--task 3` or `--task all` to run once without the menu.
In PyCharm run module `deep_learning_course.lab01` without arguments for the menu.
Code: `task01.py`–`task06.py`, `extra.py`; shared data/plotting: `_shared.py`.
Plots and results are saved to `outputs/lab01/task01/`, etc.

The original mini-batch `lr=0.05` diverges on the full dataset. Results preserve
that observation and add a comparison using a conservative training-data bound.
