# Screenshots

The images the main README points at. All four are captured from real output of this project rather than mocked up.

| File | What it is |
|------|------------|
| `terminal_run.png` | The console output of `pytest -q` followed by `python main.py`, start to finish |
| `notebook_overview.png` | The top of `notebooks/EDA_Indian_Air_Quality.ipynb`, showing the intro and the first data cells |
| `notebook_analysis.png` | The city ranking section of the same notebook, including the writeup of the seaborn ordering bug and the corrected chart |
| `eda_report_preview.png` | `outputs/EDA_Report.md` rendered the way GitHub renders it |

The charts themselves are not duplicated in here. The README links straight to `outputs/plots/`, which is where `main.py` writes them, so there is only ever one copy of each chart and it cannot fall out of date with the code.

## Adding your own

If you want to add screenshots of your own Colab session, `colab_setup.png` and `colab_run_output.png` are the two worth having. Part 3 of [GUIDE.md](../GUIDE.md) has the exact steps and the markdown to paste into the README.

One thing to watch: do not add the image links to the README before the files exist. GitHub renders a missing image as a broken icon, which looks worse than simply not having the screenshot.
