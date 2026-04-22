# macOS Installation Guide

This guide documents the macOS setup that was validated for the AntifragiCity /
SAS project on Apple Silicon.

It covers:

- XQuartz installation
- SUMO / `sumo-gui` installation
- Python environment setup with Conda
- TraCI / `sumolib` installation
- verification commands
- known limitations on Apple Silicon

## Scope

This guide is for macOS users working on this repository.

Validated environment:

- macOS on Apple Silicon
- XQuartz `2.8.5`
- SUMO `1.26.0`
- Conda environment: `vEnvAC`

## 1. Install XQuartz

Download and install XQuartz from the official site:

- https://www.xquartz.org/

Install:

- `XQuartz-2.8.5.pkg`

After installation:

1. Log out of macOS.
2. Log back in.

This step matters because XQuartz integrates with the active desktop login
session.

## 2. Install SUMO

Download and install SUMO from the official SUMO downloads page:

- https://sumo.dlr.de/docs/Downloads.php

Use the official macOS package:

- `sumo-1.26.0.pkg`

Do not use Homebrew for SUMO on macOS for this project. The SUMO team
explicitly discourages relying on the Homebrew bottles for current support.

## 3. Set shell environment

Add the following lines to `~/.zshrc`:

```bash
export SUMO_HOME="/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/share/sumo"
export PATH="$SUMO_HOME/bin:/opt/X11/bin:$PATH"
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
```

Reload your shell:

```bash
source ~/.zshrc
```

## 4. Verify the system installation

Run:

```bash
which sumo
which sumo-gui
which netconvert
echo "$SUMO_HOME"
sumo --version
sumo-gui --version
netconvert --version
```

Expected result:

- all three binaries resolve correctly
- all three report SUMO `1.26.0`

## 5. Verify XQuartz

Start XQuartz and verify it can open an X11 application:

```bash
open -a XQuartz
open -a Terminal
/opt/X11/bin/xterm
```

Expected result:

- an `xterm` window opens

If `xterm` does not open, stop here and fix XQuartz before proceeding.

## 6. Create the Conda environment

This project was validated with a Conda environment named `vEnvAC`:

```bash
conda create -n vEnvAC python=3.14
conda activate vEnvAC
python -m pip install --upgrade pip setuptools wheel
```

## 7. Install Python dependencies

From the repository root:

```bash
cd /path/to/sumo-accident-simulation
pip install -e .
pip install "traci==1.26.0" "sumolib==1.26.0"
pip install -r "$SUMO_HOME/tools/requirements.txt"
```

Why install `traci` and `sumolib` explicitly:

- they are the Python-side SUMO bindings used by this project
- pinning them to `1.26.0` keeps them aligned with the installed SUMO version

## 8. Verify the Python environment

Run:

```bash
python -c "import traci, sumolib; print(traci.__version__); print(sumolib.__file__)"
python -c "import yaml, pandas, numpy, matplotlib, seaborn, scipy, requests; print('python deps ok')"
python runner.py --help
```

Expected result:

- `traci` imports successfully
- `sumolib` imports successfully
- the standard Python dependencies import successfully
- `runner.py --help` prints usage information without import errors

## 9. First project run

Run the default configuration:

```bash
python runner.py --config config.yaml
```

If `config.yaml` sets:

```yaml
sumo:
  binary: sumo
```

the simulation runs headless.

If it sets:

```yaml
sumo:
  binary: sumo-gui
```

the simulation attempts to open the SUMO GUI.

## 10. Thessaloniki generator

The Thessaloniki generator expects `SUMO_HOME` to point to the SUMO `share/sumo`
directory, not the installer root.

This is the correct value:

```bash
export SUMO_HOME="/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/share/sumo"
```

With that in place, this should work:

```bash
python generate_thessaloniki.py --help
```

## Apple Silicon note

On Apple Silicon Macs, the installation itself works and headless SUMO runs
correctly, but `sumo-gui` rendering through XQuartz may be unreliable.

Observed behavior on Apple Silicon:

- SUMO starts
- TraCI connects
- the simulation runs
- vehicle counts and metrics update normally
- the GUI window may remain gray or repaint incorrectly
- window resizing may leave unrendered or transparent regions
- `GLXBadContext` may appear in the terminal

This appears to be an external XQuartz / OpenGL / Apple Silicon issue rather
than a repository issue.

Practical recommendation:

- use `binary: sumo` for reliable simulation runs on Apple Silicon macOS
- use `sumo-gui` on Intel macOS or Linux when visual inspection is required

## Known-good headless workflow on Apple Silicon

For production runs on Apple Silicon, use:

```yaml
sumo:
  binary: sumo
```

Then run:

```bash
python runner.py --config config.yaml
```

This path was validated and works reliably.

If you still want visual feedback while running headless, enable the Python-side
dashboard instead of relying on `sumo-gui`:

```yaml
sumo:
  binary: sumo

output:
  live_progress: true
  live_progress_refresh_steps: 300
```

Or launch it ad hoc:

```bash
python runner.py --config config.yaml --live-progress
```

This updates `live_progress.png` in the run output folder and, when the local
Matplotlib backend supports it, opens a live window with occupancy, speed,
throughput, active-accident charts, and a live network-load map. To refresh the
map every SUMO step, set `output.live_progress_refresh_steps` to the same value
as `sumo.step_length`.

## Troubleshooting

### `traci` import fails

Check:

```bash
echo "$SUMO_HOME"
echo "$PYTHONPATH"
python -c "import traci"
```

Make sure:

- `SUMO_HOME` points to `.../share/sumo`
- `PYTHONPATH` contains `$SUMO_HOME/tools`

### `sumo`, `sumo-gui`, or `netconvert` not found

Check:

```bash
echo "$PATH"
which sumo
which sumo-gui
which netconvert
```

Make sure:

- `PATH` includes `$SUMO_HOME/bin`

### `generate_thessaloniki.py` cannot find typemap or `randomTrips.py`

Check:

```bash
echo "$SUMO_HOME"
ls "$SUMO_HOME/data/typemap/osmNetconvert.typ.xml"
ls "$SUMO_HOME/tools/randomTrips.py"
```

If those files are missing, `SUMO_HOME` is wrong.

### GUI opens but stays gray on Apple Silicon

This is a known limitation for this setup.

Recommended action:

- switch to headless mode with `binary: sumo`
- use another machine for GUI-based inspection if required

## References

- SUMO downloads: https://sumo.dlr.de/docs/Downloads.php
- SUMO install docs: https://sumo.dlr.de/docs/Installing/index.html
- TraCI docs: https://sumo.dlr.de/docs/TraCI.html
- XQuartz: https://www.xquartz.org/
- XQuartz Apple Silicon rendering issue: https://github.com/XQuartz/XQuartz/issues/31
