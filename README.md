# Fermentation of Cocoa Pulp

In this repository, we use the `pydae` module to solve a system of differential-algebraic equations (DAEs) modelling the fermentation of cocoa beans.

## Installation

To clone the repository, create a new directory and enter the following into the terminal:

```
git clone git@github.com:mghosh00/FermentingCocoa.git
```

From here, enter the following commands to install the repository and all its dependencies:

```
cd FermentingCocoa
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

Note that on Windows, you'll need to replace `venv/bin/activate` with `venv/Scripts/activate`. You may also need to replace `python3` with `python`.

Once the package is installed, you can run one of the toy simulations using the following commands:

```
cd fermenting_cocoa
python3 scripts/sim/run_pH_pydae.py
```
