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
python3 scripts/sim/run_pH_citric_pydae.py
```

## Run scripts

The procedure for running scripts such as `run_pH_citric_pydae.py` consists of the following steps:

- All parameters for the simulation are read in from the .json file `pH_T_O2_citric/params.json` (inside the `resources` folder).
- The initial conditions and scales for each quantity are extracted. We then rescale the dimensional initial conditions using the scalings.
- The constant cation concentration is calculated using the initial conditions on $\text{pH}$ and $c_{\text{CA}}$, the citric acid concentration.
- We build the model using the `pydae` `Builder` class.
- We then run the model a number of times (to assess the speed). Inside this loop, different parameter values can be varied.
- We then rescale all variables to their dimensional equivalents and plot the simulation. This plot is saved inside `resources/initial/pH_T_O2_citric/time_traces_pydae.png`.
