# Dual Control: On Exploration–Exploitation in Linear Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MATLAB code for the numerical case studies in

> T. J. Meijer and A. Rantzer, "Dual Control: On Exploration–Exploitation in Linear Systems," *Annual Review of Control, Robotics, and Autonomous Systems*, vol. 10, 2027.

See the paper for details.

## Usage

Requires Manopt (http://manopt.org/).

Run `Main.m`. It simulates the three controllers and plots the results (Figure 4 of the paper). The model, noise type and parameters are set at the top of the file.

## Files

- `Main.m`: simulation and plotting (entry point)
- `wRLS_CE_LQR.m`: weighted recursive least squares CE-LQR
- `CEC.m`: regret rate minimizing certainty-equivalence controller
- `Minimax.m`: minimax optimal dual controller
- `dependencies/`: helper functions

## Citation

```bibtex
@article{meijer2027dual,
  title   = {Dual Control: On Exploration--Exploitation in Linear Systems},
  author  = {T.J.~{Meijer} and A.~{Rantzer}},
  journal = {Annual Review of Control, Robotics, and Autonomous Systems},
  volume  = {10},
  pages   = {1--26},
  year    = {2027}
}
```

## Contact and license

Tomas J. Meijer (tomas.meijer@control.lth.se) and Anders Rantzer, Department of Automatic Control, Lund University.
Released under the [MIT License](LICENSE).
