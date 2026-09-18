# RIS prototype

RIS currently contains a **minimal numerical prototype**, not a production or institutional-grade financial system.

## Implemented today

The repository has one Python entry point, `main.py`, which:

- defines a two-variable toy objective using sine and cosine;
- approximates its Hessian with centered finite differences;
- evaluates the Hessian at a fixed example point;
- prints the resulting matrix.

The current code uses NumPy. A broader dependency list is retained in `requirements.txt`, but those packages are **not evidence that the corresponding framework features have been implemented**.

## Run the current prototype

```bash
python -m pip install 'numpy>=1.26,<3'
python main.py
```

The repository CI performs an exact-source checkout, compiles the Python entry point, installs the bounded NumPy runtime dependency, and executes the current demo.

## Current evidence boundary

This repository does **not** currently establish:

- an institutional-grade risk or IRR stabilization platform;
- production financial infrastructure;
- validated forecasting or optimization performance;
- trading or investment returns;
- a complete research benchmark or paper result.

Any future claims should be tied to checked-in implementation, tests, retained evaluation artifacts, and a clearly defined financial/research protocol.
