# Multiple_SAV

**Hydrodynamic Phase-Field Modeling and Numerical Simulation of Vesicle Growth and Shrinkage**

A Julia implementation of a coupled hydrodynamic phase-field model for simulating the dynamics of lipid vesicles, including growth, shrinkage, and shape evolution under osmotic stress. The solver is designed for both 2D and 3D domains and supports multi-vesicle configurations.

---

## Table of Contents

- [Mathematical Model](#mathematical-model)
- [Numerical Methods](#numerical-methods)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Output](#output)

---

## Mathematical Model

The model couples three physical subsystems through a unified energy functional:

**Phase fields.** The vesicle membrane is represented by a diffuse-interface phase field $\phi \in [-1, 1]$, where $\phi = 1$ marks the vesicle interior and $\phi = -1$ the exterior. A second field $\psi$ tracks the solute concentration inside and outside the membrane, governing osmotic exchange.

**Free energy.** The total energy decomposes as

$$\mathcal{F} = \gamma_{\text{surf}} \mathcal{F}_{\text{surf}} + \gamma_{\text{bend}} \mathcal{F}_{\text{bend}} + \mathcal{F}_{\text{osm}} + \gamma_{\text{area}} \mathcal{F}_{\text{area}}$$

where the four terms capture surface tension, Helfrich bending rigidity, osmotic free energy, and a penalty that enforces conservation of enclosed area (2D) or volume (3D). The surface and bending energies use a Willmore-type diffuse-interface approximation via the phase-field parameter $\varepsilon$.

**Hydrodynamics.** The vesicle is immersed in a viscous incompressible fluid governed by the Navier–Stokes equations. The phase-field chemical potentials $\mu$ and $\nu$ enter as body forces, creating two-way coupling between the membrane dynamics and the flow.

**SAV reformulation.** To achieve unconditional energy stability, all nonlinear energy contributions are reformulated using the Scalar Auxiliary Variable (SAV) approach with three auxiliary scalars $R_1, R_2, R_3$ (one per energy term) and a fourth scalar $Q$ from the velocity splitting. This reduces the nonlinear problem to a sequence of linear solves per timestep.

---

## Numerical Methods

| Aspect | Method |
|---|---|
| Spatial discretization | Pseudo-spectral (Fourier), periodic domain |
| FFT backend | FFTW via `rfft`/`irfft` (real-to-complex, ~half storage) |
| Time integration | BDF1 (first step) → BDF2 (all subsequent steps) |
| Nonlinear terms | Second-order explicit extrapolation |
| $\psi$ linear solve (2D) | BiCGSTAB with diagonal preconditioner (`Krylov.jl`) |
| $\psi$ linear solve (3D) | Spectral direct solve (constant-coefficient approximation) |
| Pressure–velocity coupling | Rotational incremental projection method |
| Linear systems | Solved spectrally by pointwise division in Fourier space |
| Energy stability | Guaranteed unconditionally by SAV construction |

The seven-step splitting per timestep is:

1. **Step 1** — Solve the decoupled linear subproblems for $\phi$, $\mu$, $\psi$, $\nu$ (homogeneous + particular parts).
2. **Step 2** — Assemble and solve the $3 \times 3$ linear system for the SAV scalars $R_1^{n+1}, R_2^{n+1}, R_3^{n+1}$.
3. **Step 3** — Reconstruct the full fields $\phi^{n+1}, \mu^{n+1}, \psi^{n+1}, \nu^{n+1}$ by linear combination.
4. **Step 4** — Solve for the two intermediate velocity components $\tilde{\mathbf{u}}_1, \tilde{\mathbf{u}}_2$.
5. **Step 5** — Solve the scalar equation for $Q^{n+1}$.
6. **Step 6** — Combine $Q$-split terms to obtain $\tilde{\mathbf{u}}^{n+1}$.
7. **Step 7** — Pressure correction (Poisson solve + velocity projection) to enforce $\nabla \cdot \mathbf{u} = 0$.

---

## Project Structure

```
Multiple_SAV/
├── src/
│   ├── Types.jl            # Config, FieldState, Operators, Step1Cache, BDFCoeff (2D)
│   ├── SpectralUtils.jl    # FFTW plans, wavenumber arrays, spectral transforms (2D)
│   ├── Utils.jl            # Energy functionals, variational derivatives H1/H2/H3/MG (2D)
│   ├── Init.jl             # Parameter setup and initial condition generation (2D)
│   ├── Solvers.jl          # Seven-step solver (Steps 1–7) (2D)
│   ├── simulation.jl       # Time-loop driver: run_simulation()
│   ├── Types_3d.jl         # Same as Types.jl, extended to 3D
│   ├── SpectralUtils_3d.jl # 3D spectral operators and FFTW plans
│   ├── Utils_3d.jl         # 3D energy and variational derivative functions
│   ├── Init_3d.jl          # 3D parameter setup and initial conditions
│   └── Solvers_3d.jl       # 3D seven-step solver
├── scripts/
│   ├── Main.jl             # Entry point: loads all modules, defines main()
│   └── visualize.jl        # Plotting utilities (heatmaps, energy curves, isosurfaces)
├── results/                # Saved PNG figures (auto-created)
├── MAT/                    # Saved .mat state snapshots (auto-created)
├── Project.toml            # Julia package dependencies
├── Manifest.toml           # Exact locked dependency versions (reproducible builds)
├── install_env.jl          # One-shot environment setup script
└── .env                    # Thread count settings for Julia and FFTW
```

**Key design invariants across all files:**

- `Config` is fully immutable; constructed once before the time loop.
- `FieldState` arrays are always updated in-place (`copyto!`, `.=`); never pointer-aliased.
- `Operators` pre-allocates all working buffers; hot-path functions are allocation-free.
- 2D fields are rank-3 arrays `(Nx, Ny, N)` and 3D fields are rank-4 `(Nx, Ny, Nz, N)`, where `N` is the number of vesicles.
- Velocity is `(Nx, Ny, 2)` in 2D and `(Nx, Ny, Nz, 3)` in 3D.

---

## Requirements

- **Julia** ≥ 1.9 (developed and tested on Julia 1.12)
- A CPU with AVX2 support is recommended for FFTW performance
- For 3D runs at resolution 128³: ≥ 32 GB RAM recommended

**Julia packages** (declared in `Project.toml`):

| Package | Role |
|---|---|
| `FFTW` | Spectral transforms |
| `Krylov` | BiCGSTAB iterative solver |
| `LinearMaps` | Matrix-free operator interface |
| `CairoMakie` | 2D publication-quality figures |
| `GLMakie` | Interactive 3D rendering |
| `Meshing` | Marching Tetrahedra isosurface extraction |
| `GeometryBasics` | Mesh primitives for 3D visualization |
| `MAT` | Save/load `.mat` files for MATLAB interoperability |
| `ProgressMeter` | Progress bar in the time loop |
| `Revise` | Hot-reload source files during development |

---

## Installation

Clone the repository and run the setup script once:

```bash
git clone <repository-url>
cd Multiple_SAV
julia install_env.jl
```

The script will activate the project environment, resolve and install all dependencies declared in `Project.toml`, precompile the package cache, and verify that key packages load correctly. On a fresh machine with a warm package server, this typically takes 5–15 minutes, dominated by the first-time compilation of the Makie backends.

To control parallelism, edit `.env` or set environment variables before launching Julia:

```bash
# Use all available CPU cores for Julia and FFTW
export JULIA_NUM_THREADS=auto
export FFTW_NUM_THREADS=8
julia --threads auto scripts/Main.jl
```

---

## Usage

### Interactive REPL (recommended for development)

```julia
# From the project root:
julia --project=. --threads auto

# Inside the REPL:
include("scripts/Main.jl")   # loads all modules with Revise tracking

# Run with default parameters (dt=1e-6, T=1e-4, state_type=1)
state, E, A, Dt = main()

# Override any parameter
state, E, A, Dt = main(dt=1e-5, T=5e-3, state_type=3)
state, E, A, Dt = main(dt=1e-6, T=1e-3, save_interval=5e-4)
```

After editing any source file in `src/`, Revise automatically reloads it. Call `main()` again without restarting the REPL.

### Script mode

```bash
julia --project=. --threads auto scripts/Main.jl
```

### Initial condition types (`state_type`)

For `N = 1` (single vesicle):

| `state_type` | Shape |
|---|---|
| `1` | Ellipse |
| `2` | Triangular (smooth) |
| `3` | Star-shape (10-fold symmetry) |
| `4` | Random blob |

For `N ≥ 2` (multi-vesicle), use `state_type ≥ 7`.

### Switching between 2D and 3D

In `scripts/Main.jl`, comment/uncomment the corresponding file list passed to `includet`:

```julia
# 2D (default)
for file in (
    joinpath(_SRC, "Types.jl"),
    joinpath(_SRC, "SpectralUtils.jl"),
    ...
)

# 3D — swap to:
for file in (
    joinpath(_SRC, "Types_3d.jl"),
    joinpath(_SRC, "SpectralUtils_3d.jl"),
    ...
)
```

The solver API (`solve_step1` through `solve_step7`) is identical in both cases; dispatch is determined by the array rank of `FieldState` fields.

---

## Configuration

All physical and numerical parameters are set in `set_para_base()` inside `Init.jl` (2D) or `Init_3d.jl` (3D). Key parameters:

| Parameter | Symbol | Description |
|---|---|---|
| `epsilon` | $\varepsilon$ | Interface thickness; smaller = sharper membrane |
| `gamma_bend` | $\gamma_{\text{bend}}$ | Helfrich bending modulus |
| `gamma_area` | $\gamma_{\text{area}}$ | Area/volume constraint penalty weight |
| `gamma_in/out` | $\gamma_{\text{in/out}}$ | Osmotic energy weights inside/outside |
| `psi_in_v` | $\psi_{\text{in}}$ | Target solute concentration inside |
| `psi_out_v` | $\psi_{\text{out}}$ | Target solute concentration outside |
| `M_phi` | $M_\phi$ | Phase-field mobility |
| `M0_psi` | $M_0^\psi$ | Base solute mobility |
| `eta` | $\eta$ | Fluid viscosity |
| `S1`–`S4` | | SAV stabilization coefficients |
| `C1`–`C3` | | SAV energy shift constants (must keep $W_i + C_i > 0$) |
| `goal` | | `:g` for growth, `:s` for shrinkage |

The `goal` symbol selects initial conditions and osmotic concentration profiles consistent with either a growing or shrinking vesicle scenario.

**Grid and time:**

```julia
Nx, Ny = 128, 128      # spatial resolution
Lx, Ly = 1.0, 1.0      # domain size
dt     = 1e-6           # timestep (adaptive warm-up uses finer steps initially)
T      = 1e-3           # total simulation time
```

The time loop uses a two-phase schedule: a fine warm-up phase (`dt = 1e-7` for `t ∈ [0, 1e-5]`) followed by the main phase at the specified `dt`. This avoids instability from the cold start.

---

## Output

Results are written to two locations (both created automatically):

**`results/`** — PNG figures saved at each `save_interval`:
- `phi_t<time>.png` — phase field $\phi$ (summed over vesicles for `N > 1`)
- `phi_final.png` — final state
- `energy_history.png` — modified SAV energy vs. time
- `area_history.png` — relative area/volume error vs. time

**`MAT/case*/`** — MATLAB-compatible `.mat` snapshots containing all field arrays (`phi`, `psi`, `mu`, `nu`, `u`, `p`) and scalar SAV variables (`R1`, `R2`, `R3`, `Q`) at each saved timestep.

The return value of `main()` and `run_simulation()` is `(state, energy_history, area_ratio_history, Dt)`, giving direct access to the final `FieldState` and full time series for post-processing in Julia.

**3D visualization** is handled separately via `plot_iso()` in `scripts/visualize.jl`, which extracts isosurfaces of $\phi$ using the Marching Tetrahedra algorithm and renders them interactively with GLMakie.
