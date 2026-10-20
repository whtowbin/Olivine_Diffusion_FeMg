# FORSTER

**F**itting **O**livine **R**esidence & **S**torage **T**imescales from **E**lemental
**R**elaxation.

Olivine that grows in a changing magma records the change as zoning: a rim with a
different forsterite content (Fo) from its core. While the crystal sits in hot magma,
Fe and Mg exchange across that boundary and the sharp step relaxes into a smooth
profile. The wider the profile, the longer the crystal stayed hot. FORSTER estimates
that time by fitting a model of Fe-Mg diffusion to Fo profiles measured across zoned
olivine.

You give FORSTER the measured profile, the temperature, pressure and oxygen fugacity
during diffusion, the direction of the profile in the crystal, and a *starting
profile*. The starting profile is a step function, your best estimate of the zoning
before diffusion began. FORSTER runs the diffusion model forward in time and reports
the time at which the model best matches the data. It also gives 1σ and 2σ ranges,
which should contain the true time 68% and 95% of the time. It can fit the starting
profile as well, and carry the uncertainty in temperature, pressure, oxygen fugacity
and the diffusion law (the equation that gives the Fe-Mg diffusivity) into the time.

<figure markdown="span">
  ![The desktop app with a fitted profile](images/app-monte-carlo.png)
  <figcaption>The desktop app. The plot shows the measured points (circles), the
  starting profile (orange, which you can drag), the best fit with its 1σ and 2σ bands,
  and the residuals. On the right is the Uncertainties tab with a Monte Carlo over the
  conditions.</figcaption>
</figure>

<figure markdown="span">
  ![A fitted profile written by forster fit](images/quickstart-fit.png){ width="640" }
  <figcaption>The figure that <code>forster fit</code> writes for the worked example
  included with FORSTER, a published melt-vein olivine profile from Towbin and Plank
  (2026). The best fit is 5.1 days, with a 1σ range of 4.2-6.2 days.</figcaption>
</figure>

## Three ways to use it

- **The desktop app** (`forster gui`): load profiles, draw the starting profile with
  the mouse, fit, and look at the results. No programming. [The desktop app](app.md).
- **The command line** (`forster fit`): two spreadsheets in, a folder of figures and
  a summary table out. Suited to fitting many profiles at once. [Quick start](quickstart.md).
- **Python** (`import forster`): everything above as functions, plus features not yet
  in the tables or the app (joint fits of several profiles from one crystal, and
  profiles of Ni, Mn and Ca). [Python API](api.md).

All three read and write the same tables, so you can start in the app and rerun the
same fits from the command line, or the other way round.

## Where to go next

| If you want to... | Read |
|---|---|
| install FORSTER | [Installation](installation.md) |
| fit the worked example in five minutes | [Quick start](quickstart.md) |
| get your microprobe spreadsheet into FORSTER, and plan your analyses | [Preparing data](data.md) |
| choose the starting profile, conditions, orientation and diffusion law | [Setting up a run](setup.md) |
| understand what the time and its range mean | [How the fit works](how-it-works.md) |
| include the starting profile, T, P, fO2 and the diffusion law in the uncertainty | [Uncertainty beyond the fit](uncertainty.md) |
| read `summary.csv` and act on FORSTER's warnings | [Reading the results](results.md) |
| look up a column | [Column reference](columns.md) |
| look up a term (diffusion length, prior, ESS, ...) | [Glossary](glossary.md) |

![How FORSTER turns a profile into a time](images/schematic-workflow.png)

*From inputs to a time distribution. **a** The inputs: a measured Fo profile, the
conditions, the crystal orientation and a step-shaped starting profile. **b** The
model runs forward in time, and the step widens with the diffusion length √(D t).
**c** Each time step is compared with the data. The best fit is the time of least
misfit, and the range spans the times where the misfit is within about 1 of that
minimum. **d** Optionally, uncertain step positions and segment Fo are fitted along
with the time. **e** A Monte Carlo draws T, P, fO2 and the parameters of the
diffusion law. **f** The resulting distributions of time. The curves are
illustrative.*

## Status

FORSTER is in development (version 0.1.0.dev0) and not yet on PyPI. Its results
differ from those of earlier versions of this code. The [changelog](changelog.md)
says what changed and by how much.

## How to cite

> Towbin, W. H., and Plank, T. (2026). *Journal of Petrology* 67(9), egag077.

Please also cite the source of the diffusion law you used (see
[Setting up a run](setup.md#the-diffusion-law)). Some of the references that FORSTER
cites are still being checked before the first release. The list is in
[citations-to-confirm.md](https://github.com/whtowbin/Olivine_Diffusion_FeMg/blob/master/docs/references/citations-to-confirm.md)
on GitHub.
