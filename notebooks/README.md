# Notebooks — working through `resources/topics_guide.md`

One notebook per Part of the roadmap, written in the same style as `NB_0_foundations.ipynb`
and `NB_2_statistical_thinking.ipynb`: section headings are **questions**, prose interprets
the numbers the code actually printed, and **Try it** boxes ask you to write an answer in
your own words before moving on.

Every notebook is committed **with its outputs already executed**.

## Structure of each notebook

| Opening | |
|---|---|
| `What will you be able to do by the end?` | concrete, checkable objectives |
| `What should already be familiar?` | which earlier notebook and section you need |
| `What route will we take?` | a Part / Question / Main output table |

Then numbered sections, each headed by a question. Throughout:

- **`### Try it N.`** — a question about what just happened, followed by a
  **`**Your answer**`** cell for you to fill in. **56 of these across the eight notebooks.**
  Several ask you to *predict before running*; a few ask you to write the sentence you would
  actually publish.

| Closing | |
|---|---|
| `What can we conclude?` | a table of what was established and where it is used next |
| `What should you submit?` | a checklist, plus optional "Going further" extensions |

## The notebooks

| Notebook | Roadmap | Question it answers | Try it |
|---|---|---|---:|
| `00_foundations.ipynb` | Part 0 | What mathematics does the rest of this stand on? | 9 |
| `01_svm_kernel_gateway.ipynb` | Part 1 | Where does a kernel come from in the first place? | 7 |
| `02_gaussian_processes.ipynb` | Part 2 | How does a kernel become a posterior over functions? | 7 |
| `03_kernel_design.ipynb` | Part 3 | How do we build kernels for the data we actually have? | 7 |
| `04_hierarchical_kernels.ipynb` | Part 4 | What if the kernel's *form* is the thing we don't know? | 6 |
| `05_high_dimensional.ipynb` | Part 5 | Which of 200 exposures is responsible? | 7 |
| `06_causal_temporal.ipynb` | Part 6 | *When* did the exposure matter — and is any of it causal? | 7 |
| `07_computation_scalability.ipynb` | Part 7 | How do we make all of this run at cohort scale? | 6 |

`nbkit.py` is the shared toolkit: kernels (RBF, ARD, Matérn, periodic, polynomial), PSD
diagnostics, stable Cholesky with jitter, kernel centring, plot styling, and the
`make_exposome` synthetic data generator.

Part 8 (case studies) is not implemented.

## Running

```bash
uv sync
uv run jupyter lab notebooks/
```

Notebooks import `nbkit` from this directory, so launch Jupyter from inside `notebooks/`.

## A note on what these contain

Where a method underperforms, the notebook says so and explains why rather than tuning until
the demo looks good. Several **Try it** boxes are built on exactly those moments:

- **NB-2 §2.3** — ARD makes a *confident, arbitrary* choice between two exchangeable exposure
  proxies; the winner flips with the random seed.
- **NB-4 §4.4** — the DP-GP's clustering looks mediocre overall (ARI 0.37) but is exact
  wherever the two latent subgroups are genuinely distinguishable.
- **NB-5 §5.3** — a single spike-and-slab chain reports a definite answer about which of two
  duplicated exposures matters; six chains disagree. Only pooling reveals it.
- **NB-6 §6.1** — distributed lag models need large $n$: below ~250 subjects the marginal
  likelihood is nearly flat in the lag lengthscale and no critical window is detectable.
- **NB-6 §6.4** — a flexible kernel improves confounder *adjustment* and does nothing for
  *identification*.
- **NB-7 §7.2** — mean-field VI underestimates posterior sd by ~7× once predictors correlate.
- **NB-7 §7.5** — the elpd difference between two kernels is reported with its standard
  error, and at $n = 80$ it is not decisive.

## Related work in this repo

- `Bayesian_projects/03_bayes_kernel/rff_bkmr.py` — random Fourier features for BKMR
  (NB-7 §7.1, NB-6 §6.3)
- `resources/pdfs/treed based lag.pdf` — Mork & Wilson, the tree-based alternative to the
  smooth distributed lag models in NB-6 §6.1–6.2
- `resources/pdfs/` — the rest of the source papers; see `topics_guide.md` Appendix A4
