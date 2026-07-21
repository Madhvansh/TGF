# Evidence index

This file maps each load-bearing claim in the project [README](README.md) to a
concrete, checkable artifact in this repository (or a contact who will confirm it).
It is a navigation aid for evaluators; the claims themselves live in the README.

**Scope note.** The deployment-confirmation letters cover **monitoring and analysis**
of tower operations at DCM Shriram Alkali only. Claims about dosing recommendations,
the closed-loop controller, and the second plant are the maintainer's own and are
mapped to their own artifacts below, not to the letters.

| Claim (as stated in the README) | Where stated | Checkable artifact | How to check |
|---|---|---|---|
| TGF is self-hosted on-site at DCM Shriram Alkali (four cooling towers) since June 2026, used for monitoring and analysis of tower operations | Production status & provenance; Deployment verification | Two deployment-confirmation letters — a supplier letter and a company letter — documenting a single deployment from two vantage points | Open [the supplier letter](docs/deployment/2026-07-21-hydrotech-services-supplier-letter.pdf) and [the company letter](docs/deployment/2026-07-21-dcm-shriram-alkali-company-letter.pdf); or contact the maintainer (choksimac167005@gmail.com), who will connect evaluators with the plant's utility head and plant manager |
| Advisory production use since 2025 across eight cooling towers at two Indian plants (DCM Shriram Alkali and Atul Ltd) | Lead paragraph; Production use | Maintainer attestation for the 2025 remote/advisory phase and for the Atul Ltd towers (that plant does not issue public confirmations) | Contact the maintainer; the DCM Shriram Alkali portion is additionally documented by the letters above |
| Closed-loop controller validated in backtest only — not wired to dosing hardware; a human authorizes every dose | Production status; Production use (Boundary); Backtest results | The controller is exercised by replaying historical records in simulation, not by live actuation; the boundary is stated in the docs and enforced in code | Read the [Backtest results](README.md#backtest-results-closed-loop-controller-simulated-on-historical-data) section and the Safety Layer description in the [README](README.md); inspect the `tgf_dosing/` package |
| The backtest uses 5,614 historical water-analysis records from DCM Shriram Alkali | Production status; Backtest results | The released dataset with its documented provenance and integrity hashes | See [data/README.md](data/README.md) (row count, provenance, SHA-256 checksums) and `data/Parameters_5K.csv` |
| TGF depends on cooling-tower-chem for its water-chemistry indices | Production status; Physics Engine | Declared package dependency | See `dependencies` in [pyproject.toml](pyproject.toml) and the [cooling-tower-chem](https://github.com/Madhvansh/cooling-tower-chem) repository |
| Tests and lint pass across Python 3.10–3.12 | Python-version badge; Quick Start | CI workflow and its run history | See [.github/workflows/ci.yml](.github/workflows/ci.yml) and the [Actions run history](https://github.com/Madhvansh/TGF/actions) |
| README claims describe an advisory system (not autonomous control), a framing applied deliberately over time | Throughout | Git history shows the advisory framing being tightened | See commits `e7f164e` (provenance and integrity metadata) and `95f9872` (runtime banner aligned with advisory scope) |

## Notes

- **One deployment, two vantage points.** The two letters document a single
  deployment: one from the supplier that provided TGF to the plant, one from the
  company operating it. The company letter is unsigned — titles only, no individual
  names — by the company's stated policy; the supplier letter is signed by its
  proprietor.
- **What the letters do not cover.** They speak to monitoring and analysis at DCM
  Shriram Alkali. They do not attest the dosing-recommendation workflow, the
  closed-loop controller, or the Atul Ltd towers — those rest on the maintainer's
  attestation and on the in-repo backtest and code.
