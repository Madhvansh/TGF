# Changelog

All notable changes to TGF are recorded here. The format is loosely based on
[Keep a Changelog](https://keepachangelog.com/). Tagged releases begin at v0.1.0
(2026-07-21). Earlier entries are grouped by date, derived from the Git commit
history (dates are commit dates).

## v0.1.0 — 2026-07-21

### Added
- `Deployment verification` README section and a `docs/deployment/` directory
  containing two deployment-confirmation letters for the DCM Shriram Alkali
  deployment.
- `EVIDENCE.md` claim-to-artifact index, linked from the README.
- Community-health files: `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`, and
  this changelog; a testing-policy note in `CONTRIBUTING.md`.

### Changed
- Runtime banner wording aligned with the advisory scope.
- README production-status section expanded to document the 2025 remote-advisory to
  June 2026 self-hosted timeline.

## 2026-07-20

### Changed
- Package description, dashboard and module docstrings, and in-code descriptions
  aligned with the advisory framing.

## 2026-07-19

### Added
- Documented production use as an advisory system across eight cooling towers at two
  plants.
- Data-provenance notes and file-integrity (SHA-256) metadata for the datasets.

### Changed
- Adopted the standalone `cooling-tower-chem` library as the single source of truth
  for water-chemistry indices.
- Reframed README claims around backtest results (not autonomous deployment), with
  accurate data provenance.
- Canonicalized the operator name to "DCM Shriram Alkali"; corrected Chronos-T5
  model labels in logs and docs.

### Removed
- Stale profile draft.

## 2026-06-09

### Fixed
- CI lint failures (ruff): removed unused imports and variables and empty f-strings
  (#1).

## 2026-04-07

### Changed
- Repository reorganization.

## 2026-03-29

### Added
- Initial TGF dosing implementation.

## 2026-03-23

### Added
- Initial commit: repository scaffold (excluding large model files).
