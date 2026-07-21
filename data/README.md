# TGF Datasets

## Parameters_5K.csv (Primary Dataset)

- **Source**: monthly cooling-tower water-analysis reports from multiple industrial
  cooling towers at **DCM Shriram Alkali** plants in India,
  used with the operators' permission.
- **Date range**: derived from source-report labels, roughly **2012–2025 (with gaps)**.
  Per-row timestamps were not preserved: the `Date` column is populated for only
  1 of 5,614 rows.
- **Rows**: 5,614
- **Columns**: 18
- **Nature**: periodic **lab water-analysis records** (from ~monthly reports), not
  high-frequency sensor telemetry. There is **no temperature or ORP column**.

| Column | Unit | Description |
|--------|------|-------------|
| pH | - | Hydrogen ion concentration |
| Turbidity_NTU | NTU | Water clarity |
| Free_Residual_Chlorine_ppm | ppm | Biocide residual |
| TDS_ppm | ppm | Total Dissolved Solids |
| Total_Hardness_ppm | ppm as CaCO3 | Calcium + Magnesium hardness |
| Calcium_Hardness_ppm | ppm as CaCO3 | Calcium contribution |
| Magnesium_Hardness_ppm | ppm as CaCO3 | Magnesium contribution |
| Chlorides_ppm | ppm | Chloride ion concentration |
| Phosphate_ppm | ppm | Phosphate residual |
| Total_Alkalinity_ppm | ppm as CaCO3 | Bicarbonate + carbonate alkalinity |
| Sulphates_ppm | ppm | Sulphate ion concentration |
| Silica_ppm | ppm | Silica concentration |
| Source_Sheet | - | Originating report sheet (tower + report period) |
| Date | - | Report date (present for only 1 row; see above) |
| Iron_ppm | ppm | Iron (corrosion indicator) |
| Suspended_Solids_ppm | ppm | Suspended solids |
| Conductivity_uS_cm | uS/cm | Electrical conductivity |
| Cycles_of_Concentration | - | Cycles of Concentration |

## Parameters_1.csv

Smaller subset (**1,282 rows**, 19 columns; includes a per-row `Tower` label and
populated `Date`) for quick testing.

## Data Provenance

Consolidated from ~80 monthly cooling-tower water-analysis report sheets (Excel)
from **DCM Shriram Alkali** facilities. Reports span
several towers — including Main Plant CWT, Power Plant CWT, New Plant (X-01 CT),
850 CT TPD, ECH CT, and OLD/NEW CT units (the historical archive spans more tower
units and sites than the current live deployment, which covers four towers at one
plant). The `Source_Sheet` column preserves the
originating report for every row.

## File integrity (SHA-256)

- `Parameters_5K.csv` → `52813db7c3e87c2cc0eb585102f5196f62940985298747b61329df20095846f6`
- `Parameters_1.csv` → `25da02e17b3c71304b79f4484584d064772cfa6b87202315227f5fab7d6637be`
- `cooling_tower_dataset_cleaned.csv` → `13704169c616485f6f7026f2914c0ca76ab2e541737c897d0927feb4681c590f`
