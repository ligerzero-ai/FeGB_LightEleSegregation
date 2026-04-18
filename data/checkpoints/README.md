# Cached analysis checkpoints

The pickled checkpoint DataFrames that are loaded by the figure-generation scripts will be added here when the manuscript enters peer review.

| Expected file | Loaded by | Contents |
|---|---|---|
| `04_df_KP_voronoi.pkl.gz` | `MainFigures/generate_main_figures.py` | KP-mesh segregation calculations with Voronoi descriptors. |
| `04_df_KP_voronoi_with_nn.pkl.gz` | (analysis only) | As above plus 1st–6th nearest-neighbour distances per site. |
| `07_df_KP_filtered.pkl.gz` | `MainFigures/generate_main_figures.py` (Fig.\u00a09) | KP segregation calcs after SOAP-based duplicate removal. |
| `09_df_main_final.pkl.gz` | `MainFigures/generate_main_figures.py` (Figs.\u00a07–8) | Per-site cohesion data (ANSBO, Wsep, η). KS-mesh data. |
| `08_df_compare_pairwise.pkl.gz` | `SI/generate_SI_kpoint_*.py` | Interstitial KP↔KS pairwise comparison. |
| `08_df_sub_compare_pairwise.pkl.gz` | `SI/generate_SI_kpoint_*.py` | Substitutional KP↔KS pairwise comparison. |
| `KP_Vacancy_Formation_Energy.pkl.gz` | `SI/generate_SI_vacancy_tables.py` | Per-site relaxed vacancy formation energies. |

## Provenance

These checkpoints are intermediate outputs of the full analysis pipeline. They will be released, along with the raw DFT DataFrames in `data/raw/`, when the manuscript enters peer review.
