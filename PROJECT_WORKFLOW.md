# Current team workflow

The user adopted `C:\Users\Edwin Himax\Downloads\PathSnap_Equal_Three_Member_Task_Distribution.docx` as the current workflow and identified themselves as **Member 2**. This supersedes the earlier proposed ownership split in PathSnap_Implementation_Plan.md.

- Member 1: raw data, EDA, projection, road graph, spatial index, candidate generation and graph-based transition distances.
- Member 2 (this workspace task): inference features; reference alignment with Member 1; Logistic Regression, distance-weighted KNN, Decision Tree, Random Forest, Histogram Gradient Boosting; trip-grouped CV; Bayesian optimisation; real results; model bundle; corresponding report sections.
- Member 3: HMM emissions/Viterbi; evaluation and route metrics; FastAPI; UI; Folium; two-laptop deployment.
- Shared: reference alignment (1+2), HMM integration (1+3), model integration (2+3), testing, repository review, report and presentation.

Member 2's code lives in `pathsnap_ml/`. A small raw-data geometry adapter enables independent development until Member 1 supplies the production candidate interface. It does not implement graph transitions, HMM, API or UI. Reference alignment is provisional until validated with Member 1 or replaced with organiser-provided labels. Experimental claims must retain that qualification.

The user subsequently requested implementation of the **full project**. All three workstreams are now implemented locally: `pathsnap_ml/` (ML), `pathsnap/` (directed transitions, Viterbi, reconstruction, isolated scoring, maps and reports), `api/` (FastAPI), and `ui/` (offline-capable interface). The role split still identifies review/report ownership; it no longer limits implementation scope. Physical two-laptop rehearsal, official label validation and organiser-only evaluator inputs require the team's actual evaluation setup.
