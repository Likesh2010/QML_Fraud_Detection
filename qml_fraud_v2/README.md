# QML Credit-Card Fraud Detection - v2 evaluation harness

Replaces the slow Qiskit kernel loop and the single-run, test-set-tuned protocol with a
fast exact kernel and a leakage-safe, statistically reported evaluation.

## Run it (PowerShell / bash)
```
pip install -r requirements.txt
python tests/test_qkernel.py                      # 1. verify the fast kernel (gate-level check)
python run_experiments.py --data creditcard.csv --seeds 10 --n-train 500 1000 2000 --pca-dims 4 6 8 --out results
python report.py --results results --k 4 --n 2000 # 2. tables, stats, figures, draft abstract
```
Smoke test without the dataset: `python run_experiments.py --synthetic --quick --seeds 2 --n-train 300 --out smoke`
(synthetic numbers mean nothing - it only proves the pipeline runs). Runs are resumable (finished seeds are skipped).
Approximate cost per seed at n=2000 on one CPU core: ~5 min for the full 120-config quantum grid.

## What it does
| Item | Implementation |
|---|---|
| Fast exact quantum kernel | `qkernel.py` - ZZFeatureMap statevectors + one matrix product; 2000x2000 kernel in ~0.1 s; tested against a gate-by-gate simulator (and Qiskit if installed) |
| Protocol | `data.py` - chronological train 60% / validation 20% / test 20%; scaler+PCA fit on train normals only; angle scaling clipped; duplicate-leakage check |
| Selection | `run_experiments.py` - every model's hyper-parameters chosen by validation PR-AUC; thresholds fixed on validation; test period evaluated once at natural prevalence |
| Equal tuning | quantum: 3 reps x 2 entanglements x 4 bandwidths x 5 nu = 120 configs; RBF: 24 gamma x 5 nu = 120 |
| Baselines | tuned RBF-OCSVM, Laplacian-OCSVM, Isolation Forest (4-d and 29-d), LOF, PCA reconstruction, autoencoder, supervised HGB upper bound |
| Seeds + statistics | 10 training-sample seeds; mean +/- std; paired t-CI over seeds; paired test-set bootstrap; McNemar on frauds |
| Fraud metrics | recall @ 0.1/1/5% FPR, precision and alerts per 10k at natural prevalence, cost-optimal threshold, native-nu boundary (what the old report used) |
| Ablations | training size (`--n-train`), PCA dim = qubits (`--pca-dims`), reps/entanglement/bandwidth/nu (selected-config table), kernel concentration diagnostic |
| Report | `report.py` -> `RESULTS.md`, 3 figures, `abstract_conclusion_draft.md` whose verdict wording is generated from the statistics |

## Things to state honestly in the report
- Hyper-parameter selection on validation uses fraud labels (validation PR-AUC). Call it *semi-supervised model selection*; the training itself is one-class.
- A 4-8 qubit noiseless fidelity kernel is classically simulable; "quantum" here means quantum-inspired feature maps simulated exactly. No hardware or noise is involved.
- Test-set fraud count is small (see `data_info.json`); read the CIs, not just the means.
- The cost ratio (`--cost-ratio`, default 50:1) is an assumption, not a measured bank cost.

## Updating the written report
1. Replace the abstract/conclusion with `abstract_conclusion_draft.md` after running (edit style, not claims).
2. Replace the old result tables with `RESULTS.md` tables; keep the old 500/700/1000 runs only as a "preliminary study" appendix.
3. Delete the "What to show sir" section.
4. Insert the three figures and the Streamlit screenshots.
