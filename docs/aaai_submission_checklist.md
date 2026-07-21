# AAAI Submission Checklist

The repository contains one canonical AAAI-27 source package at `docs/overleaf_submission.zip`. Upload it to a fresh Overleaf project and select `aaai_paper.tex` as the main document.

Upload `output/pdf/supplementary.pdf` as the Technical Supplement,
`output/pdf/ReproducibilityChecklist.pdf` as the Reproducibility Checklist, and
`docs/code_and_data_supplement.zip` as the Code and Data Supplement. Leave the
Media Supplement empty.

## Must Fix Before Submission

- Use `docs/aaai_paper.tex` with the included official `aaai2027.sty` / `aaai2027.bst` files.
- Keep the anonymous author setting in the submission source; do not expose local author names or acknowledgments in the review version.
- BibTeX flow is active through `docs/references.bib`; verify the final bibliography in Overleaf.
- Re-check the final page length after the latest Table 1 and figure updates.
- Keep the final BiCheck diagnostic values synchronized across metadata, paper,
  figure, and reproducer: separation `+0.0826`, AUROC `0.589`.
- Package only current sources and figures. Do not include deprecated draft/plan files such as `docs/paper_draft.md` or old experiment notes in review artifacts.

## Scientific Risks To Address If Time Allows

- A fair MoDeGraph-style graph-prompt baseline has been added under the current fixed extraction/evaluation pipeline; still avoid claiming coverage of all graph-reasoning systems.
- LongBench tests are exploratory and not adjusted across datasets and metrics; avoid family-wise significance claims.
- Add a stronger CS-correctness analysis if claiming calibration; current results support a weak diagnostic/ranking signal, not probability calibration.
- Keep 2Wiki and evidence-grounded checking results framed as boundary/diagnostic analyses, not main wins.
- Build the final source package from a whitelist rather than copying the entire `docs/` directory.

## Suggested Source-Package Whitelist

- `docs/aaai_paper.tex`
- `docs/supplementary.tex`
- `docs/ReproducibilityChecklist.tex`
- `docs/references.bib`
- `docs/aaai2027.sty` and `docs/aaai2027.bst`
- `docs/figures/image1.pdf` through `docs/figures/image7.pdf`

## Current Safe Framing

- Structural graph validity is a poor reasoning-quality signal.
- GERS-DAG produces the answer; BiCheck is evaluated separately as a post-hoc diagnostic.
- BiCheck improves CS association with correctness from near-random to weakly useful.
- Under fair full context, HotpotQA favors CoT-SC in F1. The old positive HotpotQA result is a truncated-context artifact.
- BiCheck does not alter answers in the evaluated single-path configuration.
- Oracle retrieval and sub-answer runs are independent interventions conditioned on gold decomposition, not additive module shares.
- Self-consistency is not correctness; high-CS wrong answers remain common.
