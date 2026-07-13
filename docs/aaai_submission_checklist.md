# AAAI Submission Checklist

The repository contains an AAAI-27 source package at `docs/overleaf_submission.zip`. Final Overleaf compilation and page-count verification are still required.

## Must Fix Before Submission

- Use `docs/aaai_paper.tex` with the included official `aaai2027.sty` / `aaai2027.bst` files.
- Keep the anonymous author setting in the submission source; do not expose local author names or acknowledgments in the review version.
- BibTeX flow is active through `docs/references.bib`; verify the final bibliography in Overleaf.
- Re-check page length after switching templates; the current article layout is not a valid page-count proxy.
- Package only current sources and figures. Do not include deprecated draft/plan files such as `docs/paper_draft.md` or old experiment notes in review artifacts.

## Scientific Risks To Address If Time Allows

- A fair MoDeGraph-style graph-prompt baseline has been added under the current fixed extraction/evaluation pipeline; still avoid claiming coverage of all graph-reasoning systems.
- LongBench improvements are nominal and not corrected for comparisons across datasets and metrics; avoid family-wise significance claims without a multiplicity analysis.
- Add a stronger CS-correctness analysis if claiming calibration; current results support a weak diagnostic/ranking signal, not probability calibration.
- Keep 2Wiki and evidence-grounded checking results framed as boundary/diagnostic analyses, not main wins.
- Treat `docs/paper.docx` as an internal preview only; it is not an AAAI submission source.
- Build the final source package from a whitelist rather than copying the entire `docs/` directory.

## Suggested Source-Package Whitelist

- Official AAAI main `.tex` source after template migration.
- `docs/references.bib` after final bibliography cleanup.
- `docs/figures/image1.pdf` through `docs/figures/image5.pdf` or the template-required vector equivalents.
- Any official AAAI style files required by the template.

## Current Safe Framing

- Structural graph validity is a poor reasoning-quality signal.
- Bidirectional sub-answer cross-checking improves CS association with correctness from near-random to weakly useful.
- Under fair full context, HotpotQA favors CoT-SC in F1. The old positive HotpotQA result is a truncated-context artifact.
- Bidirectional cross-checking improves a post-hoc diagnostic score but does not alter answers in the evaluated single-path configuration.
- Oracle retrieval and sub-answer runs are independent interventions conditioned on gold decomposition, not additive module shares.
- Self-consistency is not correctness; high-CS wrong answers remain common.
