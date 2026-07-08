# AAAI Submission Checklist

This repository currently contains an AAAI-oriented draft, not a ready-to-upload AAAI submission package.

## Must Fix Before Submission

- Replace the current `article`/`ctexart` source with the official AAAI template (`aaai*.sty`, bibliography style, and required copyright/anonymous settings).
- Keep the anonymous author setting in the submission source; do not expose local author names or acknowledgments in the review version.
- Convert hand-written `thebibliography` entries to the official BibTeX flow once the AAAI template is added.
- Re-check page length after switching templates; the current article layout is not a valid page-count proxy.
- Package only current sources and figures. Do not include deprecated draft/plan files such as `docs/paper_draft.md` or old experiment notes in review artifacts.

## Scientific Risks To Address If Time Allows

- A fair MoDeGraph-style graph-prompt baseline has been added under the current fixed extraction/evaluation pipeline; still avoid claiming coverage of all graph-reasoning systems.
- Increase HotpotQA sample size or add another model if API budget allows; current end-task gains are modest and bootstrap intervals cross zero.
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
- HotpotQA gains are modest and statistically mixed.
- Self-consistency is not correctness; high-CS wrong answers remain common.
