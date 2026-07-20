# AAAI 2027 Submission Handoff

Updated: 2026-07-13

## Canonical Files

- Main paper: `docs/aaai_paper.tex`
- Supplement: `docs/supplementary.tex`
- References: `docs/references.bib`
- Reproducibility checklist: `docs/ReproducibilityChecklist.tex`
- Figures: `docs/figures/image1.pdf` through `image7.pdf`
- Overleaf package: `docs/overleaf_submission.zip`

The repository intentionally keeps one submission package. Historical paper drafts,
versioned ZIP files, alternate PNG/EPS figures, and stale compiled previews were
removed to prevent accidental submission of superseded claims.

## Scientific Framing

- `GERS-DAG` is the answer-producing graph-decomposition pipeline.
- `BiCheck` is a post-hoc forward/re-derived sub-answer agreement diagnostic; graph edges are not reversed.
- LongBench end-task differences are attributed to GERS-DAG, not BiCheck.
- Oracle retrieval and sub-answer conditions are independent interventions
  conditioned on gold decomposition, not additive module shares.
- BiCheck improves correct/wrong discrimination to AUROC 0.589, but fluent errors
  can remain self-consistent; it is not a demonstrated correctness verifier.
- Legacy experiment configuration names such as `gers_cv2_fullctx` remain in code
  and experiment records for reproducibility only.

## Current Submission Status

- The paper uses the official anonymous AAAI 2027 template.
- The latest reviewed compile is six pages including references.
- Table 1 now has eight columns; the redundant `Result` column was removed to
  eliminate right-margin overflow.
- Figures 3, 4, and 6 were adjusted for label spacing and grayscale readability.
- The canonical ZIP is built from a fixed whitelist and contains only current
  sources, style files, bibliography, checklist, supplement, and seven PDF figures.

## Remaining External Checks

1. Upload `docs/overleaf_submission.zip` to a fresh Overleaf project.
2. Select `aaai_paper.tex` as the main document and compile with pdfLaTeX/BibTeX.
3. Confirm Table 1 remains inside the page boundary and the total page count is
   within the applicable AAAI limit.
4. Complete the conference submission form and final anonymity check.

## Verification Commands

```powershell
python -m py_compile experiments/gen_figures.py docs/figures/generate_aaai_figures.py
tar -tf docs/overleaf_submission.zip
python -m pytest tests/test_graph.py -v
```
