# BSPC Submission Checklist

**Production pass completed** (reference formatting, figure legibility, bibliography rendering, empty-section fix — see entries marked 🔧 below). Science untouched throughout this pass.


Compiled from Elsevier/BSPC's standard author requirements. Internal tracking document, not part of the submission package itself.

## Manuscript files

- [x] `paper.tex` — main manuscript, compiles cleanly (elsarticle, `[review]` mode for review copy)
- [x] `sections/introduction.tex`, `methods.tex`, `results.tex`, `discussion.tex`, `conclusion.tex` — all locked
- [x] 🔧 **Fixed: empty section bug.** `sections/acknowledgements.tex` previously printed a bare "Acknowledgements" heading with nothing beneath it (all content was LaTeX comments), immediately followed by "References" — confirmed via text extraction, a visibly broken-looking section in the compiled PDF. Now prints the two real, already-approved declarations with actual content (Competing Interest: none; Funding: none); the empty heading itself was removed rather than left printing nothing. CRediT and full data-availability statement remain in `declarations.tex`, pending BSPC template confirmation on whether they must also appear in-manuscript.
- [x] `refs.bib` — 5 core citations verified confident; 3 dataset citations flagged for author list/DOI verification
- [x] 🔧 **Fixed: bibliography rendering bug.** The `note = {Verify full author list before submission}` field was printing directly into the formatted, compiled bibliography (confirmed via text extraction: references [4] and [6] literally showed this sentence in the PDF). Moved to a non-printing `%` comment; the underlying verification task itself (real author lists for Codella et al., Pacheco et al.) is unchanged and still required.
- [x] 🔧 **Fixed: 3 capitalization defects.** `elsarticle-num`'s sentence-case title formatting had downcased unprotected proper nouns: "international skin imaging collaboration (isic)" → "International Skin Imaging Collaboration (ISIC)"; "ham10000" → "HAM10000"; "pad-ufes-20" → "PAD-UFES-20". Fixed by wrapping each in `{}` in `refs.bib`; recompiled and confirmed correct in the printed PDF.
- [ ] Missing volume/page identifiers for [1] Lee et al., [2] Sun et al., [3] Ganin & Lempitsky, [7] Hewitt & Liang (PMLR/proceedings page ranges) — not fabricated; verify each against its actual publication.
- [x] `titlepage.tex` — title/abstract/keywords locked; author fields are `<<placeholders>>`
- [ ] Switch `\documentclass[review]{elsarticle}` → `\documentclass{elsarticle}` (remove line numbers/double-spacing) for camera-ready, once accepted — **not before**, reviewers expect line numbers

## Supplementary

- [x] `supplementary.tex` — standalone, compiles cleanly, S1–S7, cross-references main text via `xr`/`\externaldocument`
- [ ] Confirm BSPC accepts a single combined supplementary PDF vs.\ requiring separate figure/table supplementary files (check current author guide)

## Required declarations (BSPC/Elsevier standard set)

- [x] `declarations.tex` — Conflict of Interest, Funding, Ethics, Data Availability, AI Usage Statement, Author Contributions (CRediT) — content real per stated assumptions, author-specific fields placeholdered
- [ ] Confirm whether BSPC wants declarations as a separate file at submission or pasted into the manuscript's end matter (varies by journal even within Elsevier) — check the current submission portal instructions
- [ ] AI Usage Statement: confirm current Elsevier policy wording hasn't changed since this was drafted (policy has been updated more than once) — verify against the live author guidelines before submission

## Title page requirements

- [x] Full title
- [ ] Author names, affiliations, ORCID (placeholders in `titlepage.tex` — fill in)
- [ ] Corresponding author designated with email
- [x] Abstract (locked, ~230 words, one paragraph, no citations)
- [x] Keywords (6, no more than journal max — verify BSPC's exact keyword limit, typically 6)

## Highlights

- [x] `highlights.tex` — 5 bullets, each ≤85 characters (verified programmatically)
- [ ] Confirm BSPC's current highlights count requirement (Elsevier journals vary 3–5)

## Cover letter

- [x] `cover_letter.tex` — 385 words, states novelty/fit/originality/no-conflicts, author fields placeholdered
- [ ] Insert real corresponding-author name/affiliation/address/email
- [ ] Confirm whether BSPC wants the cover letter uploaded as a separate file or pasted into the submission portal's textbox (varies)

## Figures

- [x] 6 main-text figures built, verified visually, embedded with captions and labels
- [x] 6 supplementary figures (S1–S6 content, 5 actual figure files) built and embedded
- [x] 🔧 **Font-size fix applied and verified.** All 5 raster figures were computed to render at ~4.5–7pt effective size after LaTeX's `\includegraphics` scaling to the real 345pt/4.79in text column (this was measured, not assumed). All five regenerated with roughly doubled matplotlib font sizes; recompiled and visually confirmed legible at true print size against body text and table fonts.
- [x] 🔧 Figure 5 (scorer comparison)'s DPI raised from 200→300 to match the other five figures.
- [x] Check each figure against BSPC's resolution/format requirements — now 300 DPI PNG throughout; confirm acceptable, or convert to vector/EPS/TIFF if BSPC's current template requires it
- [ ] Verify color figures are readable in grayscale if BSPC charges for color print (check current color policy — online-only color is usually free)
- [ ] **Graphical abstract** (if BSPC requires one for this article type): recommend reusing Figure 6 (the dumbbell plot) rather than commissioning a new image — it already is the paper's single-glance summary (distance scorers at chance, probes well above, per rung), which is exactly what a graphical abstract should be. Elsevier typically wants it as a standalone image file (not embedded in the PDF); crop `figures/figure_dumbbell.png` to just the plot if so.

## Tables

- [x] 4 main-text tables, 3 supplementary tables, all from verified source data

## Formatting pass (not yet done — final step before submission)

- [ ] Fix remaining overfull/underfull hbox warnings (minor, ~10 in main text, 1 in supplementary)
- [ ] Resolve hyperref "Token not allowed in a PDF string (Unicode)" warnings (cosmetic, affects PDF bookmark text only)
- [ ] Terminology consistency pass: "distance-based" (not "distance based"/"distance–based"), "$k$-NN" (not "kNN"/"KNN"), "PAD-UFES" (not "PAD UFES"), "ISIC-test"/"ISIC-train" hyphenation, tense consistency, capitalization of proper nouns (Mahalanobis, Fisher, etc.)
- [ ] Check reference numbering style matches BSPC's exact required format (currently `elsarticle-num`, numeric — verify this is BSPC's preferred style, not author-year)
- [ ] Spell out all acronyms at first use (AUROC, OOD, GRL, SVM, etc.) — verify each main-text section does this independently or points to a first-use location

## Author-specific information still required (see also the pre-submission audit)

- [ ] Real author names, affiliations, emails, ORCID iDs
- [ ] CRediT role assignments
- [ ] Confirm no conflicts of interest, or state them
- [ ] Confirm no funding, or state funder/grant number
- [ ] Decide whether to make code/data (analysis scripts, results CSVs) publicly available and provide a URL

## Journal-specific checks (verify against BSPC's live author guide before submitting — not done here)

- [ ] Manuscript type (Original Research Article vs.\ other BSPC article types) and any type-specific requirements
- [ ] Graphical abstract — check if BSPC currently requires one for this article type
- [ ] Word count limit, if any, for this article type
- [ ] Suggested reviewers / opposed reviewers (optional at most journals, check if BSPC's portal requests this)
