# JOSS paper

`paper.md` and `paper.bib` are the JOSS submission, in the layout JOSS expects
(a `paper/` directory, with the bibliography named in the frontmatter).

## ⚠️ The bibliography exists in more than one place

`paper.bib` here is the same bibliography used by the companion AJ/PSJ paper,
where it is called `layup_paper.bib`. They must be kept in step — a correction
applied to one and not the other will show up as an inconsistency between the two
published papers.

Known copies as of 2026-08-24:

| copy | role |
|---|---|
| `paper/paper.bib` (here) | the JOSS submission |
| `layup_paper.bib` in the AJ paper repository | the AJ/PSJ submission |

A recent example of why this matters: `bernstein2000` — the central method
citation — carried a paraphrased title and the wrong given name for the second
author ("Bhrigu" rather than "Bharat"). The DOI always resolved, so every
automated check passed and nothing looked wrong. It was corrected in both copies
on 2026-08-24, verified against Crossref.

## Author names — resolved 2026-08-24

The JOSS paper carried "Thomas Ruch" and "Carrie Holt" while the AJ paper had
"Thomas R. Ruch" and "Carrie E. Holt". Both now use the initialled forms, so the
two papers agree exactly: same 18 authors, same order.

The evidence, from Crossref:

- **Carrie E. Holt** — clear. Six of nine recent works use the initialled form,
  including **all four that carry her ORCID** (RNAAS 2026, PSJ 2026, ApJL 2026,
  ApJL 2025, PSJ 2025). The variants are all ORCID-less journal-style differences
  ("C. Holt" in Icarus, "Carrie E Holt" in MNRAS).
- **Thomas R. Ruch** — thinner. Only two publications, one each way: PSJ 2026
  (*heliostack*) as "Thomas R. Ruch", ApJL 2025 as "Thomas Ruch". Chosen on two
  weak tiebreakers — it is the more recent, and it is the same collaboration that
  brings him onto this paper. ⚠️ **Worth confirming with him**; the default is
  defensible but rests on two data points.

Note that both ORCID *records* give the shorter forms. That is the name each typed
into a profile, not the name they publish under, so it is not the authority for
how a byline should read.

## Before submission

- `date:` in the frontmatter is the submission date and will need updating.
- Several entries carry truncated author lists with a `note` field reading
  "restore from sorcha_paper.bib if required" — editorial TODOs that should not
  ship in a submitted artifact.
- `veres2017` and the two Anthropic entries are currently cited by neither paper.
  Cite them or cull them; JOSS prefers a tight reference list.
