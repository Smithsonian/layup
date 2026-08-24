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

## ⚠️ Author-name discrepancy with the AJ paper

The two papers carry the same 18 authors in the same order, but two names differ:

| | JOSS (`paper.md`) | AJ (`aj_paper.tex`) | ORCID record |
|---|---|---|---|
| 17 | Thomas Ruch | Thomas R. Ruch | Thomas Ruch |
| 18 | Carrie Holt | Carrie E. Holt | Carrie Holt |

The JOSS form matches the ORCID records; the AJ form was entered later and adds
middle initials. Every other author uses a middle initial where they have one, so
the JOSS entries also look inconsistent beside their co-authors.

**Deliberately not resolved here** — how someone's name appears in print is theirs
to choose. Ask both authors which form they want and make the two papers agree
before either is submitted.

## Before submission

- `date:` in the frontmatter is the submission date and will need updating.
- Several entries carry truncated author lists with a `note` field reading
  "restore from sorcha_paper.bib if required" — editorial TODOs that should not
  ship in a submitted artifact.
- `veres2017` and the two Anthropic entries are currently cited by neither paper.
  Cite them or cull them; JOSS prefers a tight reference list.
