# Working on daw

Read `guidelines.md` before changing anything. It is the source of truth for
how we test, lint and commit. The short version:

- Every new feature, bug fix or example ships with rigorous tests: known
  values, an independent reference implementation, Hypothesis properties,
  edge cases, and (for sound) measurements of the audio. A smoke test is not
  enough. The per-feature checklists are in `guidelines.md` section 5.
- A fixed bug gets a regression test that fails without the fix.
- Run everything with `nix flake check`; iterate with `nix develop -c pytest`.
  Coverage of `src/daw` must stay at 100% (line and branch).
- Run `nix develop -c python tools/mutation_check.py` after changing core logic.
- New files must be `git add`ed before the flake can see them.
- Commits: authored by the repo owner only, no `Co-Authored-By` trailers, unsigned
  (`--no-gpg-sign`). Do not push unless asked.
