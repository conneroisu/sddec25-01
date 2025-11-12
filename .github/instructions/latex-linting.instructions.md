---
applyTo: "*.tex,**/*.tex,*.bib"
---

# LaTeX Linting

This document describes the LaTeX linting instructions for the project.

We use the flake.nix provided lint command which uses the `chktex` and `ltx-spell` packages on the entire project.

So, you can run the linting command by running `nix develop -c 'lint'`.
