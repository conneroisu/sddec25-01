---
applyTo: "*.tex,**/*.tex,*.bib"
---

# LaTeX Linting

This document describes the LaTeX linting instructions for the project.

We use the flake.nix provided `lint` command which uses the `chktex` package to lint on the entire project.

So, you can run the linting command by running `nix develop -c 'lint'`.

You should make sure after making changes to the LaTeX files that the linting command succeeds without errors or warnings.
