# Spec Delta: Build System Extension for Poster

## ADDED Requirements

### Requirement: Local Poster Compilation Support
The system SHALL extend local build tools to support compiling the poster LaTeX project with the same environment and commands used for the main documentation, enabling developers to build poster locally using `nix develop` environment.

#### Scenario: Compile poster with ltx-compile
- **WHEN** developer runs `nix develop -c ltx-compile poster/poster.tex` from project root
- **THEN** poster.pdf is generated successfully in poster/ directory
- **AND** compilation uses pdflatex engine via latexmk
- **AND** output includes success message

#### Scenario: Watch mode for active development
- **WHEN** developer runs `nix develop -c ltx-watch poster/poster.tex` from project root
- **THEN** poster is compiled initially
- **AND** poster is automatically recompiled whenever poster/ files change
- **AND** developer can save poster.tex and see updated PDF within 5 seconds
- **AND** Ctrl+C terminates watch mode

#### Scenario: Flexible engine selection
- **WHEN** developer needs to use different LaTeX engine
- **THEN** `nix develop -c ltx-compile poster/poster.tex pdflatex` uses pdflatex (default)
- **AND** `nix develop -c ltx-compile poster/poster.tex xelatex` uses xelatex (if needed)
- **AND** `nix develop -c ltx-compile poster/poster.tex lualatex` uses lualatex (if needed)

#### Scenario: Consistent environment
- **WHEN** developer uses `nix develop` environment
- **THEN** same Nix dependencies are available as for main document
- **AND** texliveFull, latexmk, biber, and other tools are accessible
- **AND** environment is reproducible across different machines

---

### Requirement: Linting and Validation for Poster
The system SHALL extend static analysis tooling to validate LaTeX syntax and style in poster files, ensuring code quality and consistency with project standards.

#### Scenario: Lint poster files
- **WHEN** developer runs `nix develop -c lint` from project root
- **THEN** lint command checks main.tex, sections/*.tex, AND poster/poster.tex
- **AND** lint reports any LaTeX style violations or warnings
- **AND** lint completes successfully (exit code 0) if no errors

#### Scenario: Detect LaTeX errors
- **WHEN** poster files contain common LaTeX errors (undefined macros, syntax errors)
- **THEN** lint command detects and reports them
- **AND** error messages include file name and line number
- **AND** developer can identify and fix issues before pushing

#### Scenario: Optional poster sections linting
- **WHEN** poster/sections/ directory exists with modular content files
- **THEN** lint command checks poster/sections/*.tex if directory exists
- **AND** all modular files are validated
- **AND** no errors are missed in optional structure

#### Scenario: Lint integration with development workflow
- **WHEN** developer makes changes to poster/ files
- **THEN** developer runs `nix develop -c lint` to validate before committing
- **AND** lint provides fast feedback (completes in <5 seconds)
- **AND** developer is encouraged to check lint before pushing

---

### Requirement: CI/CD Pipeline Compilation
The system SHALL extend GitHub Actions CI/CD pipeline to automatically compile poster on every push and pull request, validating that poster builds successfully and generating artifacts for review.

#### Scenario: Poster compilation in CI pipeline
- **WHEN** developer pushes commits to any branch
- **THEN** GitHub Actions workflow is triggered
- **AND** CI pipeline includes separate `build-poster` job (parallel to `build` job)
- **AND** poster/poster.tex is compiled with `xu-cheng/latex-action@v4`
- **AND** compilation completes successfully without errors

#### Scenario: Artifact generation in CI
- **WHEN** poster compilation succeeds in CI
- **THEN** poster.pdf is uploaded as workflow artifact
- **AND** artifact is named "poster" for easy identification
- **AND** artifact is available for download in "Artifacts" section of workflow run
- **AND** artifact retains for 90 days (default GitHub retention)

#### Scenario: Build failure on poster errors
- **WHEN** poster.tex contains errors that prevent compilation
- **THEN** build-poster job fails
- **AND** CI workflow shows failure status
- **AND** error logs are available in CI output for debugging
- **AND** developer is notified of compilation failure (if PR notifications enabled)

#### Scenario: Parallel builds for efficiency
- **WHEN** CI workflow runs
- **THEN** `build` job (main document) and `build-poster` job run in parallel
- **AND** total CI execution time is approximately same as sequential builds
- **AND** both jobs report status independently (one can pass while other fails)

#### Scenario: Integration with existing CI
- **WHEN** main document CI job runs
- **THEN** main document compilation is unchanged and unaffected
- **AND** poster build is completely independent
- **AND** failure in poster build doesn't prevent main document artifact upload
- **AND** existing CI behavior is preserved

---

### Requirement: Release Workflow Integration
The system SHALL extend GitHub Actions release workflow to automatically compile poster with version number and attach to GitHub releases, ensuring poster is always available as release artifact.

#### Scenario: Poster compilation in release workflow
- **WHEN** new release is created on main branch
- **THEN** GitHub Actions release workflow is triggered
- **AND** workflow compiles poster/poster.tex with version number
- **AND** poster.pdf is generated successfully
- **AND** compilation uses same version number as main document release

#### Scenario: Version number in poster artifact
- **WHEN** poster.pdf is generated during release
- **THEN** PDF is renamed to include version: `sddec25-01-poster-vX.Y.Z.pdf`
- **AND** version number matches main document version (from VERSION file)
- **AND** filename pattern matches main document pattern: `sddec25-01-vX.Y.Z.pdf`

#### Scenario: Poster attachment to GitHub release
- **WHEN** release workflow completes successfully
- **THEN** poster PDF is attached to GitHub release
- **AND** poster appears in "Assets" section of release alongside main document
- **AND** poster can be downloaded from release page
- **AND** both main document and poster are available for download

#### Scenario: Release workflow orchestration
- **WHEN** release workflow runs
- **THEN** main document compilation completes first
- **AND** release tag is created
- **AND** main document PDF is attached
- **AND** poster is compiled
- **AND** poster PDF is attached
- **AND** release is complete with all assets

#### Scenario: Version synchronization
- **WHEN** VERSION file is read during release
- **THEN** same version is used for both main document and poster
- **AND** both PDFs have same version number in filename
- **AND** both PDFs appear in same GitHub release
- **AND** version is synchronized automatically (no manual step needed)

---

### Requirement: Flake.nix Configuration
The system SHALL extend flake.nix Nix development configuration to support poster compilation with existing build scripts and dependencies, maintaining Nix-based reproducibility.

#### Scenario: Extend lint script in flake.nix
- **WHEN** flake.nix is examined
- **THEN** `lint` script configuration includes poster files:
  - Original: `chktex main.tex sections/*.tex`
  - Updated: `chktex main.tex sections/*.tex poster/poster.tex`
- **AND** script handles optional poster/sections/ directory (if exists)
- **AND** lint can be extended in future without breaking existing behavior

#### Scenario: No new dependencies added
- **WHEN** flake.nix is updated for poster support
- **THEN** no new Nix packages are added to dependencies
- **AND** texliveFull (existing) includes tikzposter
- **AND** latexmk, pdflatex, biber (existing) support poster
- **AND** build environment is unchanged

#### Scenario: Flake consistency
- **WHEN** developer reads flake.nix
- **THEN** configuration style is consistent with existing patterns
- **AND** new script additions follow established naming conventions
- **AND** comments explain any non-obvious decisions

---

### Requirement: Build Documentation
The system SHALL document poster compilation process in project documentation, explaining how to build locally, troubleshoot issues, and integrate with CI/CD.

#### Scenario: README build instructions
- **WHEN** developer reads main README.md (if exists)
- **THEN** documentation mentions poster as presentation capability
- **AND** README includes reference to `poster/README.md` for detailed instructions
- **AND** quick reference shows poster compilation commands

#### Scenario: Poster-specific build documentation
- **WHEN** developer examines poster/README.md
- **THEN** documentation includes:
  - Compilation: `nix develop -c ltx-compile poster/poster.tex`
  - Watch mode: `nix develop -c ltx-watch poster/poster.tex`
  - Linting: `nix develop -c lint`
  - Expected output location: `poster/poster.pdf`
  - Typical compile time: <30 seconds
- **AND** examples are clear and tested

#### Scenario: CI/CD documentation
- **WHEN** developer wants to understand poster in CI/CD
- **THEN** documentation explains:
  - Poster compiles on every push (CI pipeline)
  - Poster included in releases (release workflow)
  - Artifact available for PR review
  - Version synchronization with main document
- **AND** links to GitHub Actions workflows are provided

#### Scenario: Troubleshooting guide
- **WHEN** developer encounters build errors
- **THEN** poster/README.md includes troubleshooting section with:
  - Common errors and solutions
  - How to check for missing assets
  - How to validate bibliography
  - How to get more detailed error output
  - Contact/support information

---

### Requirement: Build Artifact Organization
The system SHALL organize build artifacts (PDFs, auxiliary files) logically, preventing unnecessary commits and maintaining clean git repository.

#### Scenario: Poster PDF in gitignore
- **WHEN** developer checks .gitignore
- **THEN** *.pdf pattern is present (or similar)
- **AND** poster.pdf is not tracked in git
- **AND** generated PDFs are excluded from version control
- **AND** only LaTeX source files are committed

#### Scenario: Auxiliary files ignored
- **WHEN** LaTeX compilation creates auxiliary files
- **THEN** .gitignore properly excludes:
  - .aux, .auxlock, .bbl, .bcf, .blg, .fdb_latexmk, .fls, .log, .out, .synctex.gz, .toc
- **AND** poster.aux, poster.log, etc. are not committed
- **AND** repository remains clean of build artifacts

#### Scenario: CI artifacts retained
- **WHEN** CI workflow produces poster.pdf
- **THEN** artifact is stored separately from git
- **AND** artifact is downloadable from GitHub Actions
- **AND** artifact is not pushed to git repository
- **AND** developer can download and review before pulling

---

## Summary

This specification defines the **build-system** extension capability, establishing requirements for integrating poster compilation into local development environment and GitHub Actions CI/CD pipeline. The capability ensures that poster builds are validated automatically, artifacts are generated reliably, and releases include both main document and poster.

**Key Deliverables**:
1. Extended `flake.nix` lint script for poster validation
2. GitHub Actions `build-poster` job in CI pipeline
3. Poster compilation and attachment in release workflow
4. Build documentation in poster/README.md
5. Artifact organization and git hygiene

**Success Criteria**:
- Poster compiles locally with same `nix develop` environment
- Lint passes on poster files
- CI pipeline automatically compiles poster on every push
- Poster artifact appears in workflow and releases
- Version is synchronized between main document and poster
- Build system is reproducible and well-documented
