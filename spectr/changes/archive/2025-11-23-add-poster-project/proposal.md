# Change: Add Senior Design Symposium Poster Project

## Why

The VisionAssist project requires a professional visual presentation for the Iowa State University Senior Design Symposium. Currently, there is no mechanism to generate a formatted academic poster from project documentation. A dedicated tikzposter-based LaTeX project will enable:

1. **Professional symposium presentation** with A0 portrait format suitable for print
2. **Consistent branding** with ISU cardinal and gold theme
3. **Reuse of project assets** (figures, technical diagrams, references)
4. **Automated PDF generation** via CI/CD pipeline

This addition maintains the existing modular documentation approach while adding presentation capability for stakeholder engagement and project visibility.

## What Changes

- **New `poster/` directory** with self-contained tikzposter LaTeX project
  - Main file: `poster.tex` (A0 portrait, tikzposter document class)
  - Configuration: `poster-config.tex` (ISU custom theme definition)
  - Optional modular structure: `poster/sections/` for content organization
  - Documentation: `poster/README.md` (usage and customization guide)

- **Shared resources** (no duplication)
  - Figures from `../assets/` (unet.png, kv260.png, title-logo.png, etc.)
  - Bibliography via `../references.bib` (IEEE citation style)
  - Shared VERSION file with main document

- **ISU-branded custom theme**
  - Cardinal red (#C8102E) and gold (#F1BE48) color scheme
  - Professional layout matching university brand guidelines
  - Configurable font sizing and block styles

- **Build system integration**
  - Extend `flake.nix` lint script to include poster validation
  - Support poster compilation via `ltx-compile poster/poster.tex`
  - Enable watch mode: `ltx-watch poster/poster.tex`

- **CI/CD pipeline extension**
  - Add separate `build-poster` job to `.github/workflows/ci.yml`
  - Automatic poster PDF generation on every push/PR
  - Upload poster as artifact for review
  - Include poster in GitHub releases (tagged as `sddec25-01-poster-vX.Y.Z.pdf`)

- **Release workflow update**
  - Update `.github/workflows/release.yml` to compile poster with version number
  - Attach poster PDF to GitHub release alongside main document

## Impact

**New Capabilities**:
- `poster-generation` - Academic poster document generation with tikzposter
- `build-system` - Extended build tooling for poster compilation and linting

**Code Modifications**:
- `flake.nix` - Add poster linting to lint script; add poster build scripts
- `.github/workflows/ci.yml` - Add `build-poster` job
- `.github/workflows/release.yml` - Add poster compilation and artifact
- `.gitignore` - Add poster-specific auxiliary files (if not already covered by `*.pdf` ignore)

**No Breaking Changes**: Existing main document compilation and CI/CD workflow unchanged

**Backward Compatible**: Poster is entirely optional; main document builds and releases independently

## Dependencies

- **No new dependencies**: tikzposter included in existing `texliveFull`
- **No external services**: All compilation local or in standard CI/CD runner
- **No changes to existing specs**: First proposal in repository; adds new capabilities only

## Success Criteria

1. ✅ Poster compiles locally with `nix develop -c ltx-compile poster/poster.tex`
2. ✅ Poster PDF generated successfully in CI/CD pipeline
3. ✅ ISU custom theme renders correctly (colors, fonts, layout)
4. ✅ Figures from `assets/` display correctly in poster
5. ✅ Bibliography integration works (citations render)
6. ✅ Poster included in GitHub releases with proper versioning
7. ✅ Lint passes on poster LaTeX files
8. ✅ `spectr validate add-poster-project --strict` passes
