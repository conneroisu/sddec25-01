# VisionAssist Senior Design Poster

This directory contains the A0 portrait format poster for the VisionAssist senior design project (SDDEC25-01) at Iowa State University.

## Quick Start

To compile the poster locally:

```bash
cd poster
nix develop ../ -c ./build.sh
```

**Result**: Generates `poster/poster.pdf` (A0 portrait, 841mm x 1189mm / 33.1" x 46.8")

**Note**: Due to a known bug in tikzposter v2.0, direct use of `ltx-compile` may report errors despite successful PDF generation. The `build.sh` script handles this automatically.

## Directory Structure

```
poster/
├── poster.tex              # Main entry point - document structure
├── poster-config.tex       # ISU theme configuration (colors, fonts, styling)
├── sections/               # Modular content blocks
│   ├── title.tex          # Title, authors, institution
│   ├── introduction.tex   # Introduction and problem statement
│   ├── methodology.tex    # Methodology and approach
│   ├── results.tex        # Performance results
│   ├── conclusion.tex     # Conclusion and future work
│   └── acknowledgments.tex # Acknowledgments
├── poster.pdf             # Generated output (not committed to git)
└── README.md              # This file
```

**External dependencies (relative paths)**:
- `../assets/` - Reused figures (kv260.png, unet.png, title-logo.png, xilinx-vart-stack.png)
- `../references.bib` - Bibliography shared with main design document

## Asset Reuse Guidelines

The poster reuses figures from the main project's `../assets/` directory to maintain a single source of truth.

**Referencing assets in poster sections**:
```latex
\includegraphics[width=0.8\linewidth]{../assets/unet.png}
```

**Benefits**:
- Single source of truth for all project figures
- Updates to assets automatically reflect in poster
- Consistent visual identity across documentation

**Adding new figures**:
1. Place the figure in `../assets/` (root project assets directory)
2. Reference it in poster sections using relative path: `../assets/filename.png`
3. Do not duplicate figures in `poster/` directory

## Theme Customization Guide

All styling and theming is centralized in `poster-config.tex`. This file defines the Iowa State University branding.

### ISU Color Palette

Current colors defined in `poster-config.tex`:

- **Cardinal Red**: RGB(200, 16, 46) / `#C8102E`
- **Gold**: RGB(241, 190, 72) / `#F1BE48`
- **Dark Gray**: RGB(60, 60, 60) / `#3C3C3C`
- **Light Gray**: RGB(240, 240, 240) / `#F0F0F0`

### Customizing Colors

Edit `poster-config.tex` to change colors:

```latex
\definecolor{isucardinal}{RGB}{200,16,46}  % Primary brand color
\definecolor{isugold}{RGB}{241,190,72}     % Accent color
```

### Font Sizes

Base font size is **25pt** for readability at 6+ feet viewing distance. Adjust in `poster.tex`:

```latex
\documentclass[25pt, a0paper, portrait]{tikzposter}
```

Font size options: `12pt`, `14pt`, `17pt`, `20pt`, `25pt` (default for A0)

### Block Styling

Block appearance is controlled in `poster-config.tex`:

- **Rounded corners**: `\tikzposterlatexaffectionproofoff`
- **Border width**: Defined in block style settings
- **Spacing**: Controlled by `\setlength` commands

## Editing Content

Poster content is split into modular sections in the `sections/` directory. Each section is a self-contained LaTeX file.

### Section Files

| File | Content |
|------|---------|
| `title.tex` | Title, authors, institution (auto-generated from `\title`, `\author`, `\institute` in `poster.tex`) |
| `introduction.tex` | Problem statement, motivation, project goals |
| `methodology.tex` | Technical approach, architecture, design decisions |
| `results.tex` | Performance metrics, benchmarking, compliance validation |
| `conclusion.tex` | Key achievements, future work, project impact |
| `acknowledgments.tex` | Sponsors, advisors, collaborators, IEEE standards |

### Editing Workflow

1. **Edit section content**: Open the relevant `.tex` file in `sections/` directory
2. **Watch mode (recommended)**: Auto-recompile on file changes
   ```bash
   cd poster
   nix develop -c ltx-watch poster.tex
   ```
3. **Single compilation**: Compile once after editing
   ```bash
   cd poster
   nix develop -c ltx-compile poster.tex
   ```
4. **Cancel watch mode**: Press `Ctrl+C` to stop auto-compilation

### Content Guidelines

- **Concise language**: Use bullet points and short sentences for readability
- **Visual hierarchy**: Use bold text sparingly for emphasis
- **White space**: Avoid overcrowding blocks with too much text
- **Figures**: Include captions and reference assets from `../assets/`

## Printing Guidelines

### Format Specifications

- **Size**: A0 portrait (841mm x 1189mm / 33.1" x 46.8")
- **Paper type**: Recommended 200gsm poster paper or matte cardstock
- **Color profile**: RGB (poster is optimized for RGB printing)
- **Resolution**: PDF generation at 300+ DPI equivalent

### Print Recommendations

**Professional printing**:
- Use a professional print shop with large-format printers
- Request full-color poster on matte paper
- Digital printing preferred for color accuracy

**Test printing**:
- Print at 25% scale (fits on letter size: 8.5" x 11.7") to verify colors and layout
- Check for text readability and color accuracy before full-size print

**Viewing distance**:
- Poster is designed for readability from **6+ feet away**
- 25pt base font ensures legibility at conference poster sessions

### Color Accuracy

- Printed colors may vary slightly from screen display
- ISU cardinal red (#C8102E) should be vibrant and saturated
- Request color matching if printing professionally

## Build Commands Reference

All commands assume you are in the `poster/` directory.

### Compilation Commands

```bash
# Single compilation (standard workflow)
cd poster
nix develop -c ltx-compile poster.tex

# Watch mode (auto-recompile on file changes)
cd poster
nix develop -c ltx-watch poster.tex
# Press Ctrl+C to stop

# Clean auxiliary files
cd poster
latexmk -C poster.tex
```

### Linting and Validation

```bash
# Lint check (includes poster files as of flake.nix Phase 4)
nix develop -c lint

# Word count (from root directory)
nix develop -c ltx-wordcount poster/poster.tex
```

### Development Environment

```bash
# Enter Nix development shell
nix develop

# Development shell includes:
# - TeX Live Full (all LaTeX packages)
# - latexmk (build automation)
# - chktex (LaTeX linting)
# - biber (bibliography backend)
```

## Contributing Guidelines

### Before Committing

1. **Test local compilation**: Ensure `nix develop -c ltx-compile poster/poster.tex` succeeds
2. **Run lint check**: Fix any errors from `nix develop -c lint`
3. **Verify asset paths**: Ensure all `\includegraphics` use correct relative paths (`../assets/`)
4. **Review PDF output**: Check `poster.pdf` for formatting and content accuracy

### Code Style

- **Edit modular sections**: Always edit files in `sections/`, not `poster.tex` directly
- **Relative paths**: Use `../assets/` for figures, `../references.bib` for bibliography
- **Atomic commits**: One logical change per commit
- **Commit messages**: Follow conventional commits format (e.g., `docs(poster): update results section`)

### Common Pitfalls

- **Do not commit generated files**: `.pdf`, `.aux`, `.bbl`, etc. are in `.gitignore`
- **Do not duplicate assets**: Always reference `../assets/`, never copy files to `poster/`
- **Do not edit `poster.tex` for content**: Content belongs in `sections/` files
- **Do not use absolute paths**: Use relative paths (`../assets/`) for portability

## Troubleshooting

### Tikzposter Package Bug (Known Issue)

**Warning**: `Missing character: There is no 1 in font nullfont!`
- **Cause**: Bug in tikzposter v2.0 package - outputs "1=1" during initialization before fonts are loaded
- **Impact**: Causes `ltx-compile` to report exit code 12, but **PDF generates correctly**
- **Solution**: Use `./build.sh` instead of `ltx-compile` - the build script handles this automatically
- **Details**: The `.latexmkrc` file enables force mode to complete compilation despite warnings

This is a well-documented bug in tikzposter version 2.0 and does not affect the output quality. The generated PDF is valid and complete.

### Compilation Errors

**Error**: `! LaTeX Error: File 'poster-config.tex' not found`
- **Cause**: Running `ltx-compile` from wrong directory
- **Solution**: Always run from `poster/` directory: `cd poster && nix develop -c ltx-compile poster.tex`

**Error**: `! LaTeX Error: File '../assets/unet.png' not found`
- **Cause**: Image file missing or misspelled in `sections/*.tex`
- **Solution**: Verify filename in `../assets/` directory and check spelling in LaTeX code

**Error**: `! Package biblatex Error: File '../references.bib' not found`
- **Cause**: Bibliography file missing
- **Solution**: Ensure `references.bib` exists in root project directory

### Visual Issues

**Text too small/large**:
- **Solution**: Adjust font size in `poster.tex`: `\documentclass[20pt, a0paper, portrait]{tikzposter}`
- Options: `12pt`, `14pt`, `17pt`, `20pt`, `25pt`

**Colors look wrong**:
- **Solution**: Check color definitions in `poster-config.tex`
- Verify RGB values match ISU brand guidelines

**Images not loading**:
- **Solution**: Verify relative paths in `sections/*.tex` files
- Check that assets exist in `../assets/` directory
- Ensure image file extensions are lowercase (`.png`, not `.PNG`)

**Blocks overlapping**:
- **Solution**: Adjust column widths in `poster.tex`: `\column{0.5}` (50% width)
- Reduce content length in sections

### Build System Issues

**Lint warnings**:
- **Stylistic warnings** (dash length, table formatting) are non-blocking
- **Errors** (missing files, syntax errors) must be fixed
- See lint output for specific line numbers and issues

**Nix development shell not loading**:
- **Solution**: Ensure `flake.lock` is up to date: `nix flake update`
- Check that `flake.nix` is valid: `nix flake check`

## Links and References

- **Main project**: [README.md](../README.md) (root directory)
- **Design document**: [spectr/changes/add-poster-project/design.md](../spectr/changes/add-poster-project/design.md)
- **Development setup**: [CLAUDE.md](../CLAUDE.md) for Nix/direnv configuration
- **Spectr change proposal**: [spectr/changes/add-poster-project/proposal.md](../spectr/changes/add-poster-project/proposal.md)

## Support

For questions or issues:
1. Check this README for troubleshooting steps
2. Review the main project documentation
3. Consult the design document for technical decisions
4. Contact the senior design team (SDDEC25-01)

---

**Last updated**: 2025-11-19 (Phase 4: Build System Integration)
