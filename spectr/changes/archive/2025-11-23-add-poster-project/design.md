# Design Document: ISU-Branded Poster System

## Context

The VisionAssist senior design project requires a professional academic poster for presentation at the Iowa State University Senior Design Symposium. This design document outlines architectural decisions for creating a sustainable, reusable, and maintainable poster infrastructure using tikzposter.

**Stakeholders**:
- Project team (primary authors and maintainers)
- ISU Senior Design program (distribution/presentation context)
- Future teams (reusable template for similar projects)

**Constraints**:
- Must fit A0 portrait format (33.1" x 46.8" / 841mm x 1189mm)
- Must be viewable from 6+ feet away (reading distance for symposium)
- Must match ISU branding guidelines
- Must reuse existing project assets and references
- Must compile within existing development environment

**Requirements**:
- Professional appearance suitable for academic/symposium context
- Automatic CI/CD compilation and artifact generation
- Maintainable modular structure
- Zero breaking changes to main documentation project

---

## Goals

### Primary Goals
1. **Professional Presentation**: Create visually appealing, readable poster for symposium presentation
2. **ISU Branding**: Implement cardinal and gold color scheme consistent with university brand
3. **Asset Reuse**: Leverage existing figures, diagrams, and references from main documentation
4. **Automation**: Integrate with CI/CD for automatic compilation and release attachment
5. **Maintainability**: Structure code for easy updates and customization

### Non-Goals
- Automatic content extraction from main document (manual content composition)
- Support for multiple poster formats (A0 portrait fixed)
- Beamer slide deck (separate capability, future consideration)
- Web-based poster editor (static LaTeX only)
- Real-time collaboration platform (use existing git workflow)

---

## Decisions

### Decision 1: Use tikzposter for Poster Generation

**What**: Use tikzposter LaTeX package (included in texliveFull) for poster generation

**Why**:
- ✅ Already available in development environment (no new dependencies)
- ✅ Professional output comparable to PowerPoint/Illustrator posters
- ✅ Version controllable (LaTeX source in git)
- ✅ Integrates with existing build pipeline (latexmk/pdflatex)
- ✅ Strong community support and documentation
- ✅ Customizable themes (can create ISU brand theme)

**Alternatives Considered**:
1. **Beamer with poster mode**: Less visual flexibility, primarily designed for slides
2. **PowerPoint/Google Slides**: Not version controllable, requires export workflow, external tool
3. **Inkscape/Adobe Illustrator**: Non-collaborative, binary format, expensive (Illustrator)
4. **HTML/CSS (web poster)**: Overkill for print poster, adds framework complexity
5. **Custom TikZ drawings**: Would require significant manual styling work

**Trade-offs**:
- Pro: Version controlled, repeatable, professional
- Con: Steeper learning curve than WYSIWYG tools
- Mitigation: Comprehensive README and template provide guidance

**Status**: ✅ Accepted

---

### Decision 2: ISU Cardinal & Gold Custom Theme

**What**: Create custom tikzposter theme using ISU cardinal red and gold as primary colors

**Design Choices**:
- **Cardinal Red**: `RGB(200, 16, 46)` (#C8102E) - primary accent, headings, borders
- **Gold**: `RGB(241, 190, 72)` (#F1BE48) - secondary accent, highlights
- **Dark Gray**: `RGB(60, 60, 60)` (#3C3C3C) - body text
- **Light Gray**: `RGB(240, 240, 240)` (#F0F0F0) - backgrounds, subtle contrast

**Typography**:
- **Title block**: Large (40-48pt), bold, white on cardinal background
- **Block titles**: Medium (28-32pt), cardinal red, bold
- **Body text**: Regular (18-20pt), dark gray, sans-serif (sans-serif for better readability at distance)
- **Captions**: Small (14-16pt), gray, italics

**Layout Principles**:
- Title block: Full width, cardinal background with gold accent line
- Content blocks: Rounded corners (tikzposter default), gold borders (2pt), white fill
- Spacing: Generous padding around content (readability from distance)
- Hierarchy: Clear visual separation between title, main content, and footer

**Implementation**:
```latex
% In poster-config.tex
\definecolor{isucardinal}{RGB}{200, 16, 46}
\definecolor{isugold}{RGB}{241, 190, 72}
\definecolor{darkgray}{RGB}{60, 60, 60}
\definecolor{lightgray}{RGB}{240, 240, 240}

\definecolorstyle{isu}{}{
  \colorlet{colorOne}{isucardinal}
  \colorlet{colorTwo}{isugold}
  \colorlet{colorThree}{lightgray}
}{
  % Foreground colors
  \colorlet{textfg}{darkgray}
  \colorlet{titlebg}{isucardinal}
  \colorlet{titlefg}{white}
  \colorlet{blocktitlebg}{white}
  \colorlet{blocktitlefg}{isucardinal}
}
```

**Why This Approach**:
- Cardinal and gold are official ISU colors (recognizable, professional)
- Sufficient contrast for readability (cardinal on white, white on cardinal)
- Flexible color palette supports both light and dark backgrounds
- Customizable in one central file (`poster-config.tex`)

**Alternatives Considered**:
1. **Default tikzposter themes**: Functional but not branded; feels generic
2. **Professional template services**: Black-box, not version controllable
3. **Grayscale theme**: Professional but lacks ISU personality; less engaging

**Trade-offs**:
- Pro: Branded, distinctive, matches university guidelines
- Con: Requires careful color testing to ensure readability
- Mitigation: Test print at 100% scale before symposium

**Status**: ✅ Accepted

---

### Decision 3: Directory Structure & Modularity

**What**: Separate `poster/` directory at project root with optional modular subsections

**Structure**:
```
poster/
├── poster.tex              # Main entry point
├── poster-config.tex       # Theme configuration (ISU colors, fonts)
├── sections/              # Optional modular content
│   ├── title.tex
│   ├── introduction.tex
│   ├── methodology.tex
│   ├── results.tex
│   ├── conclusion.tex
│   └── acknowledgments.tex
└── README.md              # Usage and customization guide
```

**Why This Approach**:
- ✅ Parallel to main project structure (familiar pattern)
- ✅ Self-contained: can compile independently
- ✅ Modular: content can be organized logically
- ✅ Reusable: template can be copied for future posters
- ✅ Clean: separate concerns from main document

**Shared Resources**:
- `../assets/` - Reused via relative paths in poster.tex
- `../references.bib` - Reused via relative paths in poster.tex
- `../VERSION` - Shared version number in releases

**Alternatives Considered**:
1. **Nested in main.tex**: Less separation, harder to manage independently
2. **Flat directory**: All files at root level; becomes cluttered
3. **presentations/ → poster/ → slides/**: Overkill if no slides planned yet
4. **Separate git repository**: Loses synchronization with main project timeline

**Trade-offs**:
- Pro: Clean separation, reusable, familiar
- Con: Two builds instead of one (poster and document separate)
- Mitigation: CI/CD handles both builds automatically

**Status**: ✅ Accepted

---

### Decision 4: Asset and Bibliography Reuse

**What**: Poster reuses figures and bibliography from main project via relative paths

**Approach**:
- Figures: `\includegraphics{../assets/figure-name.png}`
- Bibliography: `\addbibresource{../references.bib}`
- No asset duplication (single source of truth)

**Rationale**:
- ✅ Single source of truth (no outdated copies)
- ✅ Synchronized updates (figure changes reflect in both documents)
- ✅ Storage efficiency (no duplicate files)
- ✅ Consistency (same citations in poster and main document)

**Managing Asset Updates**:
- If figures are updated in `assets/`, both documents reflect changes automatically
- Poster content remains stable if only main document text is updated
- Bibliography updates automatically via shared references.bib

**Cited References in Poster**:
- Use same IEEE citation style as main document
- Only cite key papers (space-limited on poster)
- Include references section with critical citations

**Alternatives Considered**:
1. **Copy assets**: Simple but creates maintenance burden and inconsistency
2. **Symlinks**: Platform-dependent, can break on Windows
3. **Embedding**: Increases file size, harder to update

**Trade-offs**:
- Pro: Single source of truth, synchronized, efficient
- Con: Relative path sensitivity (moving poster breaks links)
- Mitigation: Directory structure is stable; unlikely to change

**Status**: ✅ Accepted

---

### Decision 5: CI/CD Integration Strategy

**What**: Poster compilation integrated into existing GitHub Actions workflow

**Build Strategy**:
- **CI (ci.yml)**: Compile poster on every push/PR (validates syntax)
  - Separate job from main document (parallel builds)
  - Upload poster.pdf as artifact for PR review
  - Fail build if poster doesn't compile

- **Release (release.yml)**: Include poster in releases
  - Compile poster with version number: `sddec25-01-poster-vX.Y.Z.pdf`
  - Attach to GitHub release alongside main document
  - Tag poster version same as main document

**Implementation**:
```yaml
# In ci.yml
build-poster:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: xu-cheng/latex-action@v4
      with:
        root_file: poster/poster.tex
    - uses: actions/upload-artifact@v4
      with:
        name: poster
        path: poster.pdf

# In release.yml
- name: Compile poster
  uses: xu-cheng/latex-action@v4
  with:
    root_file: poster/poster.tex
- name: Rename poster with version
  run: mv poster.pdf sddec25-01-poster-${{ env.VERSION }}.pdf
- name: Upload to release
  uses: ncipollo/release-action@v1
  with:
    artifacts: sddec25-01-poster-${{ env.VERSION }}.pdf
```

**Why This Approach**:
- ✅ Automatic validation (no broken posters merged to main)
- ✅ Parallel builds (faster than sequential)
- ✅ Artifact preservation (reviewers can download PR poster)
- ✅ Release integration (poster always available in releases)
- ✅ Version synchronization (poster and document same version)

**Failure Handling**:
- If poster fails to compile: Build fails, PR cannot merge (forces fix)
- If poster compiles but has warnings: Build succeeds (warnings don't fail)
- Mitigation: Local testing via `nix develop -c ltx-compile` before push

**Alternatives Considered**:
1. **Manual compilation**: No automation, easy to forget poster
2. **Optional CI job**: Allows broken posters to merge (inconsistent)
3. **Separate workflow**: Overcomplicated, harder to maintain sync

**Trade-offs**:
- Pro: Automatic, reliable, prevents broken releases
- Con: Adds CI/CD execution time (~1 minute per build)
- Mitigation: Parallel jobs keep overall time reasonable

**Status**: ✅ Accepted

---

### Decision 6: Versioning Strategy

**What**: Poster shares VERSION file with main document; both get same version number in releases

**Rationale**:
- ✅ Single version source of truth
- ✅ Poster and document always synchronized in releases
- ✅ Simpler release workflow (no dual versioning)
- ✅ Customers/stakeholders see unified project version

**Release Naming**:
- Main document: `sddec25-01-vX.Y.Z.pdf`
- Poster: `sddec25-01-poster-vX.Y.Z.pdf`
- Both attached to same GitHub release `vX.Y.Z`

**Version Bumping**:
- Poster changes alone do NOT bump version (document is primary)
- Version bumps when main document changes (poster updates automatically)
- If poster-only critical fix needed: Consider if it justifies version bump

**Alternatives Considered**:
1. **Independent poster version**: Separate versioning, confusing for users
2. **No versioning on poster**: Hard to track which version is deployed
3. **Embed version in PDF**: Extra step, error-prone

**Trade-offs**:
- Pro: Unified versioning, simpler releases
- Con: Poster updates alone can't trigger version bump
- Mitigation: Rare case (most updates happen with document)

**Status**: ✅ Accepted

---

### Decision 7: Development Workflow (Local Compilation)

**What**: Developers compile poster locally using same `nix develop` environment as main document

**Commands**:
```bash
# Compile once
nix develop -c ltx-compile poster/poster.tex

# Watch mode (auto-recompile on changes)
nix develop -c ltx-watch poster/poster.tex

# Lint check
nix develop -c lint  # (includes poster after flake.nix update)
```

**Why This Approach**:
- ✅ Consistent with existing workflow
- ✅ No new tools or dependencies to learn
- ✅ Reproducible (Nix handles environment)
- ✅ Fast feedback (watch mode for active development)

**Advantages**:
- Same development environment as main document
- Pre-configured tools (latexmk, pdflatex, etc.)
- Lint validation included
- Watch mode for rapid iteration

**Testing Before Push**:
1. `nix develop -c ltx-compile poster/poster.tex` (compiles successfully)
2. `nix develop -c lint` (passes linting)
3. Manual PDF review (check visuals, readability)
4. Push to branch

**Alternatives Considered**:
1. **Direct pdflatex calls**: Less reliable, no latexmk benefits
2. **Docker container**: Extra complexity, slower startup
3. **GitHub Codespaces**: Not available to all team members

**Trade-offs**:
- Pro: Consistent, fast, integrated
- Con: Requires nix/direnv setup (already required for main project)
- Mitigation: Setup already done for main project

**Status**: ✅ Accepted

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| **Relative path breakage** | Low | High | Establish stable directory structure; document structure in README |
| **Color contrast issues** | Low | Medium | Test cardinal/gold contrast; manual print review before symposium |
| **Bibliography compilation error** | Low | Medium | Include sample citations in template; validate bibliography early |
| **CI/CD timeout** | Low | Low | Poster compile typically <30s; generous timeout in action |
| **Image asset missing** | Low | Medium | Validate all assets exist before commits; test locally |
| **Typography unreadable at distance** | Low | Medium | Test print at 100% scale (can't easily test on screen); use established sizing guidelines |
| **Theme customization breaks build** | Low | Low | Keep config in separate file; test before committing |

### High-Priority Mitigations

1. **Pre-Symposium Validation**: Print poster at 100% scale 2 weeks before event; verify readability from 6+ feet
2. **CI/CD Early Testing**: Test poster build during initial phases; don't wait until final push
3. **Asset Validation**: Verify all referenced assets exist; no typos in paths
4. **Relative Path Documentation**: Document directory structure and path resolution in README

---

## Migration Plan

**Phase 1 (Initial Setup)**:
1. Create `poster/` directory structure
2. Create `poster.tex` and `poster-config.tex` with ISU theme
3. Add sample content blocks with placeholder text
4. Test local compilation

**Phase 2 (Content Population)**:
1. Extract key content from main document sections
2. Add project figures (unet.png, kv260.png, etc.)
3. Add bibliography citations
4. Refine layout and spacing

**Phase 3 (CI/CD Integration)**:
1. Update `.github/workflows/ci.yml` with poster build job
2. Update `.github/workflows/release.yml` with poster compilation
3. Test on feature branch; verify artifacts created
4. Merge to main

**Phase 4 (Validation & Polish)**:
1. Final lint and compilation checks
2. Print review and readability validation
3. Color and contrast verification
4. Ready for symposium

**No Changes Required to**:
- Main document (poster is independent)
- Build scripts for main document (parallel builds)
- Release artifacts for main document (main PDF unchanged)

---

## Open Questions

1. **Custom ISU logo placement**: Should logo be in title block or corner? Need asset dimensions
2. **Poster content depth**: How many sections fit well on A0? Recommend 5-6 blocks max
3. **Print vendor/specifications**: Any special requirements (DPI, paper stock, lamination)?
4. **Presentation setup**: Will poster be printed large-format or displayed digitally at symposium?
5. **Post-symposium use**: Will poster be archived or updated for future presentations?

---

## Success Metrics

1. ✅ Poster compiles without errors locally and in CI/CD
2. ✅ ISU cardinal and gold colors render correctly
3. ✅ All referenced assets load and display
4. ✅ Bibliography citations render correctly
5. ✅ Text is readable from 6+ feet away (manual verification)
6. ✅ Layout is visually balanced and professional
7. ✅ Poster PDF appears in GitHub releases
8. ✅ Lint passes on poster files
9. ✅ Team approves visual design before symposium
10. ✅ Poster successfully used for symposium presentation

---

## References

- **tikzposter documentation**: http://www.ctan.org/pkg/tikzposter
- **ISU Brand Guidelines**: https://www.iastate.edu/brand/ (cardinal red, gold colors)
- **Academic Poster Standards**: A0 portrait is industry standard for academic symposiums
- **Typography for Print**: Sans-serif recommended for readability at distance
