# Implementation Tasks: Add Poster Project

## 1. Core Poster Structure

### 1.1 Create poster directory and main files
- [ ] Create `poster/` directory at project root
- [ ] Create `poster/poster.tex` with tikzposter document class configuration
  - Set document class: `\documentclass[25pt, a0paper, portrait]{tikzposter}`
  - Configure page margins and spacing
  - Include custom theme definition from `poster-config.tex`
- [ ] Verify poster compiles with `nix develop -c ltx-compile poster/poster.tex`

### 1.2 Create poster configuration file
- [ ] Create `poster/poster-config.tex`
  - Define ISU color palette (cardinal #C8102E, gold #F1BE48)
  - Create tikzposter color style using `\definecolorstyle`
  - Create background style using `\definebackgroundstyle`
  - Set block and title styling
  - Configure fonts and text alignment

### 1.3 Set up asset reuse
- [ ] Update `poster.tex` to reference `../assets/` for figures
  - Example: `\includegraphics[width=0.3\textwidth]{../assets/unet.png}`
- [ ] Create sample blocks using project figures (unet.png, kv260.png, title-logo.png)
- [ ] Verify all images display correctly

### 1.4 Integrate bibliography
- [ ] Configure bibliography in `poster.tex`
  - Add `\addbibresource{../references.bib}`
  - Configure IEEE citation style via biblatex
  - Add sample citations in poster content
- [ ] Verify bibliography compiles and citations render correctly

## 2. ISU Custom Theme Implementation

### 2.1 Design ISU color scheme
- [ ] Define primary colors in `poster-config.tex`:
  - Cardinal red: `\definecolor{isucardinal}{RGB}{200, 16, 46}` (#C8102E)
  - Gold: `\definecolor{isugold}{RGB}{241, 190, 72}` (#F1BE48)
  - Accent colors (dark gray, light gray for contrast)
- [ ] Apply colors to title block (cardinal background with gold accents)
- [ ] Apply colors to content blocks (gold borders, cardinal headings)

### 2.2 Create professional layout
- [ ] Set title block styling:
  - Large, prominent title (project name: VisionAssist)
  - Subtitle area (for institution, authors, date)
  - Logo placement (ISU logo in corner)
- [ ] Configure content blocks:
  - Rounded corners or custom shape
  - Consistent spacing and padding
  - Shadow or border effects for depth
- [ ] Set up column layout options:
  - 1-column layout option (full width)
  - 2-column layout option (left/right split)
  - 3-column layout option (header + 2-column content)

### 2.3 Configure fonts and typography
- [ ] Set main font (use standard serif or sans-serif)
- [ ] Configure font sizes:
  - Title: large (40-50pt)
  - Block headers: medium (28-32pt)
  - Body text: readable (18-24pt)
  - Small text: 14-16pt for captions
- [ ] Ensure readability from 6+ feet away (symposium viewing distance)

## 3. Poster Content Structure

### 3.1 Create modular sections
- [ ] Create `poster/sections/` directory (optional but recommended)
- [ ] Create template files:
  - `poster/sections/title.tex` (project title, authors, institution)
  - `poster/sections/introduction.tex` (problem statement, motivation)
  - `poster/sections/methodology.tex` (approach, technical overview)
  - `poster/sections/results.tex` (key findings, performance metrics)
  - `poster/sections/conclusion.tex` (impact, next steps)
  - `poster/sections/acknowledgments.tex` (funding, team credits)

### 3.2 Populate content
- [ ] Extract key points from `sections/01-introduction.tex` for introduction block
- [ ] Extract methodology from `sections/04-design.tex` for methodology block
- [ ] Extract results/performance from `sections/06-implementation.tex` for results block
- [ ] Add conclusions from `sections/07-conclusion.tex` for conclusion block
- [ ] Add project team names and acknowledgments

### 3.3 Add visual elements
- [ ] Place title-logo.png prominently (top or corner)
- [ ] Include system architecture diagram (unet.png or kv260.png with captions)
- [ ] Add performance metrics in visual format (bars, icons)
- [ ] Include QR code for GitHub repo (optional but recommended)

## 4. Build System Integration

### 4.1 Update Nix flake
- [ ] Modify `flake.nix` lint script:
  - Change from: `chktex main.tex sections/*.tex`
  - Change to: Include `chktex poster/poster.tex poster/sections/*.tex` (if sections exist)
- [ ] Verify lint passes: `nix develop -c lint`

### 4.2 Verify local compilation
- [ ] Test compilation with pdflatex: `nix develop -c ltx-compile poster/poster.tex`
- [ ] Test watch mode: `nix develop -c ltx-watch poster/poster.tex` (Ctrl+C to exit)
- [ ] Verify `poster.pdf` is generated correctly
- [ ] Check for compilation warnings or errors

### 4.3 Create README for poster project
- [ ] Create `poster/README.md` with:
  - Quick start instructions (how to compile)
  - Directory structure explanation
  - Asset reuse guidelines
  - Theme customization guide (how to change colors, fonts)
  - Printing guidelines (resolution, DPI, paper type)
  - Contributing guidelines

## 5. CI/CD Pipeline Integration

### 5.1 Update GitHub Actions CI workflow
- [ ] Modify `.github/workflows/ci.yml`:
  - Add new job: `build-poster` (parallel to `build` job)
  - Configure job to compile `poster/poster.tex`
  - Use `xu-cheng/latex-action@v4` with `root_file: poster/poster.tex`
  - Add artifact upload for `poster.pdf`
- [ ] Test CI workflow by pushing to feature branch
- [ ] Verify `poster.pdf` appears in workflow artifacts

### 5.2 Update GitHub Actions release workflow
- [ ] Modify `.github/workflows/release.yml`:
  - Add step to compile poster (similar to main document compilation)
  - Rename poster PDF to include version: `sddec25-01-poster-vX.Y.Z.pdf`
  - Upload renamed poster to GitHub release
  - Ensure poster PDF appears in release assets alongside main document
- [ ] Test release workflow on feature branch
- [ ] Create test release tag and verify both PDFs attached

### 5.3 Verify release integration
- [ ] Check that main document release still works unchanged
- [ ] Verify poster PDF is generated with correct version number
- [ ] Confirm both PDFs are attached to GitHub release
- [ ] Test downloading releases and verifying PDF integrity

## 6. Validation and Testing

### 6.1 Validate Spectr proposal
- [ ] Run `spectr validate add-poster-project --strict`
- [ ] Fix any validation errors
- [ ] Ensure all requirements have scenarios
- [ ] Verify delta operations (ADDED, MODIFIED, etc.) are correct

### 6.2 Test poster functionality
- [ ] Compile poster locally: `nix develop -c ltx-compile poster/poster.tex`
- [ ] Verify `poster.pdf` renders correctly:
  - [ ] Title block displays with correct ISU colors
  - [ ] All images load and display (no broken links)
  - [ ] Bibliography appears with correct citations
  - [ ] Text is readable at expected viewing distance
  - [ ] Layout is visually appealing (no overlapping elements)
- [ ] Verify figures appear correctly:
  - [ ] Assets from `../assets/` load without errors
  - [ ] Image sizing is appropriate
  - [ ] Captions render correctly

### 6.3 Test build system
- [ ] Verify lint passes: `nix develop -c lint`
- [ ] Verify compilation works: `nix develop -c ltx-compile poster/poster.tex`
- [ ] Verify watch mode works: `nix develop -c ltx-watch poster/poster.tex`
- [ ] Verify no auxiliary files left in git (check `.gitignore`)

### 6.4 Test CI/CD
- [ ] Push branch with poster changes
- [ ] Verify CI builds main document (unchanged behavior)
- [ ] Verify CI builds poster (new job)
- [ ] Verify both artifacts appear in workflow
- [ ] Create test release and verify both PDFs attached

## 7. Documentation and Finalization

### 7.1 Update project documentation
- [ ] Add poster section to main `README.md` (if exists)
  - Quick link to `poster/README.md`
  - Brief description of poster purpose
- [ ] Update `CLAUDE.md` project context (if needed)
  - Mention poster as presentation capability
- [ ] Add poster build instructions to development guide (if exists)

### 7.2 Final validation
- [ ] Run full `spectr validate add-poster-project --strict` (all items must pass)
- [ ] Review proposal.md for completeness
- [ ] Review design.md for clarity
- [ ] Review all spec deltas for scenarios and requirements
- [ ] Check that tasks.md covers all work items

### 7.3 Commit and cleanup
- [ ] Verify no uncommitted changes outside `spectr/changes/add-poster-project/`
- [ ] Commit proposal files with message: `proposal(poster): add senior design symposium poster project`
- [ ] Push to feature branch for review

## Summary

**Phase 1 (Core Structure)**: Tasks 1.1-1.4 - Establish basic poster and asset integration
**Phase 2 (ISU Branding)**: Tasks 2.1-2.3 - Implement custom theme
**Phase 3 (Content)**: Task 3 - Add content and visuals
**Phase 4 (Build)**: Task 4 - Local build system integration
**Phase 5 (CI/CD)**: Task 5 - GitHub Actions integration
**Phase 6 (Validation)**: Task 6 - Comprehensive testing
**Phase 7 (Finalization)**: Task 7 - Documentation and cleanup

**Estimated effort**: 6-8 hours total
- Phase 1: 1 hour
- Phase 2: 1.5 hours
- Phase 3: 1.5 hours
- Phase 4: 0.5 hours
- Phase 5: 1.5 hours
- Phase 6: 1 hour
- Phase 7: 0.5 hours
