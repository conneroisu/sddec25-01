.
├── documents
│   ├── DesignDocSemester1.pdf
│   ├── EngineeringStandards.pdf
│   ├── LightningTalk.pdf
│   ├── Report1Semester1.pdf
│   ├── Report1Semester2.pdf
│   ├── Report2Semester1.pdf
│   ├── Report2Semester2.pdf
│   ├── Report3Semester1.pdf
│   ├── Report3Semester2.pdf
│   ├── Report4Semester1.pdf
│   ├── Report4Semester2.pdf
│   ├── Report5Semester1.pdf
│   ├── Report5Semester2.pdf
│   ├── Report6Semester1.pdf
│   ├── Report7Semester1.pdf
│   ├── Report8Semester1.pdf
│   ├── TestingStrategy.pdf
│   ├── UserTestingPlan.pdf
│   └── *.pdf
├── poster                   ( <- A0 portrait poster project)
│   ├── poster.tex           ( <- main poster entry point)
│   ├── poster-config.tex    ( <- ISU theme configuration)
│   ├── sections             ( <- modular poster content)
│   │   ├── title.tex
│   │   ├── introduction.tex
│   │   ├── methodology.tex
│   │   ├── results.tex
│   │   ├── conclusion.tex
│   │   └── acknowledgments.tex
│   ├── poster.pdf           ( <- generated A0 PDF, not committed)
│   └── README.md            ( <- poster-specific documentation)
├── VERSION ( <- version of the project automatically updated DO NOT EDIT)
├── flake.nix
├── isusdd.cls
├── main.tex ( <- this is the main design document file)
├── README.md
├── references.bib
└── sections
    ├── 01-introduction.tex
    ├── 02-requirements.tex
    ├── 03-project-plan.tex
    ├── 04-design.tex
    ├── 05-testing.tex
    ├── 06-implementation.tex
    ├── 07-conclusion.tex
    └── abstract.tex

# Project Context

## Purpose

This repository contains documentation for **VisionAssist** (SDDEC25-01), an Iowa State University Computer Engineering senior design project focused on optimizing AI-powered eye tracking systems for real-time medical monitoring and assistive technology.

**Application Domain**: Medical assistive technology for individuals with mobility impairments (particularly wheelchair users with conditions like cerebral palsy or epilepsy) to:
- Detect early warning signs of medical episodes through eye movement and posture analysis
- Autonomously respond to medical distress by repositioning the wheelchair to a safer position
- Provide proactive safety monitoring without compromising user independence

## Tech Stack

### Documentation Stack (Primary)
- **LaTeX**: Full TeX Live distribution for academic documentation
  - `pdflatex` and `latexmk` for compilation
  - `biblatex` with `biber` backend for IEEE-style references
  - Custom `isusdd.cls` document class (Iowa State University Senior Design Document)
  - `tikzposter` class for A0 portrait poster generation (poster/ subdirectory)

### Development Environment
- **Nix/NixOS**: Development environment management via `flake.nix`
- **direnv**: Automatic environment loading (`.envrc` with "use flake")

### LaTeX Packages & Dependencies
- **Document Structure**: geometry, hyperref, microtype, ragged2e
- **Poster Formatting**: tikzposter (A0 poster layout), booktabs (professional tables)
- **Code & Syntax**: listings, xcolor
- **Mathematics**: amsmath, amssymb
- **Graphics**: graphicx, float, subcaption
- **Bibliography**: biblatex (IEEE style)
- **Font Encoding**: UTF-8 input, T1 font encoding

### Embedded System Stack (Separate Repository)
*Note: The following technologies are used in the implementation but the code lives in a separate repository:*
- **Hardware Platform**: AMD Kria KV260 Development Board (Zynq UltraScale+ MPSoC, 4GB DDR, DPU for neural network acceleration)
- **Programming Language**: C++17/C++20
- **AI Framework**: Vitis-AI and Vitis-Runtime for edge deployment
- **Model Format**: ONNX (optimized neural network format)
- **Neural Network**: U-Net semantic segmentation model
- **Concurrency**: POSIX threading for multi-core ARM processing

## Project Conventions

### Code Style

#### LaTeX Document Conventions

**File Organization**:
- Modular chapter-based structure (`sections/*.tex`)
- Main document: `main.tex` (design document)
- Poster project: `poster/` subdirectory (A0 format poster with `tikzposter` class)
- Separate bibliography: `references.bib` (shared between main document and poster)
- Custom document class: `isusdd.cls` (main document only)
- Numbered section files: `01-introduction.tex`, `02-requirements.tex`, etc.

**Citation and References**:
- Use IEEE citation style with biblatex
- Bibliography managed with biber backend
- Citations format: `\cite{key}` for inline, `\parencite{key}` for parenthetical
- All references stored in `references.bib` with consistent BibTeX formatting

**Document Structure**:
- Formal academic writing with clear subsections
- Consistent hierarchy: chapters → sections → subsections
- Cross-referencing: `\label{chap:name}`, `\label{sec:name}`, `\ref{label}`
- Use of tables, enumerations, and itemized lists for clarity

**Naming Conventions**:
- LaTeX files: Numbered with descriptive names (`01-introduction.tex`)
- Labels: Prefix with type (`chap:`, `sec:`, `subsec:`, `fig:`, `tab:`)
- Figures: Descriptive filenames in `figures/` directory
- Consistent capitalization in section headings

**Formatting Standards**:
- Font encoding: UTF-8 input, T1 output
- Consistent indentation (2 spaces in LaTeX source)
- Line length: Aim for <100 characters for readability
- Comments: Use `%` for single-line comments, explain complex LaTeX macros

#### General Development Practices

**Documentation**:
- Comprehensive inline comments for complex logic
- README files for each major directory
- Maintain up-to-date documentation in sync with implementation

**Version Control**:
- Meaningful commit messages following conventional commits format (see Git Workflow)
- Commits should be atomic (one logical change per commit)
- Never commit generated files (PDFs, aux files) - use `.gitignore`
- Keep commit history clean and readable

**Code Review**:
- All changes reviewed by at least one team member before merging
- Review checklist: compilation success, formatting consistency, documentation completeness
- Address review comments promptly

**Spell Checking and Validation**:
- Run `nix develop -c 'lint'` before committing LaTeX changes (includes poster files)
- Verify compilation with `nix develop -c 'ltx-compile'` succeeds without errors
  - Main document: `ltx-compile main.tex`
  - Poster: `ltx-compile poster/poster.tex` (from root) or `ltx-compile poster.tex` (from poster/ directory)
- Check PDF output for formatting issues

### Architecture Patterns

#### Documentation Architecture

**Modular Document Structure**:
- Each major section in separate `.tex` file under `sections/`
- Main document (`main.tex`) includes sections with `\input{}`
- Separation of concerns: content (sections), style (isusdd.cls), references (references.bib)
- Reusable components: custom LaTeX commands, common preambles

**Build Pipeline**:
```
Source Files (*.tex) → pdflatex (pass 1) → biber (bibliography) →
pdflatex (pass 2) → PDF Output
```

**Document Hierarchy**:
```
main.tex (root)
├── isusdd.cls (custom class)
├── sections/
│   ├── abstract.tex
│   ├── 01-introduction.tex
│   ├── 02-requirements.tex
│   ├── 03-project-plan.tex
│   ├── 04-design.tex
│   ├── 05-testing.tex
│   ├── 06-implementation.tex
│   └── 07-conclusion.tex
└── references.bib (bibliography)
```

**Documentation as Primary Artifact**:
- LaTeX documents are the deliverable (not support materials)
- All design decisions documented before implementation
- Traceability from requirements through design to testing
- Version controlled alongside code (in separate repo)

#### Poster Architecture

The `poster/` subdirectory contains a self-contained LaTeX project for generating an A0 portrait format academic poster (841mm x 1189mm / 33.1" x 46.8") using the `tikzposter` document class.

**Poster Document Structure**:
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
└── README.md              # Poster-specific documentation
```

**External Dependencies**:
- `../assets/` - Shared figures (kv260.png, unet.png, title-logo.png, xilinx-vart-stack.png)
- `../references.bib` - Shared bibliography with main design document

**Design Principles**:
- **Modular architecture**: Content separated into `sections/*.tex` files for maintainability
- **Theme separation**: All styling (colors, fonts, block styles) isolated in `poster-config.tex`
- **Asset reuse**: Single source of truth for figures in `../assets/` directory
- **ISU branding**: Iowa State University color palette (Cardinal Red #C8102E, Gold #F1BE48)
- **Readability**: 25pt base font size for 6+ feet viewing distance

**Compilation**:
```bash
# Standard compilation (from poster/ directory)
cd poster
nix develop -c ltx-compile poster.tex

# Watch mode for active editing
cd poster
nix develop -c ltx-watch poster.tex

# From root directory
nix develop -c ltx-compile poster/poster.tex
```

**Key Differences from Main Document**:
- Uses `tikzposter` class instead of `isusdd.cls`
- Two-column layout optimized for poster format
- Shared bibliography and assets with main document
- Independent compilation (does not depend on main.tex)
- Designed for physical printing, not digital reading

**Build Integration**:
- Poster files included in lint checks (`nix develop -c 'lint'`)
- Same development environment as main document (TeX Live Full)
- Separate PDF output (poster.pdf vs main.pdf)

For detailed poster editing guidelines, see `poster/README.md`.

#### LaTeX Documentation Testing

**Compilation Testing**:
- **Tool**: `latexmk` with `pdflatex`
- **Frequency**: Before every commit
- **Success Criteria**: PDF generation without errors

**Bibliography Validation**:
- **Tool**: `biber` (automatically run by latexmk)
- **Success Criteria**: All citations resolve, no undefined references

**Visual Inspection**:
- **Frequency**: After significant formatting changes
- **Checks**:
  - Page breaks are appropriate
  - Figures and tables are properly positioned
  - Cross-references are correct
  - Table of contents is accurate
  - Headers and footers are consistent

**Word Count Tracking**:
- **Tool**: `ltx-wordcount` (texcount)
- **Purpose**: Ensure compliance with length requirements
- **Process**: `ltx-wordcount main.tex`

#### Continuous Compilation

**Watch Mode**:
- **Tool**: `ltx-watch main.tex`
- **Purpose**: Automatic recompilation on file changes during active writing
- **Benefits**: Immediate feedback on LaTeX errors, faster iteration

**Pre-commit Validation**:
- Run compilation test before every commit
- Abort commit if compilation fails
- Consider implementing git pre-commit hooks for automation

#### System Testing Philosophy (Reference)
*From project documentation - applies to embedded implementation:*
- **Test Early and Often**: Continuous validation throughout development
- **Unit Testing**: Feature map testing, layer-by-layer validation
- **Integration Testing**: Pipeline testing, thread synchronization verification
- **System Testing**: Full throughput and accuracy validation (60 FPS, 99.8% IoU)
- **Performance Testing**: Benchmarking against 33.2ms per 4-frame target
- **Hardware Testing**: FPGA validation on Kria KV260 development board

### Git Workflow

#### Branching Strategy

**Main Branch**:
- `main`: Production-ready documentation (always deployable)
- Protected branch: requires review before merge
- All deliverables generated from `main`

**Feature Branches**:
- Create feature branches for significant changes: `feature/testing-strategy`, `feature/design-chapter`
- Branch naming convention: `feature/`, `fix/`, `docs/`, `refactor/`
- Short-lived branches: merge within 1-2 weeks

**Working on Branches**:
```bash
# Create feature branch
git checkout -b feature/implementation-chapter

# Make changes, commit regularly
git add sections/06-implementation.tex
git commit -m "docs: add algorithm optimization section"

# Push to remote
git push -u origin feature/implementation-chapter

# Create pull request for review
# After approval, merge to main
```

#### Commit Conventions

**Conventional Commits Format**:
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `docs`: Documentation changes (LaTeX content)
- `style`: Formatting changes (no content change)
- `fix`: Fix errors in documentation or LaTeX code
- `feat`: Add new sections or major content
- `refactor`: Restructure without changing content
- `test`: Add or update validation scripts
- `chore`: Maintenance tasks (update .gitignore, build scripts)

**Examples**:
```bash
# Good commit messages
git commit -m "docs(requirements): add functional requirements for eye tracking"
git commit -m "fix(build): correct latexmk configuration for bibliography"
git commit -m "style(formatting): fix table alignment in testing chapter"
git commit -m "feat(testing): add comprehensive user testing plan"

# Bad commit messages (avoid these)
git commit -m "latest"
git commit -m "updates"
git commit -m "fix"
```

**Commit Best Practices**:
- Write clear, descriptive commit messages
- Use imperative mood ("add" not "added" or "adds")
- Keep subject line under 72 characters
- Reference issues/tickets in footer if applicable
- Make atomic commits (one logical change per commit)

#### Pull Request Process

**Creating PRs**:
1. Push feature branch to remote
2. Create pull request with descriptive title
3. Fill out PR template (if available) with:
   - Summary of changes
   - Sections affected
   - Compilation verification
   - Spell check confirmation

**Code Review**:
- At least one team member reviews before merge
- Reviewers check:
  - Content accuracy and clarity
  - LaTeX compilation success
  - Formatting consistency
  - No spelling errors
  - Appropriate citations

**Merging**:
- Squash commits if many small WIP commits
- Use merge commit for feature branches with clear history
- Delete feature branch after merge

### Release Workflow

**Semester Deliverables**:
- Tag releases for major deliverables: `v1.0-design-doc`, `v1.1-report-1`, etc.
- Generate PDF from `main` branch at tagged commit
- Upload to `documents/` directory for archival
- Create GitHub release with PDF attachment

**Version Numbering**:
- Semester deliverables: `vX.Y-description`
- Minor updates: increment Y
- Major milestones: increment X

## Domain Context

### Medical Assistive Technology

This project operates in the **medical assistive technology domain**, specifically focused on enhancing independence and safety for individuals with mobility impairments who use powered wheelchairs.

**Target User Population**:
- **Primary Users**: Individuals with mobility impairments using powered wheelchairs
  - Conditions include: cerebral palsy, epilepsy, cardiovascular disorders, neuromuscular diseases
  - Age range: Primarily adults, but system designed for adaptability
  - Technical proficiency: Varies widely (system must be intuitive)
- **Secondary Users**: Caregivers, family members, support staff
  - Need visibility into system status and alerts
  - May assist with configuration and troubleshooting
- **Tertiary Users**: Healthcare providers, emergency responders
  - Rely on system data for medical monitoring
  - Need accurate, reliable detection of medical episodes

**Use Cases**:
1. **Medical Episode Detection**: System monitors for early warning signs (abnormal eye movements, posture changes)
2. **Autonomous Safety Response**: Upon detecting distress, system safely repositions wheelchair (e.g., to reclined position)
3. **Continuous Monitoring**: Background health monitoring without user interaction

**Safety and Privacy Requirements**:
- **Real-Time Detection**: <100ms latency for emergency medical response
- **High Accuracy**: 99.8% Intersection over Union (IoU) for eye tracking segmentation
- **Privacy-Preserving**: All processing on-device (no cloud, no external data transmission)
- **Fail-Safe Design**: System must default to safe state on failure
- **User Autonomy**: Proactive assistance without removing user control

**Medical Standards Compliance**:
This project adheres to IEEE standards for AI-based medical devices:
- **IEEE 3129-2023**: AI-based image recognition robustness testing
- **IEEE 2802-2022**: AI-based medical device performance evaluation
- **IEEE 7002-2022**: Data privacy process implementation
- **IEEE 2952-2023**: Secure computing based on trusted execution environment

### Technical Performance Context

**Critical Performance Metrics**:
- **Processing Speed**: 60 FPS (4 frames in <33.2ms total, ~8.3ms per frame)
- **Accuracy**: 99.8% IoU (Intersection over Union) for semantic segmentation
- **Latency**: <100ms for medical emergency detection and response
- **Memory**: <4GB usage (average 3.2GB on Kria KV260)

### Academic Project Context

**Project Structure**:
- **Institution**: Iowa State University, Department of Computer Engineering
- **Course**: Senior Design (SDDEC25-01)
- **Methodology**: Hybrid Waterfall + Agile with sprint-based development

### Scope Constraints

**Documentation Repository Limitations**:
- This repository focuses on documentation, not implementation
- C++ code maintained in separate repository
- Documentation must be comprehensive but not overwhelming

**Implementation Constraints** (Reference for separate repo):
- Previous year's architecture must be understood and extended
- Cannot redesign from scratch (time limitations)
- Must maintain backward compatibility with existing interfaces

### Dependency Management

**Nix Flake Lock**:
- `flake.lock` pins exact versions of all dependencies
- Ensures reproducible builds across time and machines
- Update strategy: Review and test updates before committing

**LaTeX Package Versions**:
- TeX Live distribution updated periodically (e.g., TeX Live 2025)
- Custom document class (`isusdd.cls`) version controlled in repository

<!-- spectr:START -->
# Spectr Instructions

Instructions for AI coding assistants using Spectr for spec-driven development.

## TL;DR Quick Checklist

- Search existing work: `spectr spec list --long`, `spectr list` (use `rg` only for full-text search)
- Decide scope: new capability vs modify existing capability
- Pick a unique `change-id`: kebab-case, verb-led (`add-`, `update-`, `remove-`, `refactor-`)
- Scaffold: `proposal.md`, `tasks.md`, `design.md` (only if needed), and delta specs per affected capability
- Write deltas: use `## ADDED|MODIFIED|REMOVED|RENAMED Requirements`; include at least one `#### Scenario:` per requirement
- Validate: `spectr validate [change-id] --strict` and fix issues
- Request approval: Do not start implementation until proposal is approved

## Three-Stage Workflow

### Stage 1: Creating Changes
Create proposal when you need to:
- Add features or functionality
- Make breaking changes (API, schema)
- Change architecture or patterns
- Optimize performance (changes behavior)
- Update security patterns

Triggers (examples):
- "Help me create a change proposal"
- "Help me plan a change"
- "Help me create a proposal"
- "I want to create a spec proposal"
- "I want to create a spec"

Loose matching guidance:
- Contains one of: `proposal`, `change`, `spec`
- With one of: `create`, `plan`, `make`, `start`, `help`

Skip proposal for:
- Bug fixes (restore intended behavior)
- Typos, formatting, comments
- Dependency updates (non-breaking)
- Configuration changes
- Tests for existing behavior

**Workflow**
1. Review `spectr/project.md`, `spectr list`, and `spectr list --specs` to understand current context.
2. Choose a unique verb-led `change-id` and scaffold `proposal.md`, `tasks.md`, optional `design.md`, and spec deltas under `spectr/changes/<id>/`.
3. Draft spec deltas using `## ADDED|MODIFIED|REMOVED Requirements` with at least one `#### Scenario:` per requirement.
4. Run `spectr validate <id> --strict` and resolve any issues before sharing the proposal.

### Stage 2: Implementing Changes
Track these steps as TODOs and complete them one by one.
1. **Read proposal.md** - Understand what's being built
2. **Read design.md** (if exists) - Review technical decisions
3. **Read tasks.md** - Get implementation checklist
4. **Implement tasks sequentially** - Complete in order
5. **Confirm completion** - Ensure every item in `tasks.md` is finished before updating statuses
6. **Update checklist** - After all work is done, set every task to `- [x]` so the list reflects reality
7. **Approval gate** - Do not start implementation until the proposal is reviewed and approved

### Stage 3: Archiving Changes
After deployment, create separate PR to:
- Move `changes/[name]/` → `changes/archive/YYYY-MM-DD-[name]/`
- Update `specs/` if capabilities changed
- Use `spectr archive <change-id> --skip-specs --yes` for tooling-only changes (always pass the change ID explicitly)
- Run `spectr validate --strict` to confirm the archived change passes checks

## Before Any Task

**Context Checklist:**
- [ ] Read relevant specs in `specs/[capability]/spec.md`
- [ ] Check pending changes in `changes/` for conflicts
- [ ] Read `spectr/project.md` for conventions
- [ ] Run `spectr list` to see active changes
- [ ] Run `spectr list --specs` to see existing capabilities

**Before Creating Specs:**
- Always check if capability already exists
- Prefer modifying existing specs over creating duplicates
- Use `spectr show [spec]` to review current state
- If request is ambiguous, ask 1–2 clarifying questions before scaffolding

### Search Guidance
- Enumerate specs: `spectr spec list --long` (or `--json` for scripts)
- Enumerate changes: `spectr list` (or `spectr change list --json` - deprecated but available)
- Show details:
  - Spec: `spectr show <spec-id> --type spec` (use `--json` for filters)
  - Change: `spectr show <change-id> --json --deltas-only`
- Full-text search (use ripgrep): `rg -n "Requirement:|Scenario:" spectr/specs`

## Quick Start

### CLI Commands

```bash
# Essential commands
spectr list                  # List active changes
spectr list --specs          # List specifications
spectr show [item]           # Display change or spec
spectr validate [item]       # Validate changes or specs
spectr archive <change-id> [--yes|-y]   # Archive after deployment (add --yes for non-interactive runs)

# Project management
spectr init [path]           # Initialize Spectr
spectr update [path]         # Update instruction files

# Interactive mode
spectr show                  # Prompts for selection
spectr validate              # Bulk validation mode

# Debugging
spectr show [change] --json --deltas-only
spectr validate [change] --strict
```

### Command Flags

- `--json` - Machine-readable output
- `--type change|spec` - Disambiguate items
- `--strict` - Comprehensive validation
- `--no-interactive` - Disable prompts
- `--skip-specs` - Archive without spec updates
- `--yes`/`-y` - Skip confirmation prompts (non-interactive archive)

## Directory Structure

```
spectr/
├── project.md              # Project conventions
├── specs/                  # Current truth - what IS built
│   └── [capability]/       # Single focused capability
│       ├── spec.md         # Requirements and scenarios
│       └── design.md       # Technical patterns
├── changes/                # Proposals - what SHOULD change
│   ├── [change-name]/
│   │   ├── proposal.md     # Why, what, impact
│   │   ├── tasks.md        # Implementation checklist
│   │   ├── design.md       # Technical decisions (optional; see criteria)
│   │   └── specs/          # Delta changes
│   │       └── [capability]/
│   │           └── spec.md # ADDED/MODIFIED/REMOVED
│   └── archive/            # Completed changes
```

## Creating Change Proposals

### Decision Tree

```
New request?
├─ Bug fix restoring spec behavior? → Fix directly
├─ Typo/format/comment? → Fix directly
├─ New feature/capability? → Create proposal
├─ Breaking change? → Create proposal
├─ Architecture change? → Create proposal
└─ Unclear? → Create proposal (safer)
```

### Proposal Structure

1. **Create directory:** `changes/[change-id]/` (kebab-case, verb-led, unique)

2. **Write proposal.md:**
```markdown
# Change: [Brief description of change]

## Why
[1-2 sentences on problem/opportunity]

## What Changes
- [Bullet list of changes]
- [Mark breaking changes with **BREAKING**]

## Impact
- Affected specs: [list capabilities]
- Affected code: [key files/systems]
```

3. **Create spec deltas:** `specs/[capability]/spec.md`
```markdown
## ADDED Requirements
### Requirement: New Feature
The system SHALL provide...

#### Scenario: Success case
- **WHEN** user performs action
- **THEN** expected result

## MODIFIED Requirements
### Requirement: Existing Feature
[Complete modified requirement]

## REMOVED Requirements
### Requirement: Old Feature
**Reason**: [Why removing]
**Migration**: [How to handle]
```
If multiple capabilities are affected, create multiple delta files under `changes/[change-id]/specs/<capability>/spec.md`—one per capability.

4. **Create tasks.md:**
```markdown
## 1. Implementation
- [ ] 1.1 Create database schema
- [ ] 1.2 Implement API endpoint
- [ ] 1.3 Add frontend component
- [ ] 1.4 Write tests
```

5. **Create design.md when needed:**
Create `design.md` if any of the following apply; otherwise omit it:
- Cross-cutting change (multiple services/modules) or a new architectural pattern
- New external dependency or significant data model changes
- Security, performance, or migration complexity
- Ambiguity that benefits from technical decisions before coding

Minimal `design.md` skeleton:
```markdown
## Context
[Background, constraints, stakeholders]

## Goals / Non-Goals
- Goals: [...]
- Non-Goals: [...]

## Decisions
- Decision: [What and why]
- Alternatives considered: [Options + rationale]

## Risks / Trade-offs
- [Risk] → Mitigation

## Migration Plan
[Steps, rollback]

## Open Questions
- [...]
```

## Spec File Format

### Critical: Scenario Formatting

**CORRECT** (use #### headers):
```markdown
#### Scenario: User login success
- **WHEN** valid credentials provided
- **THEN** return JWT token
```

**WRONG** (don't use bullets or bold):
```markdown
- **Scenario: User login**  ❌
**Scenario**: User login     ❌
### Scenario: User login      ❌
```

Every requirement MUST have at least one scenario.

### Requirement Wording
- Use SHALL/MUST for normative requirements (avoid should/may unless intentionally non-normative)

### Delta Operations

- `## ADDED Requirements` - New capabilities
- `## MODIFIED Requirements` - Changed behavior
- `## REMOVED Requirements` - Deprecated features
- `## RENAMED Requirements` - Name changes

Headers matched with `trim(header)` - whitespace ignored.

#### When to use ADDED vs MODIFIED
- ADDED: Introduces a new capability or sub-capability that can stand alone as a requirement. Prefer ADDED when the change is orthogonal (e.g., adding "Slash Command Configuration") rather than altering the semantics of an existing requirement.
- MODIFIED: Changes the behavior, scope, or acceptance criteria of an existing requirement. Always paste the full, updated requirement content (header + all scenarios). The archiver will replace the entire requirement with what you provide here; partial deltas will drop previous details.
- RENAMED: Use when only the name changes. If you also change behavior, use RENAMED (name) plus MODIFIED (content) referencing the new name.

Common pitfall: Using MODIFIED to add a new concern without including the previous text. This causes loss of detail at archive time. If you aren't explicitly changing the existing requirement, add a new requirement under ADDED instead.

Authoring a MODIFIED requirement correctly:
1) Locate the existing requirement in `spectr/specs/<capability>/spec.md`.
2) Copy the entire requirement block (from `### Requirement: ...` through its scenarios).
3) Paste it under `## MODIFIED Requirements` and edit to reflect the new behavior.
4) Ensure the header text matches exactly (whitespace-insensitive) and keep at least one `#### Scenario:`.

Example for RENAMED:
```markdown
## RENAMED Requirements
- FROM: `### Requirement: Login`
- TO: `### Requirement: User Authentication`
```

## Troubleshooting

### Common Errors

**"Change must have at least one delta"**
- Check `changes/[name]/specs/` exists with .md files
- Verify files have operation prefixes (## ADDED Requirements)

**"Requirement must have at least one scenario"**
- Check scenarios use `#### Scenario:` format (4 hashtags)
- Don't use bullet points or bold for scenario headers

**Silent scenario parsing failures**
- Exact format required: `#### Scenario: Name`
- Debug with: `spectr show [change] --json --deltas-only`

### Validation Tips

```bash
# Always use strict mode for comprehensive checks
spectr validate [change] --strict

# Debug delta parsing
spectr show [change] --json | jq '.deltas'

# Check specific requirement
spectr show [spec] --json -r 1
```

## Happy Path Script

```bash
# 1) Explore current state
spectr spec list --long
spectr list
# Optional full-text search:
# rg -n "Requirement:|Scenario:" spectr/specs
# rg -n "^#|Requirement:" spectr/changes

# 2) Choose change id and scaffold
CHANGE=add-two-factor-auth
mkdir -p spectr/changes/$CHANGE/{specs/auth}
printf "## Why\\n...\\n\\n## What Changes\\n- ...\\n\\n## Impact\\n- ...\\n" > spectr/changes/$CHANGE/proposal.md
printf "## 1. Implementation\\n- [ ] 1.1 ...\\n" > spectr/changes/$CHANGE/tasks.md

# 3) Add deltas (example)
cat > spectr/changes/$CHANGE/specs/auth/spec.md << 'EOF'
## ADDED Requirements
### Requirement: Two-Factor Authentication
Users MUST provide a second factor during login.

#### Scenario: OTP required
- **WHEN** valid credentials are provided
- **THEN** an OTP challenge is required
EOF

# 4) Validate
spectr validate $CHANGE --strict
```

## Multi-Capability Example

```
spectr/changes/add-2fa-notify/
├── proposal.md
├── tasks.md
└── specs/
    ├── auth/
    │   └── spec.md   # ADDED: Two-Factor Authentication
    └── notifications/
        └── spec.md   # ADDED: OTP email notification
```

auth/spec.md
```markdown
## ADDED Requirements
### Requirement: Two-Factor Authentication
...
```

notifications/spec.md
```markdown
## ADDED Requirements
### Requirement: OTP Email Notification
...
```

## Best Practices

### Simplicity First
- Default to <100 lines of new code
- Single-file implementations until proven insufficient
- Avoid frameworks without clear justification
- Choose boring, proven patterns

### Complexity Triggers
Only add complexity with:
- Performance data showing current solution too slow
- Concrete scale requirements (>1000 users, >100MB data)
- Multiple proven use cases requiring abstraction

### Clear References
- Use `file.ts:42` format for code locations
- Reference specs as `specs/auth/spec.md`
- Link related changes and PRs

### Capability Naming
- Use verb-noun: `user-auth`, `payment-capture`
- Single purpose per capability
- 10-minute understandability rule
- Split if description needs "AND"

### Change ID Naming
- Use kebab-case, short and descriptive: `add-two-factor-auth`
- Prefer verb-led prefixes: `add-`, `update-`, `remove-`, `refactor-`
- Ensure uniqueness; if taken, append `-2`, `-3`, etc.

## Tool Selection Guide

| Task | Tool | Why |
|------|------|-----|
| Find files by pattern | Glob | Fast pattern matching |
| Search code content | Grep | Optimized regex search |
| Read specific files | Read | Direct file access |
| Explore unknown scope | Task | Multi-step investigation |

## Error Recovery

### Change Conflicts
1. Run `spectr list` to see active changes
2. Check for overlapping specs
3. Coordinate with change owners
4. Consider combining proposals

### Validation Failures
1. Run with `--strict` flag
2. Check JSON output for details
3. Verify spec file format
4. Ensure scenarios properly formatted

### Missing Context
1. Read project.md first
2. Check related specs
3. Review recent archives
4. Ask for clarification

## Quick Reference

### Stage Indicators
- `changes/` - Proposed, not yet built
- `specs/` - Built and deployed
- `archive/` - Completed changes

### File Purposes
- `proposal.md` - Why and what
- `tasks.md` - Implementation steps
- `design.md` - Technical decisions
- `spec.md` - Requirements and behavior

### CLI Essentials
```bash
spectr list              # What's in progress?
spectr show [item]       # View details
spectr validate --strict # Is it correct?
spectr archive <change-id> [--yes|-y]  # Mark complete (add --yes for automation)
```

Remember: Specs are truth. Changes are proposals. Keep them in sync.

<!-- spectr:END -->
