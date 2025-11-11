.
├── AGENTS.md
├── build.sh
├── CLAUDE.md
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
│   └── UserTestingPlan.pdf
├── flake.lock
├── flake.nix
├── isusdd.cls
├── main.tex ( <- this is the main file)
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

**Project Goal**: Optimize a U-Net semantic segmentation algorithm to enable real-time eye tracking (60 FPS, processing 4 frames in <33.2ms total) on embedded hardware (AMD Kria KV260) while maintaining 99.8% IoU accuracy.

**Application Domain**: Medical assistive technology for individuals with mobility impairments (particularly wheelchair users with conditions like cerebral palsy or epilepsy) to:
- Enable eye-tracking control for wheelchairs and assistive devices
- Detect early warning signs of medical episodes through eye movement and posture analysis
- Autonomously respond to medical distress by repositioning the wheelchair to a safer position
- Provide proactive safety monitoring without compromising user independence

**Repository Scope**: This repository is dedicated to project documentation, including design documents, testing strategies, status reports, and technical specifications. The actual C++ implementation code for the embedded system is maintained in a separate repository.

**Target Audience**: Academic advisors, project clients, team members, and future maintainers who need to understand the project context, requirements, design decisions, and testing approaches.

## Tech Stack

### Documentation Stack (Primary)
- **LaTeX**: Full TeX Live distribution for academic documentation
  - `pdflatex` and `latexmk` for compilation
  - `biblatex` with `biber` backend for IEEE-style references
  - Custom `isusdd.cls` document class (Iowa State University Senior Design Document)
- **Build Tools**:
  - Custom `build.sh` script for LaTeX compilation
  - `ltx-compile`: Compile LaTeX documents
  - `ltx-watch`: Auto-compile on file changes
  - `ltx-clean`: Clean auxiliary files
  - `ltx-spell`: Spell checking with aspell
  - `ltx-wordcount`: Word counting with texcount

### Development Environment
- **Nix/NixOS**: Development environment management via `flake.nix`
- **direnv**: Automatic environment loading (`.envrc` with "use flake")
- **Git/GitHub**: Version control (client has read-only access)
- **Docker**: Containerization for development consistency

### LaTeX Packages & Dependencies
- **Document Structure**: geometry, hyperref, microtype, ragged2e
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

### Communication & Collaboration
- **GitHub**: Version control and client tracking
- **Telegram**: Client communication
- **Discord**: Team communication

## Project Conventions

### Code Style

#### LaTeX Document Conventions

**File Organization**:
- Modular chapter-based structure (`sections/*.tex`)
- Main document: `main.tex`
- Separate bibliography: `references.bib`
- Custom document class: `isusdd.cls`
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
- Use OpenSpec methodology for specification-driven development

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
- Run `ltx-spell main.tex` before committing LaTeX changes
- Verify compilation with `./build.sh` succeeds without errors
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

**OpenSpec Specification-Driven Development**:
- Specifications live in `openspec/specs/`
- Change proposals in `openspec/changes/`
- Archived (completed) changes in `openspec/changes/archive/`
- Project conventions documented in `openspec/project.md` (this file)
- Use slash commands: `/openspec:proposal`, `/openspec:apply`, `/openspec:archive`

**Documentation as Primary Artifact**:
- LaTeX documents are the deliverable (not support materials)
- All design decisions documented before implementation
- Traceability from requirements through design to testing
- Version controlled alongside code (in separate repo)

#### Embedded System Architecture (Reference)
*Note: Implementation in separate repository*
- Four-stage pipelined architecture with POSIX threading
- Multi-threaded execution across ARM Cortex-A53 cores
- Round-robin DPU scheduling to prevent algorithm starvation
- Efficient buffer management within 4GB memory constraints

### Testing Strategy

#### LaTeX Documentation Testing

**Compilation Testing**:
- **Tool**: `latexmk` with `pdflatex`
- **Frequency**: Before every commit
- **Success Criteria**: PDF generation without errors
- **Process**: Run `./build.sh` and verify exit code 0

**Spell Checking**:
- **Tool**: `ltx-spell` (aspell with English dictionary)
- **Frequency**: Before every commit, especially after content changes
- **Process**: `ltx-spell main.tex` and review flagged words
- **False Positives**: Add technical terms to personal dictionary

**Bibliography Validation**:
- **Tool**: `biber` (automatically run by latexmk)
- **Success Criteria**: All citations resolve, no undefined references
- **Process**: Check for warnings in build.log

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

#### Client Access Policy

- **Client has READ-ONLY access** to the repository
- Client can view all commits, branches, and history
- Client cannot push or create branches
- All client-visible commits should be clean and professional
- Avoid exposing WIP or incomplete work on main branch

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
1. **Eye-Tracking Control**: User controls wheelchair direction and speed through eye movements
2. **Medical Episode Detection**: System monitors for early warning signs (abnormal eye movements, posture changes)
3. **Autonomous Safety Response**: Upon detecting distress, system safely repositions wheelchair (e.g., to reclined position)
4. **Continuous Monitoring**: Background health monitoring without user interaction

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
- **Power**: Battery-optimized for mobile wheelchair deployment

**Baseline Context**:
- Previous year's implementation: 160ms per frame (too slow for real-time)
- **Project Goal**: 19x speedup while maintaining accuracy
- Achieved through: algorithm optimization, hardware acceleration (DPU), multi-threading

**Real-World Constraints**:
- Operates in variable lighting conditions
- Must handle head movements and user position shifts
- Continuous operation for hours (battery life considerations)
- Must maintain performance under sustained thermal load

### Academic Project Context

**Project Structure**:
- **Institution**: Iowa State University, Department of Computer Engineering
- **Course**: Senior Design (SDDEC25-01)
- **Team Name**: VisionAssist
- **Timeline**: Two-semester project (Fall 2024, Spring 2025)
- **Methodology**: Hybrid Waterfall + Agile with sprint-based development

**Deliverables**:
- Weekly status reports (8 per semester)
- Design document (comprehensive technical design)
- Testing strategy document
- User testing plan
- Lightning talk presentation
- Engineering standards compliance document
- Final project demonstration

**Evaluation Criteria**:
- Technical complexity and innovation
- Requirements fulfillment
- Documentation quality and completeness
- Testing rigor and validation
- Team collaboration and communication
- Client satisfaction

**Client Relationship**:
- Regular communication via Telegram
- Monthly formal check-ins
- Client has read-only GitHub access for tracking progress
- Client provides domain expertise and requirements validation

## Important Constraints

### Technical Constraints

**Hardware Limitations**:
- **Memory**: Limited to 4GB DDR on AMD Kria KV260
  - Must fit neural network model, frame buffers, and processing pipeline
  - Average usage: 3.2GB (leaves minimal headroom)
- **Processing Power**: Shared DPU between multiple algorithms
  - DPU scheduling must prevent algorithm starvation
  - Must balance workload across ARM cores
- **Power Budget**: Battery-powered wheelchair deployment
  - Optimize for energy efficiency
  - Thermal management under sustained load

**Performance Requirements**:
- **Real-Time Processing**: Strict 60 FPS deadline (16.67ms per frame budget)
  - Current target: 4 frames in <33.2ms total
  - Cannot degrade accuracy to achieve speed
- **Accuracy Floor**: Must maintain 99.8% IoU
  - No compromise on segmentation quality
  - Safety-critical application requires high reliability
- **Latency Ceiling**: <100ms for emergency medical response
  - Includes detection, decision, and actuation time

**Platform Dependency**:
- Solution dependent on AMD Kria KV260 architecture (Zynq UltraScale+ MPSoC)
- Vitis-AI framework required for DPU utilization
- Limited portability to other embedded platforms
- FPGA fabric utilization introduces complexity

**Concurrency Complexity**:
- Multi-threaded pipeline with POSIX threading
- Thread synchronization overhead
- Race conditions and deadlock prevention
- Load balancing across cores

### Project Constraints

**Timeline**:
- **Academic Schedule**: Fixed semester deadlines (no extensions)
- **Two-Semester Scope**: Must complete within Spring 2025
- **Milestone Deliverables**: Specific dates for documents and presentations
- **Weekly Reporting**: Consistent progress demonstrations required

**Budget**:
- **No Hardware Upgrades**: Must work with existing Kria KV260
- **Cost-Effective Solution**: Academic project budget limitations
- **Open-Source Tools**: Prefer free/open-source over commercial licenses

**Team Resources**:
- **Student Team**: Limited to team members' availability and skill sets
- **Learning Curve**: Time required to learn Vitis-AI, FPGA development, embedded optimization
- **Access to Hardware**: Shared lab equipment (potential scheduling conflicts)

**Client Constraints**:
- **Read-Only Repository Access**: Client cannot directly contribute code/docs
- **NDA Restrictions**: Some numerical values are placeholders (cannot disclose proprietary data)
- **Communication Channels**: Primarily Telegram and monthly meetings

### Regulatory and Standards Constraints

**IEEE Standards Compliance**:
- Must document adherence to IEEE 3129-2023, 2802-2022, 7002-2022, 2952-2023
- Standards-driven design and testing processes
- Traceability from standards to implementation

**Medical Device Context**:
- While not FDA-regulated (academic project), design principles align with medical device standards
- Safety-critical application requires rigorous validation
- Privacy and security must meet healthcare data standards

**Academic Integrity**:
- Proper attribution of previous work (building on previous team's implementation)
- Original contributions clearly distinguished from baseline
- Citation of all references and prior art

### Scope Constraints

**Documentation Repository Limitations**:
- This repository focuses on documentation, not implementation
- C++ code maintained in separate repository
- OpenSpec methodology recently adopted (specs still being created)
- Documentation must be comprehensive but not overwhelming

**Implementation Constraints** (Reference for separate repo):
- Previous year's architecture must be understood and extended
- Cannot redesign from scratch (time limitations)
- Must maintain backward compatibility with existing interfaces

## External Dependencies

### Hardware Dependencies

**AMD Kria KV260 Development Board**:
- **Purpose**: Primary embedded platform for edge AI processing
- **Criticality**: Absolute dependency (project cannot function without it)
- **Version**: Specific to Kria KV260 (Zynq UltraScale+ MPSoC)
- **Procurement**: Provided by university lab (limited availability)
- **Documentation**: [AMD Kria KV260 Documentation](https://www.xilinx.com/products/som/kria/kv260-vision-starter-kit.html)

**Development Workstations**:
- Sufficient compute power for LaTeX compilation, Docker containers
- Adequate disk space for TeX Live distribution (~7GB)

### Software Dependencies

**Core Development Tools**:
- **Nix/NixOS** (v2.x or later): Development environment management
  - Declarative `flake.nix` defines all dependencies
  - Ensures reproducible builds across team members
  - Critical: All team members must use Nix for consistency
- **Git** (v2.x or later): Version control
  - GitHub for remote hosting and client access
- **Docker Engine**: Containerization for Vitis-AI development (separate repo)

**LaTeX Ecosystem**:
- **TeX Live** (full distribution): LaTeX compilation, packages
  - Provided by Nix flake (texliveFull)
  - Includes all required packages automatically
- **latexmk**: Build automation for LaTeX documents
- **biber**: Bibliography processing (biblatex backend)
- **aspell**: Spell checking (English dictionary)
- **texcount**: Word counting for LaTeX documents

**AI/ML Frameworks** (Separate Repo):
- **Vitis-AI**: AMD's framework for edge AI deployment on Kria KV260
  - Version compatibility critical (must match Kria firmware)
  - Docker-based development flow
- **ONNX Runtime**: Neural network model format and execution
- **GCC/G++**: ARM cross-compilation toolchain for embedded C++

**Language Server Protocols** (Optional but Recommended):
- **texlab**: LaTeX LSP for editor integration
- **ltex-ls**: Grammar and spell checking LSP

### External Services

**GitHub**:
- **Purpose**: Version control hosting, collaboration, client access
- **Repository**: github.com/[organization]/sddec25-01-dd
- **Access Control**: Team has read/write, client has read-only
- **Dependency Level**: High (primary collaboration platform)

**Communication Platforms**:
- **Telegram**: Client communication (real-time messages)
- **Discord**: Team communication and coordination
- **Criticality**: Medium (could switch platforms if needed)

### Reference Materials and Prior Work

**Previous Team's Work**:
- **Dependency**: Baseline implementation (160ms per frame)
- **Access**: Code repository, documentation, design decisions
- **Purpose**: Understanding existing architecture, identifying optimization opportunities
- **Criticality**: High (building on previous work, not starting from scratch)

**Academic Literature**:
- **U-Net Paper**: Ronneberger et al. - semantic segmentation architecture
- **Eye-Tracking Research**: Domain-specific papers on medical eye tracking
- **Optimization Techniques**: Embedded AI optimization literature

**IEEE Standards Documentation**:
- **IEEE 3129-2023**, **2802-2022**, **7002-2022**, **2952-2023**
- **Access**: Through university library subscriptions
- **Purpose**: Compliance validation, testing strategy guidance

**AMD/Xilinx Documentation**:
- Vitis-AI User Guide
- Kria KV260 Reference Manual
- Zynq UltraScale+ MPSoC Technical Reference
- **Access**: Public documentation, vendor support forums

### Dependency Management

**Nix Flake Lock**:
- `flake.lock` pins exact versions of all dependencies
- Ensures reproducible builds across time and machines
- Update strategy: Review and test updates before committing

**LaTeX Package Versions**:
- TeX Live distribution updated periodically (e.g., TeX Live 2024)
- Custom document class (`isusdd.cls`) version controlled in repository

**Risk Mitigation**:
- **Hardware Failure**: Backup Kria KV260 board if available
- **Service Outages**: Local Git repositories, offline development capability
- **Version Conflicts**: Nix ensures reproducible environments
- **Documentation Loss**: Regular commits, GitHub backups

---

*Last Updated*: 2025-11-11
*Document Version*: 1.0
*Maintained By*: VisionAssist Team (SDDEC25-01)
