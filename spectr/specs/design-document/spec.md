# Design Document Specification

## Purpose

This specification defines the structure and content requirements for the VisionAssist project design document.

## Requirements

### Requirement: Dataset Selection Documentation
The design document SHALL include comprehensive documentation of training dataset selection decisions, including rationale for dataset choice, benefits, limitations, and domain transfer considerations.

#### Scenario: OpenEDS dataset context documented
- **WHEN** reader examines the Training Dataset section
- **THEN** document explains OpenEDS origin (VR headset eye-tracking at 200 Hz)
- **AND** document acknowledges VR-to-medical domain difference
- **AND** document presents both benefits and limitations transparently

#### Scenario: Benefits of OpenEDS documented
- **WHEN** reader seeks to understand why OpenEDS was chosen
- **THEN** document lists advantages: large scale (12,759+ annotated images), pixel-level segmentation masks, public availability, established research benchmark, professional annotation quality
- **AND** document explains how these benefits support the project goals

#### Scenario: Limitations of OpenEDS documented
- **WHEN** reader seeks to understand dataset limitations
- **THEN** document lists disadvantages: VR domain vs medical assistive technology domain, controlled illumination vs variable real-world lighting, healthy volunteer participants vs patients with medical conditions
- **AND** document acknowledges potential domain gap in model generalization

#### Scenario: Scalable practices rationale documented
- **WHEN** reader questions the domain mismatch decision
- **THEN** document explains that training methodology, preprocessing pipelines, and evaluation practices transfer across domains
- **AND** document notes client acknowledgment that scalable practices were the priority over domain-specific data

#### Scenario: Visual reference included
- **WHEN** reader views the dataset section
- **THEN** document includes figure showing OpenEDS sample image
- **AND** figure has descriptive caption explaining what is shown
- **AND** figure is properly cited to OpenEDS dataset

### Requirement: Precomputed Dataset Reference
The design document SHALL include a direct reference to the precomputed OpenEDS dataset hosted on HuggingFace Hub, enabling readers to access the exact training data used in the implementation.

#### Scenario: HuggingFace dataset link provided
- **WHEN** reader examines the Dataset Acquisition subsection
- **THEN** document includes HuggingFace Hub link to `Conner/openeds-precomputed`
- **AND** link is formatted as a proper citation or footnote
- **AND** link is accessible via web browser

#### Scenario: Precomputed artifacts explained
- **WHEN** reader seeks to understand dataset preprocessing
- **THEN** document explains that precomputed version includes binary labels, spatial weights, and distance maps
- **AND** document notes these artifacts eliminate runtime preprocessing overhead
- **AND** document clarifies GPU training efficiency benefits

#### Scenario: Script cross-reference included
- **WHEN** reader wants to understand how dataset was generated
- **THEN** document references `training/precompute.py` as the generation script
- **AND** document explains preprocessing pipeline workflow
- **AND** document maintains consistency with existing OpenEDS dataset documentation

#### Scenario: Reproducibility enhanced
- **WHEN** reader attempts to replicate training experiments
- **THEN** HuggingFace link provides direct access to exact training data
- **AND** precomputed format matches `train.py` expectations
- **AND** dataset version and source are traceable
