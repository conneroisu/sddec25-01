## ADDED Requirements

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
