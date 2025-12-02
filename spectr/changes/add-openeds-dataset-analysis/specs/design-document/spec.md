## ADDED Requirements

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
