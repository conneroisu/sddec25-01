## ADDED Requirements

### Requirement: Skip Existing Precomputed Chunks

The precompute application SHALL skip processing for a split (train/validation) if NPZ chunk files already exist in the output directory, and SHALL count the actual samples in existing chunks.

#### Scenario: Chunks already exist

- **WHEN** running precompute with existing chunks in `parquet_chunks/train/`
- **THEN** the train split processing is skipped
- **AND** a message is printed showing the chunk count and sample count
- **AND** the sample count is determined by loading each NPZ and counting images

#### Scenario: No existing chunks

- **WHEN** running precompute with no existing chunks for a split
- **THEN** the split is processed normally and chunks are created

#### Scenario: Partial chunks exist

- **WHEN** only train chunks exist but not validation chunks
- **THEN** train processing is skipped with accurate sample count
- **AND** validation is processed normally
