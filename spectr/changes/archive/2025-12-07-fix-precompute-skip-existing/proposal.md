# Change: Skip Already Precomputed Values

## Why

The precompute application currently reprocesses all files from scratch on every run, wasting significant time (~30+ minutes) when chunks already exist. Since the source dataset is static, once a split is preprocessed, it should be skipped on subsequent runs.

## What Changes

- Add chunk existence detection before processing each split (train/validation)
- Skip `process_and_save_split()` if chunks already exist for that split
- Print informative message when skipping
- No changes to the `--force` behavior or validation mode

## Impact

- Affected code: `apps/precompute/precompute/main.py`
- No breaking changes
- No new dependencies
- Reduces redundant processing from ~30 minutes to ~0 seconds when chunks exist
