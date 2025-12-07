## 1. Implementation

- [x] 1.1 Add `has_existing_chunks(split_dir: Path) -> tuple[bool, int, int]` helper function that counts actual samples in NPZ chunks
- [x] 1.2 Modify `main()` to check for existing chunks before calling `process_and_save_split()`
- [x] 1.3 Print skip message with chunk count AND sample count when skipping a split
- [x] 1.4 Display accurate sample counts in summary output

## 2. Verification

- [x] 2.1 Python syntax verified with `py_compile`
- [ ] 2.2 Test that second run skips processing and shows correct sample counts
- [ ] 2.3 Verify HuggingFace dataset creation still works with existing chunks
