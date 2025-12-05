## 1. Documentation Updates
- [x] 1.1 Add HuggingFace dataset link to Dataset Acquisition subsection in `sections/06-implementation.tex`
- [x] 1.2 Add explanation of precomputed artifacts (binary labels, spatial weights, distance maps)
- [x] 1.3 Cross-reference `training/precompute.py` script that generates the dataset
- [x] 1.4 Verify LaTeX compilation succeeds with new content
- [x] 1.5 Run spell check (`nix develop -c 'lint'`)

## 2. Validation
- [x] 2.1 Verify HuggingFace link is accessible and correct
- [x] 2.2 Confirm dataset repository name matches `train.py` constant (`Conner/openeds-precomputed`)
- [x] 2.3 Check that PDF renders correctly with new content
- [x] 2.4 Run `spectr validate add-precomputed-dataset-link --strict`
