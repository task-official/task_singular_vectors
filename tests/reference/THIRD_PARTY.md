# Frozen numerical test references

These files are independent numerical oracles, not runtime implementations.
Tests do not access GitHub or require Git history at runtime.

* `tsvm_reference.py`: AntoAndGar/task_singular_vectors at
  `a6c97188f7aa0bb20753de4f2939be89dd64f1c3`, Apache-2.0 (repository LICENSE).
* `iso_reference.py`: danielm1405/iso-merging at
  `d34010af8ba129ef4d17a91ca9943b413fe06f8c`, Apache-2.0 (repository LICENSE).
  Source blob `1e02c7b455ece261b8433e95f449432add2c7ee3`.
* `dc_reference.py`: Tobeginwith/DC-Merge at
  `089af77a378562272461e0f041217ec440a4a230`, MIT (`DC_LICENSE.txt`).
  Source blob `ab3915adb2c0ed12e73cb00821c6ddb0e91a82aa`.
  Only the FFT `dc_merge`, `keep_topk_percent`, and `ties_small` definitions
  are retained; unrelated methods are omitted.
* `gse_reference.py`: a frozen snapshot of this workspace's
  `GlobalRawEnergyTopR` path before refactoring, with source digest in its header.
  Only that selection branch and its numerical dependencies are retained.

Reference source spelling and numerical operation order are intentionally
preserved. Production changes must not rewrite these oracles to match a new
implementation.

* `experimental_reference.py` loads the frozen pre-migration source under
  `archive/legacy_code/src/methods/experimental/legacy_variants.py` for tests only.
  Production never imports archived modules. The research source retains its
  original numerical rules, including the endpoint behavior corrected in v2.
