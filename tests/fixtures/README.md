`runtime_before_072a87d.npz` records complete outputs from pristine decoder commit
072a87d294ed2e9388d1843a1065b41aff043ba4, using Python 3.12.14, NumPy 1.26.4,
Stim 1.16.0 and PyMatching 2.2.dev2 (fork commit 7a26e6a8ef20080e9eab7240ce33581cc3880d03).

It covers ordinary stage-2/original-DEM selection, color-correlated decoding,
relifting, perturbation M=1 and alpha=0, each with comparative decoding off/on.
All use d=3, rounds=3, uniform noise .02, 24 detector shots sampled with seed 51,
full output, validity checking and candidate export. Correlated b=2; relifting
retains non-edge-like errors. Perturbation has seed 19 and original-DEM scoring,
with M=1/alpha=1 or M=3/alpha=0.

Nested containers are flattened into slash-delimited keys; their lengths are
stored explicitly. None is encoded by the string __NONE__. Object arrays and
pickle loading are not used. Detector inputs are also saved in the fixture.
These frozen tests run only on the recorded NumPy/Stim/PyMatching versions,
since library changes can alter native tie choices, weights or DEMs. The
independent fresh-build tests run on other supported environments as well.
