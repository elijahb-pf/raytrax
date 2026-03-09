# Integration tests

These tests compare raytrax against reference output from
[TRAVIS](https://github.com/travis-ecrh/travis), a well-established ECRH ray-tracing code.
The reference data is committed to the repository, so **TRAVIS is not required to run the
tests**.

## Running the tests

From the repository root:

```bash
pytest integration_tests/ -m integration -v
```

The test suite loads the pre-generated reference file
`integration_tests/data/travis_w7x_reference.json` and compares raytrax results against it.

## Regenerating the reference data

When the physics or the test scenario changes you can regenerate the reference with
`generate_travis_reference.py`.  This requires the `travis-nc` binary to be on your `PATH`
and a VMEC wout equilibrium file:

```bash
python integration_tests/generate_travis_reference.py \
    --travis-exe $(which travis-nc) \
    --equilibrium /path/to/wout_w7x.nc \
    --output integration_tests/data/travis_w7x_reference.json
```

Commit the updated JSON file together with any code changes that motivated the regeneration.
