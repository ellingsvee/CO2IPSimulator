# Test suite

The suite protects the scientific and public behavior of CO2IPSimulator:

- grid construction and preservation of sealing layers;
- spill-graph parity with the reference implementation;
- trap-filling mass balance and physically expected parameter trends;
- observation and summary-statistic construction;
- forward-model parameter propagation; and
- ABC-SMC prior, posterior, predictive, and end-to-end behavior.

Tests intentionally do not pin plot styling, private example helpers, or trivial import
details. Those assertions made harmless presentation refactors unnecessarily expensive
without increasing confidence in the simulator or inference results.

Run the fast suite with:

```bash
uv run pytest -m "not slow"
```

The `slow` group validates the separately distributed Sleipner data and is skipped when
those data are unavailable:

```bash
uv run pytest -m slow
```
