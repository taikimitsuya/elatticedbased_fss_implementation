# Build Instructions

## 1. Python Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install numpy scipy pytest
```

## 2. C Extension

### Linux / WSL

```bash
gcc -O3 -march=native -ffast-math -shared -fPIC \
    -o lattice_fss/fss_fast.so lattice_fss/fss_fast.c -lm
```

### macOS

```bash
gcc -O3 -march=native -ffast-math -dynamiclib \
    -o lattice_fss/fss_fast.dylib lattice_fss/fss_fast.c -lm
```

The C extension provides 155–166× speedup for GadgetSampler
and 56–75× for KeyUpdate compared to pure Python.
If the build fails, the implementation falls back to NumPy automatically.

## 3. Run Tests

```bash
python -m pytest tests/ -v
```

## 4. Run Benchmark

```bash
cd benchmarks
python benchmark_fss.py
```
