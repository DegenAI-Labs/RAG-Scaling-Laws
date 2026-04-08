# GPU FAISS Build Guide

This guide builds FAISS with GPU + Python bindings from source and validates the install on a GPU node.

## 1) Optional prerequisites (local user install)

Use this only if your system SWIG/Bison are too old or missing.

### Build SWIG (local)

```bash
cd "$HOME"
rm -rf swig-4.4.1 swig-4.4.1.tar.gz

# Official release tarball (not GitHub snapshot)
curl -L -o swig-4.4.1.tar.gz \
  https://downloads.sourceforge.net/project/swig/swig/swig-4.4.1/swig-4.4.1.tar.gz

tar -xzf swig-4.4.1.tar.gz
cd swig-4.4.1
./configure --prefix="$HOME/.local" --with-pcre-prefix="$HOME/.local"
make -j8
make install
```

### Build Bison (local)

```bash
cd "$HOME"
curl -L -o bison-3.8.2.tar.gz https://ftp.gnu.org/gnu/bison/bison-3.8.2.tar.gz
tar -xzf bison-3.8.2.tar.gz
cd bison-3.8.2
./configure --prefix="$HOME/.local"
make -j8
make install
```

### Make local tools take precedence

```bash
export PATH="$HOME/.local/bin:$PATH"
which swig
swig -version
swig -swiglib
which bison
bison --version
```

## 2) Clone and configure FAISS

```bash
git clone https://github.com/facebookresearch/faiss.git
cd faiss
rm -rf build
```

Configure with GPU + Python enabled:

```bash
cmake -B build . \
  -DFAISS_ENABLE_GPU=ON \
  -DFAISS_ENABLE_PYTHON=ON \
  -DBUILD_TESTING=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90
```

Notes:

- Update `-DCMAKE_CUDA_ARCHITECTURES=90` to match your GPU architecture.
- Re-run from a clean `build/` directory when switching CUDA arch.

## 3) Build and install Python bindings

```bash
cmake --build build -j8 --target faiss
cmake --build build -j8 --target swigfaiss
cd build/faiss/python
python -m pip install --no-deps --force-reinstall .
```

## 4) Validate GPU FAISS

Run this on a GPU node:

```bash
python - <<'PY'
import faiss
print("faiss", faiss.__version__)
print("gpu api", hasattr(faiss, "StandardGpuResources"), hasattr(faiss, "index_cpu_to_gpu"))
print("num gpus", faiss.get_num_gpus())
PY
```

Expected:

- GPU APIs should be `True`
- `num gpus` should be greater than `0`
- FAISS should now run _significantly_ faster (like 10 hours vs 3 minutes faster).

