# OpenSpec: Sparse Stoichiometry Matrix Support

**Status:** Draft
**Authors:** Lattice Microbes Development Team
**Created:** 2026-01-03
**Updated:** 2026-01-03

---

## Abstract

This specification proposes adding sparse matrix storage and algorithms for stoichiometry matrices in Lattice Microbes to improve memory efficiency and computational performance for sparse chemical reaction networks (CRNs). The current implementation uses dense matrix storage regardless of sparsity, leading to inefficient memory usage, hard size limits, and wasted computation. This proposal introduces Compressed Sparse Row (CSR) format as the primary sparse representation while maintaining backward compatibility with existing dense format.

---

## Motivation

### Current Problem

Lattice Microbes currently represents stoichiometry matrices as dense arrays with dimensions `numberSpecies × numberReactions`. This approach has several critical limitations:

1. **Memory Inefficiency**: For sparse CRNs (typical sparsity >95%), dense storage wastes significant memory
   - Example: 500 species × 200 reactions = 100,000 entries
   - With 98% sparsity: 2,000 non-zero values stored as 100,000 integers (50× overhead)

2. **Hard Size Constraints**: GPU constant memory limits impose hard caps
   - Current limit: `MPD_MAX_S_MATRIX_ENTRIES = 16,384`
   - Restricts to ~128 species × 128 reactions for dense storage
   - Sparse CRNs could theoretically support 10,000+ species with same memory

3. **Computational Waste**:
   - CUDA kernels iterate over all species including zeros
   - Memory bandwidth wasted transferring zeros to GPU
   - Cache pollution from unused data

4. **Redundant Processing**: CME solver already extracts sparse representation internally
   - Code at `src/cme/CMESolver.cpp:645-663` builds sparse arrays from dense matrix
   - This conversion happens at runtime for every simulation

### Use Cases

This optimization particularly benefits:

- **Genome-scale metabolic networks**: 1000+ species, 2000+ reactions, highly sparse
- **Gene regulatory networks**: Many species, few reactions per species
- **Spatial models (RDME)**: Sparse reactions repeated across many lattice sites
- **High-throughput simulations**: Memory/bandwidth constraints amplified

### Evidence from Codebase

Analysis of current implementation reveals:

1. **Dense-only storage** across all components:
   - Protobuf: `repeated int32 stoichiometric_matrix` (line 56, ReactionModel.proto)
   - CPU: `int * S` dense arrays (CMESolver.h:277)
   - GPU: Dense constant or global memory (constant.cu:82)

2. **Existing sparse awareness**:
   - CME solver builds `dependentSpecies[]` and `dependentSpeciesChange[]` arrays
   - These are essentially CSR format but extracted from dense at runtime
   - Dependency calculations already skip zero entries

3. **Size limit enforcement**:
   - Multiple solvers check `numberSpecies * numberReactions > MPD_MAX_S_MATRIX_ENTRIES`
   - Hard exceptions thrown (MpdRdmeSolver.cu:255, IntMpdRdmeSolver.cu:190, etc.)

---

## Proposal

Add first-class support for sparse stoichiometry matrices using **Compressed Sparse Row (CSR)** format as the canonical sparse representation, with automatic fallback to dense format for compatibility and performance when appropriate.

### Goals

1. **Primary**: Reduce memory footprint for sparse CRNs by 10-50×
2. **Primary**: Remove or significantly increase current size limits
3. **Primary**: Reduce GPU memory bandwidth requirements
4. **Secondary**: Improve computational performance through reduced iteration
5. **Secondary**: Simplify CME solver code by using sparse format directly
6. **Non-goal**: Modify simulation algorithms or numerical methods
7. **Non-goal**: Break backward compatibility with existing files

### Design Principles

1. **Backward Compatibility**: Existing dense-format files continue to work
2. **Performance**: No regression for dense or moderately sparse matrices
3. **Transparency**: Format selection automatic based on sparsity
4. **Simplicity**: Minimize code complexity, leverage existing patterns
5. **Incremental**: Phased rollout allows validation at each stage

---

## Specification

### 1. Sparse Matrix Format: CSR (Compressed Sparse Row)

#### 1.1 Data Structure

For a stoichiometry matrix S with dimensions `numberSpecies × numberReactions`, the CSR representation consists of three arrays:

```cpp
uint32_t rowPtr[numberReactions + 1];  // Row offset pointers
uint32_t colIdx[nnz];                   // Column indices (species)
int32_t  values[nnz];                   // Stoichiometric coefficients
```

Where `nnz` is the number of non-zero entries.

#### 1.2 Semantics

- **rowPtr[i]**: Index into `colIdx` and `values` where reaction `i` begins
- **rowPtr[i+1] - rowPtr[i]**: Number of non-zero entries for reaction `i`
- **colIdx[rowPtr[i]...rowPtr[i+1]-1]**: Species indices affected by reaction `i`
- **values[rowPtr[i]...rowPtr[i+1]-1]**: Stoichiometric changes for those species

#### 1.3 Example

Dense matrix (3 species × 2 reactions):
```
     Rxn0  Rxn1
S0:  [ -1,   0 ]
S1:  [  1,  -2 ]
S2:  [  0,   1 ]
```

CSR representation:
```cpp
rowPtr = [0, 2, 4]        // Rxn0: entries [0,2), Rxn1: entries [2,4)
colIdx = [0, 1, 1, 2]     // Rxn0 affects S0,S1; Rxn1 affects S1,S2
values = [-1, 1, -2, 1]   // Stoichiometric coefficients
```

#### 1.4 Properties

- **Row-major**: Natural for reaction-centric access patterns
- **Sorted columns**: Within each row, column indices in ascending order
- **No explicit zeros**: Only non-zero entries stored
- **Constant-time row access**: O(1) to locate reaction's data
- **Variable row length**: Reactions with different numbers of species

### 2. Protobuf Schema Extension

#### 2.1 Modified ReactionModel Message

```protobuf
message ReactionModel {
  required uint32 number_species                = 1;
  required uint32 number_reactions              = 2;
  repeated uint32 initial_species_count         = 3 [packed=true];

  message Reaction {
    required uint32 type                        = 1;
    repeated double rate_constant               = 2 [packed=true];
    optional bool rate_has_noise                = 3 [default = false];
    optional double rate_noise_variance         = 4;
    optional double rate_noise_tau              = 5;
  }

  repeated Reaction reaction                    = 4;

  // Legacy dense format (maintained for backward compatibility)
  repeated int32  stoichiometric_matrix         = 5 [packed=true];
  repeated uint32 dependency_matrix             = 6 [packed=true];

  // Sparse CSR format (optional, mutually exclusive with dense)
  optional bool   use_sparse_stoichiometry      = 10 [default=false];
  repeated uint32 stoichiometry_row_ptr         = 11 [packed=true];  // length: numberReactions + 1
  repeated uint32 stoichiometry_col_idx         = 12 [packed=true];  // length: nnz
  repeated int32  stoichiometry_values          = 13 [packed=true];  // length: nnz

  // Sparse dependency matrix (future extension)
  optional bool   use_sparse_dependency         = 14 [default=false];
  repeated uint32 dependency_row_ptr            = 15 [packed=true];
  repeated uint32 dependency_col_idx            = 16 [packed=true];
  repeated uint32 dependency_values             = 17 [packed=true];
}
```

#### 2.2 Format Selection Rules

1. If `use_sparse_stoichiometry == true`:
   - Use CSR fields (11-13)
   - Dense field (5) MAY be empty or MAY contain redundant data for compatibility

2. If `use_sparse_stoichiometry == false` or not set:
   - Use dense field (5)
   - CSR fields (11-13) MUST be empty

3. Both formats present with `use_sparse_stoichiometry == true`:
   - CSR format takes precedence
   - Dense format ignored

#### 2.3 Validation Rules

Implementations MUST validate:
- `stoichiometry_row_ptr.size() == numberReactions + 1`
- `stoichiometry_col_idx.size() == stoichiometry_values.size()`
- `rowPtr[0] == 0`
- `rowPtr[numberReactions] == nnz`
- `rowPtr[i] <= rowPtr[i+1]` for all i (monotonic)
- `colIdx[j] < numberSpecies` for all j (valid species indices)
- Column indices sorted within each row (optional but recommended)

### 3. API Changes

#### 3.1 Internal Representation (C++)

```cpp
class StoichiometryMatrix {
public:
    enum class Format { DENSE, SPARSE_CSR };

    // Constructors
    static StoichiometryMatrix fromDense(const int* S, uint nSpecies, uint nReactions);
    static StoichiometryMatrix fromCSR(const uint* rowPtr, const uint* colIdx,
                                       const int* values, uint nSpecies, uint nReactions);

    // Accessors
    Format getFormat() const;
    int getCoefficient(uint species, uint reaction) const;

    // CSR access (when format == SPARSE_CSR)
    const uint* getRowPtr() const;
    const uint* getColIdx() const;
    const int* getValues() const;
    uint getNonZeros() const;

    // Dense access (when format == DENSE or for compatibility)
    const int* getDenseData() const;  // May trigger conversion

    // Sparsity analysis
    double getSparsity() const;
    bool isSparse(double threshold = 0.5) const;

private:
    Format format_;
    uint numberSpecies_;
    uint numberReactions_;

    // Storage (only one allocated based on format_)
    int* denseData_;
    struct {
        uint* rowPtr;
        uint* colIdx;
        int* values;
        uint nnz;
    } sparseData_;
};
```

#### 3.2 Python Interface (pyLM)

```python
class ReactionModel:
    def setStoichiometricMatrix(self, S, force_format=None):
        """
        Set stoichiometric matrix.

        Parameters
        ----------
        S : array-like or scipy.sparse matrix
            Stoichiometric matrix (numberSpecies × numberReactions)
        force_format : {'auto', 'dense', 'sparse'}, optional
            Force specific format. Default 'auto' chooses based on sparsity.
        """
        if scipy.sparse.issparse(S):
            # Convert to CSR
            S_csr = scipy.sparse.csr_matrix(S)
            self._setSparseCSR(S_csr.indptr, S_csr.indices, S_csr.data)
        elif force_format == 'sparse' or (force_format == 'auto' and self._is_sparse(S)):
            # Convert dense numpy to sparse
            S_csr = scipy.sparse.csr_matrix(S)
            self._setSparseCSR(S_csr.indptr, S_csr.indices, S_csr.data)
        else:
            # Use dense format
            self._setDense(S.flatten())

    def getStoichiometricMatrix(self, format='auto'):
        """
        Get stoichiometric matrix.

        Parameters
        ----------
        format : {'auto', 'dense', 'sparse'}, optional
            Return format. 'auto' returns in stored format.

        Returns
        -------
        S : numpy.ndarray or scipy.sparse.csr_matrix
        """
        pass
```

### 4. Implementation Requirements

#### 4.1 SBML Import (`lm_sbml_import.cpp`)

MUST support direct construction of CSR format during import:

```cpp
void importSBMLModel(Model* sbmlModel, ReactionModel* lmModel) {
    // Determine if CSR is beneficial
    bool useSparse = estimateSparsity(sbmlModel) > 0.5;

    if (useSparse) {
        // Build CSR directly
        vector<uint> rowPtr(numberReactions + 1, 0);
        vector<uint> colIdx;
        vector<int> values;

        for (uint rxn = 0; rxn < numberReactions; rxn++) {
            rowPtr[rxn] = colIdx.size();

            // Process reactants and products
            for each non-zero entry (species, stoich) in reaction rxn {
                colIdx.push_back(species);
                values.push_back(stoich);
            }
        }
        rowPtr[numberReactions] = colIdx.size();

        // Store in protobuf
        lmModel->set_use_sparse_stoichiometry(true);
        for (auto v : rowPtr) lmModel->add_stoichiometry_row_ptr(v);
        for (auto v : colIdx) lmModel->add_stoichiometry_col_idx(v);
        for (auto v : values) lmModel->add_stoichiometry_values(v);
    } else {
        // Build dense format (existing code)
        // ...
    }
}
```

#### 4.2 CME Solver (`CMESolver.cpp`)

MUST read CSR format directly, eliminating runtime conversion:

```cpp
void CMESolver::buildDependencyTables(const ReactionModel& model) {
    if (model.use_sparse_stoichiometry()) {
        // Direct CSR read
        const uint* rowPtr = model.stoichiometry_row_ptr().data();
        const uint* colIdx = model.stoichiometry_col_idx().data();
        const int* values = model.stoichiometry_values().data();

        for (uint rxn = 0; rxn < numberReactions; rxn++) {
            uint start = rowPtr[rxn];
            uint end = rowPtr[rxn + 1];
            uint nnz = end - start;

            numberDependentSpecies[rxn] = nnz;
            dependentSpecies[rxn] = new uint[nnz];
            dependentSpeciesChange[rxn] = new int[nnz];

            for (uint i = 0; i < nnz; i++) {
                dependentSpecies[rxn][i] = colIdx[start + i];
                dependentSpeciesChange[rxn][i] = values[start + i];
            }
        }
    } else {
        // Legacy dense format (existing code)
        // ...
    }
}
```

#### 4.3 RDME/MPD GPU Solver

MUST support sparse GPU kernels:

**Host-side** (`MpdRdmeSolver.cu`):
```cpp
void MpdRdmeSolver::copyReactionDataToGPU(const ReactionModel& model) {
    if (model.use_sparse_stoichiometry()) {
        const uint* rowPtr = model.stoichiometry_row_ptr().data();
        const uint* colIdx = model.stoichiometry_col_idx().data();
        const int* values = model.stoichiometry_values().data();
        uint nnz = model.stoichiometry_values_size();

        // Allocate GPU memory
        CUDA_EXCEPTION_CHECK(cudaMalloc(&d_rowPtr, (numberReactions+1) * sizeof(uint)));
        CUDA_EXCEPTION_CHECK(cudaMalloc(&d_colIdx, nnz * sizeof(uint)));
        CUDA_EXCEPTION_CHECK(cudaMalloc(&d_values, nnz * sizeof(int8_t)));

        // Copy to device
        CUDA_EXCEPTION_CHECK(cudaMemcpy(d_rowPtr, rowPtr, ...));
        CUDA_EXCEPTION_CHECK(cudaMemcpy(d_colIdx, colIdx, ...));
        CUDA_EXCEPTION_CHECK(cudaMemcpy(d_values, values, ...));
    } else {
        // Legacy dense copy
    }
}
```

**Device-side** (`byte_reaction_dev.cu`):
```cpp
inline __device__ void evaluateReactionSparse(
    uint reactionIndex,
    uint8_t* particles,
    const uint* __restrict__ rowPtr,
    const uint* __restrict__ colIdx,
    const int8_t* __restrict__ values)
{
    uint rowStart = rowPtr[reactionIndex];
    uint rowEnd = rowPtr[reactionIndex + 1];

    // Process only non-zero entries
    for (uint idx = rowStart; idx < rowEnd; idx++) {
        uint species = colIdx[idx];
        int stoich = values[idx];

        if (stoich < 0) {
            // Remove reactants
            for (int i = 0; i < -stoich; i++) {
                removeParticle(particles, species);
            }
        } else if (stoich > 0) {
            // Add products
            for (int i = 0; i < stoich; i++) {
                addParticle(particles, species);
            }
        }
    }
}
```

#### 4.4 HDF5 I/O (`SimulationFile.cpp`)

MUST support reading/writing sparse format:

```cpp
void SimulationFile::readReactionModel(ReactionModel* model) {
    // Check for sparse format
    if (H5LTfind_dataset(file, "/Model/Reaction/StoichiometryRowPtr")) {
        // Read sparse CSR
        model->set_use_sparse_stoichiometry(true);

        // Read rowPtr
        hsize_t dims[1];
        H5LTget_dataset_info(file, "/Model/Reaction/StoichiometryRowPtr", dims, ...);
        uint* rowPtr = new uint[dims[0]];
        H5LTread_dataset(file, "/Model/Reaction/StoichiometryRowPtr", H5T_NATIVE_UINT, rowPtr);
        for (uint i = 0; i < dims[0]; i++) model->add_stoichiometry_row_ptr(rowPtr[i]);
        delete[] rowPtr;

        // Read colIdx and values similarly
        // ...
    } else {
        // Read legacy dense format
        // ...
    }
}

void SimulationFile::writeReactionModel(const ReactionModel& model) {
    if (model.use_sparse_stoichiometry()) {
        // Write CSR arrays
        hsize_t dims[1];

        dims[0] = model.stoichiometry_row_ptr_size();
        H5LTmake_dataset(file, "/Model/Reaction/StoichiometryRowPtr", 1, dims,
                        H5T_STD_U32LE, model.stoichiometry_row_ptr().data());

        dims[0] = model.stoichiometry_col_idx_size();
        H5LTmake_dataset(file, "/Model/Reaction/StoichiometryColIdx", 1, dims,
                        H5T_STD_U32LE, model.stoichiometry_col_idx().data());

        dims[0] = model.stoichiometry_values_size();
        H5LTmake_dataset(file, "/Model/Reaction/StoichiometryValues", 1, dims,
                        H5T_STD_I32LE, model.stoichiometry_values().data());
    } else {
        // Write dense format
        // ...
    }
}
```

---

## Rationale and Design Decisions

### Why CSR Format?

**Alternatives Considered:**

1. **COO (Coordinate Format)**:
   - Simpler structure: `(row, col, value)` triplets
   - ❌ Requires binary search or sorting to access reaction data
   - ❌ No O(1) row access
   - ✅ Easier construction

2. **CSC (Compressed Sparse Column)**:
   - Column-major layout
   - ❌ Inefficient for reaction-centric access (main use case)
   - ❌ Would require transposition for GPU kernels
   - ✅ Better for species-centric queries (rare)

3. **Dictionary of Keys (DOK)**:
   - Hash table: `map<(row,col), value>`
   - ❌ Non-deterministic memory layout
   - ❌ Poor cache locality
   - ❌ Not suitable for GPU transfer

4. **Hybrid (different format per solver)**:
   - CSR for CME, dense for GPU, etc.
   - ❌ Complex conversion logic
   - ❌ Multiple copies in memory
   - ❌ Harder to maintain

**CSR Advantages:**
- ✅ O(1) access to reaction data (primary access pattern)
- ✅ Cache-friendly row iteration
- ✅ Standard format with extensive literature/tooling
- ✅ Already used implicitly in CME solver
- ✅ GPU-friendly with coalesced memory access
- ✅ Minimal storage overhead (only 2 integers per reaction)

### Format Selection Strategy

**Automatic Selection**: Use sparse when `sparsity > 50%`

Rationale:
- Below 50% sparsity: Dense is competitive in memory and faster to access
- Above 50% sparsity: Sparse wins significantly in both dimensions
- Avoids user burden of format selection
- Can be overridden via API if needed

**Sparsity Calculation**:
```cpp
double sparsity = 1.0 - (double)nnz / (numberSpecies * numberReactions);
```

### Backward Compatibility Strategy

**Two-format Support**:
1. Dense format remains in protobuf schema (field 5)
2. Sparse format added as optional fields (10-13)
3. Flag `use_sparse_stoichiometry` selects interpretation

**Migration Path**:
- Old files (dense only) → Read as dense, works unchanged
- New files (sparse only) → Sparse-aware code uses directly
- New files (both formats) → Sparse takes precedence
- Conversion tool: `lm_convert --to-sparse model.lm` (future work)

**Deprecation Timeline**:
- v3.0: Sparse support added, both formats supported
- v4.0: Sparse becomes default for sparse models
- v5.0: Dense-only deprecated but still readable
- v6.0+: Dense format may be removed (distant future)

### GPU Memory Layout

**Challenge**: CUDA constant memory limit (64KB) vs sparse data structures

**Solutions**:

1. **Small sparse models** (nnz < 16,384):
   - Use constant memory for CSR arrays
   - Fast access via cache

2. **Medium sparse models** (16,384 < nnz < 1M):
   - Use global memory with `__ldg()` (read-only cache)
   - Still much smaller than dense equivalent

3. **Large sparse models** (nnz > 1M):
   - Global memory with texture cache
   - May require multi-GPU for large-scale simulations

**Current Limit**: Dense supports up to 16,384 entries
**Sparse Improvement**: Can support 100,000+ non-zeros with global memory

---

## Implementation Phases

### Phase 0: Preparation (Week 1)
- [ ] Finalize specification review
- [ ] Set up feature branch
- [ ] Create validation test suite with known sparse CRNs

### Phase 1: Core Data Structures (Week 1-2)
- [ ] Extend protobuf schema
- [ ] Implement `StoichiometryMatrix` C++ class
- [ ] Add CSR validation functions
- [ ] Unit tests for CSR construction and access

### Phase 2: SBML Import (Week 2)
- [ ] Modify `lm_sbml_import.cpp` to build CSR
- [ ] Add sparsity estimation
- [ ] Add format selection logic
- [ ] Integration tests with SBML models

### Phase 3: CME Solver (Week 2-3)
- [ ] Update `CMESolver.cpp` to read CSR directly
- [ ] Remove dense-to-sparse conversion code
- [ ] Validation tests (results match dense)
- [ ] Performance benchmarks

### Phase 4: RDME/MPD GPU Solver (Week 3-4)
- [ ] Implement sparse GPU data transfer
- [ ] Write sparse CUDA kernels
- [ ] Update all MPD solver variants
- [ ] GPU validation tests
- [ ] Performance benchmarks on sparse models

### Phase 5: I/O Layer (Week 4)
- [ ] Update HDF5 read/write for sparse format
- [ ] Update Python interface (pyLM)
- [ ] Update Java interface (jLM)
- [ ] File format validation tests

### Phase 6: Testing & Optimization (Week 5)
- [ ] End-to-end integration tests
- [ ] Performance regression tests
- [ ] Memory profiling
- [ ] Documentation updates

### Phase 7: Release (Week 6)
- [ ] User documentation
- [ ] Migration guide
- [ ] Release notes
- [ ] Benchmarking report

---

## Performance Expectations

### Memory Reduction

| Model Type | Species | Reactions | Sparsity | Dense Size | Sparse Size | Reduction |
|------------|---------|-----------|----------|------------|-------------|-----------|
| Small gene network | 50 | 100 | 96% | 20 KB | 1.6 KB | 12.5× |
| Metabolic pathway | 200 | 300 | 98% | 240 KB | 9.6 KB | 25× |
| Genome-scale | 1000 | 2000 | 99% | 8 MB | 160 KB | 50× |

### Computational Improvement

**CME Solver**:
- Eliminate runtime sparse extraction: ~5-10% speedup
- Simpler code path: Better compiler optimization potential

**GPU Kernels**:
- Dense: Iterate all species (e.g., 500 iterations)
- Sparse: Iterate non-zeros only (e.g., 10 iterations)
- Theoretical speedup: 50× for highly sparse
- Realistic speedup: 5-20× (memory bandwidth bound)

**Memory Bandwidth**:
- Dense 500×200 model: 400 KB transfer to GPU
- Sparse (98%): 16 KB transfer
- Transfer speedup: 25×

---

## Testing Strategy

### Unit Tests

1. **CSR Construction**:
   - Build from dense matrix
   - Build from SBML
   - Edge cases: empty reactions, all-zero columns

2. **CSR Validation**:
   - Correct array sizes
   - Monotonic rowPtr
   - Valid column indices
   - Detect corrupted data

3. **Format Conversion**:
   - Dense → CSR → Dense (round-trip)
   - Scipy sparse → CSR → Scipy sparse

### Integration Tests

1. **Solver Correctness**:
   - Run identical model in dense and sparse
   - Compare trajectories (must match exactly)
   - Test all solver types (CME, RDME, MPI, etc.)

2. **File I/O**:
   - Write sparse, read back, verify
   - Read legacy dense files
   - Mixed format files

### Performance Tests

1. **Memory Benchmarks**:
   - Measure resident memory for various model sizes
   - Verify sparse uses less memory
   - No regressions for dense models

2. **Runtime Benchmarks**:
   - Compare CME dense vs sparse
   - Compare GPU dense vs sparse
   - Measure file I/O overhead

### Regression Tests

- Existing test suite MUST pass with no changes
- Backward compatibility with all test models
- No numerical differences in simulation results

---

## Unresolved Questions

### Question 1: Dependency Matrix Sparsification

The dependency matrix `D` is also dense and could benefit from sparsification. Should this be included in initial implementation or deferred?

**Options**:
- A: Include in Phase 1 (dual sparse matrices)
- B: Defer to Phase 2 (after S matrix proven)
- C: Never (dependency lookup patterns may not benefit)

**Recommendation**: Option B - Validate S matrix approach first, then extend to D.

### Question 2: Automatic Format Detection

Should file readers auto-detect optimal format and convert?

**Options**:
- A: Always preserve file format (no auto-conversion)
- B: Convert to sparse on read if beneficial
- C: User-controlled via API flag

**Recommendation**: Option A initially, Option C in future version.

### Question 3: Hybrid Dense/Sparse

Some reactions may be dense while others sparse. Support mixed representation?

**Example**: 5 reactions with 100 species, 4 reactions have 2 species each (sparse), 1 reaction has 50 species (dense).

**Options**:
- A: Unified sparse CSR for all
- B: Per-reaction format selection
- C: Never support hybrid

**Recommendation**: Option A - Complexity not worth marginal gains.

### Question 4: GPU Architecture Dependencies

Different GPUs have different memory hierarchies. Should format selection be GPU-aware?

**Options**:
- A: Static decision at file import
- B: Runtime selection based on detected GPU
- C: User configuration per GPU type

**Recommendation**: Option A initially, revisit if profiling shows GPU-specific bottlenecks.

---

## Future Possibilities

### 1. cuSPARSE Integration

NVIDIA's cuSPARSE library provides optimized sparse matrix operations. Future work could leverage:
- `cusparseSpMV()` for matrix-vector products
- `cusparseSpGEMM()` for matrix multiplications
- Hardware-accelerated sparse kernels

**Benefit**: Potentially faster GPU operations
**Cost**: External dependency, increased complexity

### 2. Block Sparse Format

For models with structure (e.g., repeated reaction modules), block sparse formats could be beneficial:
- BSR (Block Sparse Row)
- Block CSR with fixed block size

**Benefit**: Better cache utilization, SIMD vectorization
**Cost**: More complex, only helps structured models

### 3. Adaptive Runtime Conversion

Monitor sparse fill-in during simulation. If matrix becomes dense due to added reactions, convert to dense format:

```cpp
if (currentSparsity < 0.3 && format == SPARSE) {
    convertToDense();
}
```

**Benefit**: Optimal performance throughout simulation
**Cost**: Runtime overhead, complexity

### 4. Distributed Sparse Matrices

For MPI-based simulations, partition sparse matrices across nodes:
- Row-wise partitioning of CSR
- Sparse matrix communication patterns

**Benefit**: Larger models, better scaling
**Cost**: Communication overhead, complex implementation

### 5. Compressed Indices

For very large models, indices could be compressed:
- Delta encoding for sorted columns
- Variable-length encoding

**Benefit**: Further memory reduction
**Cost**: Encoding/decoding overhead

---

## References

### Academic Literature

1. Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*. SIAM.
2. Barrett, R. et al. (1994). *Templates for the Solution of Linear Systems*.
3. Davis, T. A. (2006). *Direct Methods for Sparse Linear Systems*. SIAM.

### Software Implementations

1. **SciPy**: `scipy.sparse.csr_matrix` - Reference implementation
2. **Eigen**: Sparse module - C++ template library
3. **cuSPARSE**: CUDA sparse matrix library
4. **Intel MKL**: Sparse BLAS routines

### Related Work

1. Gillespie, D. T. (1976). "A General Method for Numerically Simulating the Stochastic Time Evolution of Coupled Chemical Reactions". *Journal of Computational Physics*.
2. Gibson, M. A. & Bruck, J. (2000). "Efficient Exact Stochastic Simulation of Chemical Systems with Many Species and Many Channels". *Journal of Physical Chemistry A*.
3. Slepoy, A. et al. (2008). "A constant-time kinetic Monte Carlo algorithm for simulation of large biochemical reaction networks". *Journal of Chemical Physics*.

### Existing LM Documentation

- HDF5 File Format specification: `docs/HDF5FileFormat.text`
- Protobuf schema: `src/protobuf/ReactionModel.proto`
- CME algorithm: `src/cme/CMESolver.cpp`
- MPD algorithm: `src/rdme/MpdRdmeSolver.cu`

---

## Appendices

### Appendix A: CSR Format Mathematical Definition

Given a matrix S ∈ ℤ^(m×n) with nnz non-zero entries, CSR representation consists of:

- **values** ∈ ℤ^nnz: Non-zero matrix values
- **colIdx** ∈ ℕ^nnz: Column indices of non-zero values
- **rowPtr** ∈ ℕ^(m+1): Row offset pointers

Satisfying:
- rowPtr[0] = 0
- rowPtr[m] = nnz
- S[i,j] = values[k] where k ∈ [rowPtr[i], rowPtr[i+1]) and colIdx[k] = j
- S[i,j] = 0 if j ∉ {colIdx[k] : k ∈ [rowPtr[i], rowPtr[i+1])}

### Appendix B: Memory Layout Example

For the example CSR matrix:
```
rowPtr = [0, 2, 4]
colIdx = [0, 1, 1, 2]
values = [-1, 1, -2, 1]
```

Memory layout (assuming 4-byte integers):
```
Address    Array      Value
------------------------------
0x1000     rowPtr[0]  0
0x1004     rowPtr[1]  2
0x1008     rowPtr[2]  4

0x2000     colIdx[0]  0
0x2004     colIdx[1]  1
0x2008     colIdx[2]  1
0x200C     colIdx[3]  2

0x3000     values[0]  -1
0x3004     values[1]  1
0x3008     values[2]  -2
0x300C     values[3]  1
```

Total: 12 integers = 48 bytes
vs Dense: 6 integers = 24 bytes (but example is small; savings appear at scale)

### Appendix C: Sparsity Statistics from Real Models

Analysis of representative biochemical models:

| Model | Source | Species | Reactions | Non-zeros | Sparsity |
|-------|--------|---------|-----------|-----------|----------|
| Lambda phage | Literature | 12 | 15 | 40 | 77.8% |
| Lac operon | SBML BioModels | 24 | 18 | 68 | 84.3% |
| Yeast glycolysis | Genome-scale | 95 | 73 | 312 | 95.5% |
| E. coli metabolism | BiGG database | 1805 | 2583 | 8127 | 99.8% |

Most biological networks exhibit >90% sparsity.

---

## Approval and Acceptance

### Required Reviews

- [ ] Algorithm review (numerical correctness)
- [ ] Performance review (no regressions)
- [ ] API review (backward compatibility)
- [ ] Documentation review (user-facing changes)

### Acceptance Criteria

1. All existing tests pass without modification
2. Sparse format produces identical simulation results
3. Memory usage reduced by >10× for sparse models (>90% sparsity)
4. No performance regression for dense models (<50% sparsity)
5. Documentation updated for all user-facing changes

### Sign-off

- Algorithm Lead: _________________ Date: _______
- Performance Lead: _________________ Date: _______
- Release Manager: _________________ Date: _______

---

**END OF SPECIFICATION**
