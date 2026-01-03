# Sparse Stoichiometry Matrix Integration Plan

## Executive Summary

**Problem Confirmed**: The current implementation uses dense matrix storage for stoichiometry matrices throughout the codebase, leading to:
- Inefficient memory usage for sparse CRNs (chemical reaction networks)
- Hard limits on system size (MPD_MAX_S_MATRIX_ENTRIES = 16384)
- Unnecessary memory transfers to GPU
- Wasted computation iterating over zero entries

**Solution**: Implement sparse matrix storage using CSR (Compressed Sparse Row) format with backward compatibility.

---

## Current State Analysis

### Memory Usage Pattern
- **Storage**: Dense arrays of size `numberSpecies × numberReactions`
- **Typical Sparsity**: Most CRNs have 2-5 species per reaction (very sparse)
- **Example**: 500 species × 200 reactions = 100,000 entries (mostly zeros)
  - With sparsity ~98%, dense uses 400KB but only ~2KB is non-zero data

### Critical Locations

1. **Construction**: `src/cmd/lm_sbml_import.cpp:249-303`
   - Allocates dense array and populates from SBML
   - Currently: `int * S = new int[numberSpecies*numberReactions]`

2. **CPU Usage (CME)**: `src/cme/CMESolver.cpp:645-663`
   - **Already extracts sparse representation**:
     ```cpp
     dependentSpecies[reaction_i] = [species indices with S[i,j] ≠ 0]
     dependentSpeciesChange[reaction_i] = [S values]
     ```
   - This is essentially CSR format per reaction!

3. **GPU Usage (RDME)**: `src/rdme/dev/byte_reaction_dev.cu:184-248`
   - Loads entire dense row per reaction
   - Iterates through ALL species (including zeros)
   - **Major optimization opportunity**

4. **Storage**: `src/protobuf/ReactionModel.proto:56`
   - `repeated int32 stoichiometric_matrix = 5 [packed=true];`
   - Dense format

5. **Hard Limits**: Multiple solvers check size limits:
   - `src/rdme/MpdRdmeSolver.cu:255`
   - `src/rdme/IntMpdRdmeSolver.cu:190`
   - All throw exceptions when `numberSpecies * numberReactions > 16384`

---

## Proposed Sparse Matrix Format: CSR (Compressed Sparse Row)

### Why CSR?

1. **Efficient row access**: Each reaction (row) is accessed independently
2. **Already used internally**: CME solver's `dependentSpecies` arrays are CSR
3. **GPU-friendly**: Coalesced memory access patterns
4. **Standard format**: Well-documented and tested

### CSR Storage Structure

For a matrix S with dimensions `numberSpecies × numberReactions`:

```
rowPtr[numberReactions + 1]  // Offsets into colIdx/values arrays
colIdx[nnz]                   // Column indices (species indices)
values[nnz]                   // Stoichiometric coefficients
```

Where `nnz` = number of non-zero entries

### Example

Dense matrix (3 species × 2 reactions):
```
S = [[-1,  0],
     [ 1, -2],
     [ 0,  1]]
```

CSR representation:
```
rowPtr  = [0, 1, 3, 4]     // Reaction 0 has 1 entry, reaction 1 has 2, reaction 2 has 1
colIdx  = [0, 0, 1, 1]     // Species indices
values  = [-1, 1, -2, 1]   // Stoichiometric coefficients
```

---

## Implementation Plan

### Phase 1: Protobuf Extension (Backward Compatible)

**File**: `src/protobuf/ReactionModel.proto`

Add sparse format fields while keeping dense format:
```protobuf
message ReactionModel {
  // ... existing fields ...

  // Legacy dense format (kept for backward compatibility)
  repeated int32  stoichiometric_matrix = 5 [packed=true];

  // New sparse CSR format (optional for backward compatibility)
  optional bool use_sparse_stoichiometry = 10 [default=false];
  repeated uint32 stoichiometry_row_ptr = 11 [packed=true];    // length: numberReactions + 1
  repeated uint32 stoichiometry_col_idx = 12 [packed=true];    // length: nnz
  repeated int32  stoichiometry_values = 13 [packed=true];     // length: nnz
}
```

### Phase 2: SBML Import Sparse Conversion

**File**: `src/cmd/lm_sbml_import.cpp:249-303`

Current approach:
```cpp
int * S = new int[numberSpecies*numberReactions];
// ... populate ...
for (uint i=0; i<numberSpecies*numberReactions; i++)
    lmModel->add_stoichiometric_matrix(S[i]);
```

New sparse approach:
```cpp
// Build sparse representation during construction
vector<int> rowPtr(numberReactions + 1, 0);
vector<uint> colIdx;
vector<int> values;

for (uint rxn = 0; rxn < numberReactions; rxn++) {
    rowPtr[rxn] = colIdx.size();

    // Process reactants/products for this reaction
    // Only add non-zero entries to colIdx and values
    for (uint species = 0; species < numberSpecies; species++) {
        int stoich = /* calculate from SBML */;
        if (stoich != 0) {
            colIdx.push_back(species);
            values.push_back(stoich);
        }
    }
}
rowPtr[numberReactions] = colIdx.size();

// Add to protobuf
lmModel->set_use_sparse_stoichiometry(true);
for (auto val : rowPtr) lmModel->add_stoichiometry_row_ptr(val);
for (auto val : colIdx) lmModel->add_stoichiometry_col_idx(val);
for (auto val : values) lmModel->add_stoichiometry_values(val);
```

### Phase 3: CME Solver Update

**File**: `src/cme/CMESolver.cpp:645-663`

**Good news**: The CME solver already uses sparse representation internally!

Current code extracts sparse data from dense matrix:
```cpp
for (uint i=0; i<numberReactions; i++) {
    numberDependentSpecies[i]=0;
    for (uint j=0, index=i; j<numberSpecies; j++, index+=numberReactions)
        if (S[index] != 0)
            numberDependentSpecies[i]++;
    // ... extract non-zero entries ...
}
```

New code reads directly from CSR:
```cpp
if (use_sparse_stoichiometry) {
    // Read directly from CSR format
    for (uint i=0; i<numberReactions; i++) {
        uint nnz = rowPtr[i+1] - rowPtr[i];
        numberDependentSpecies[i] = nnz;
        dependentSpecies[i] = new uint[nnz];
        dependentSpeciesChange[i] = new int[nnz];

        for (uint j=0; j<nnz; j++) {
            dependentSpecies[i][j] = colIdx[rowPtr[i] + j];
            dependentSpeciesChange[i][j] = values[rowPtr[i] + j];
        }
    }
} else {
    // Legacy dense format code (keep for compatibility)
    // ... existing code ...
}
```

### Phase 4: RDME/MPD GPU Solver Update

**Files**:
- `src/rdme/MpdRdmeSolver.cu:254-356`
- `src/rdme/dev/byte_reaction_dev.cu:184-248`
- `src/cuda/constant.cu:82`

#### Host-side Changes (MpdRdmeSolver.cu)

Current approach copies dense matrix:
```cpp
// Transpose S for coalesced access
int8_t * tmpS = new int8_t[numberSpecies*numberReactions];
cudaMemcpy(...);
```

New sparse approach:
```cpp
if (use_sparse_stoichiometry) {
    // Copy CSR arrays to GPU
    cudaMalloc(&d_rowPtr, (numberReactions+1) * sizeof(uint32_t));
    cudaMalloc(&d_colIdx, nnz * sizeof(uint32_t));
    cudaMalloc(&d_values, nnz * sizeof(int8_t));

    cudaMemcpy(d_rowPtr, rowPtr, ...);
    cudaMemcpy(d_colIdx, colIdx, ...);
    cudaMemcpy(d_values, values, ...);
} else {
    // Legacy dense format
}
```

#### Device-side Changes (byte_reaction_dev.cu)

Current CUDA kernel:
```cpp
inline __device__ void evaluateReaction(...) {
    // Copy entire dense S matrix row for this reaction
    int8_t S[256];
    for (uint i=0; i<numberSpeciesC; i++) {
        S[i] = SG[reactionIndex*numberSpeciesC + i];  // Load ALL species
    }

    // Process all species (many are zero)
    for (uint i=0; i<numberSpeciesC; i++) {
        if (S[i] != 0) { /* ... */ }
    }
}
```

New sparse kernel:
```cpp
inline __device__ void evaluateReactionSparse(...) {
    // Get sparse row bounds
    uint rowStart = d_rowPtr[reactionIndex];
    uint rowEnd = d_rowPtr[reactionIndex + 1];
    uint nnz = rowEnd - rowStart;

    // Only load non-zero entries
    for (uint idx = rowStart; idx < rowEnd; idx++) {
        uint species = d_colIdx[idx];
        int stoich = d_values[idx];

        // Process only affected species (no wasted iterations)
        // ... reaction logic ...
    }
}
```

**Memory savings**: For typical sparse CRN (98% sparse):
- Dense: 16384 entries → **sparse: ~300 entries** (54× reduction!)

### Phase 5: HDF5 I/O Update

**File**: `src/io/SimulationFile.cpp:406-409, 458-459, 552`

Add support for reading/writing sparse format datasets:
```cpp
if (reactionModel->use_sparse_stoichiometry()) {
    // Write sparse CSR format
    H5LTmake_dataset(file, "/Model/Reaction/StoichiometryRowPtr", ...);
    H5LTmake_dataset(file, "/Model/Reaction/StoichiometryColIdx", ...);
    H5LTmake_dataset(file, "/Model/Reaction/StoichiometryValues", ...);
} else {
    // Write dense format (legacy)
    H5LTmake_dataset(file, "/Model/Reaction/StoichiometricMatrix", ...);
}
```

### Phase 6: Python/Java Interface Updates

**Files**:
- `src/pylm/pyLM/CME.py:242`
- `src/pylm/pyLM/RDME.py:662`
- `src/jlm/jLM/LmInteract.py:253`

Update Python/Java wrappers to support sparse format:
```python
def setStoichiometricMatrix(self, S):
    """
    S can be:
    - Dense numpy array (legacy)
    - Scipy sparse matrix (CSR, COO, etc.)
    """
    if scipy.sparse.issparse(S):
        # Convert to CSR and use sparse protobuf fields
        S_csr = scipy.sparse.csr_matrix(S)
        self.reactionModel.use_sparse_stoichiometry = True
        # ... populate CSR fields ...
    else:
        # Legacy dense format
        self.reactionModel.use_sparse_stoichiometry = False
        # ... existing code ...
```

---

## Benefits Analysis

### Memory Reduction

For a typical sparse CRN with 98% sparsity:
- **500 species × 200 reactions** = 100,000 entries
  - Dense: 100,000 × 4 bytes = **400 KB**
  - Sparse CSR: ~2,000 non-zeros × (4+4) bytes = **16 KB** (25× reduction)

### Computational Benefits

1. **CUDA kernel**: Only iterate over non-zero entries
   - Dense: 500 iterations per reaction
   - Sparse: ~10 iterations per reaction (50× speedup potential)

2. **Memory bandwidth**: Reduced GPU memory traffic
   - Dense: Transfer 400 KB to GPU
   - Sparse: Transfer 16 KB to GPU

3. **Size limits**: Remove hard constraint
   - Current: Maximum 16,384 entries (e.g., 128 species × 128 reactions)
   - Sparse: Limited only by actual non-zeros (could support 10,000 species × 1,000 reactions)

### Backward Compatibility

- Keep dense format in protobuf with flag `use_sparse_stoichiometry`
- Old files continue to work
- Automatic conversion utilities can be provided

---

## Implementation Checklist

### Core Implementation
- [ ] Update protobuf definition with sparse CSR fields
- [ ] Add sparse matrix build in SBML import
- [ ] Update CME solver to read CSR directly
- [ ] Add CSR to GPU copy in RDME solver
- [ ] Implement sparse CUDA kernel for reactions
- [ ] Update HDF5 I/O for sparse format
- [ ] Update Python/Java interfaces

### Testing
- [ ] Unit tests for sparse conversion
- [ ] Verify identical results (dense vs sparse)
- [ ] Performance benchmarks
- [ ] Test with highly sparse CRNs
- [ ] Backward compatibility tests

### Documentation
- [ ] Update build configuration docs
- [ ] Add sparse format specification
- [ ] Migration guide for existing models
- [ ] Performance tuning guide

### Optional Enhancements
- [ ] Auto-detect sparsity and choose format
- [ ] Hybrid dense/sparse based on reaction sparsity
- [ ] Conversion utilities (dense ↔ sparse)
- [ ] Sparsity statistics reporting

---

## Risks and Mitigation

### Risk 1: Complexity increase
**Mitigation**: Keep dual code paths clean, well-documented, use feature flags

### Risk 2: Backward compatibility issues
**Mitigation**: Comprehensive testing, keep dense format as fallback

### Risk 3: Performance regression for dense matrices
**Mitigation**: Add auto-detection, only use sparse when beneficial (>50% sparse)

### Risk 4: CUDA kernel complexity
**Mitigation**: Start with simple implementation, optimize incrementally

---

## Recommended Next Steps

1. **Prototype**: Implement Phase 1-2 (protobuf + SBML import)
2. **Validate**: Test sparse conversion preserves correctness
3. **Benchmark**: Measure memory reduction on real CRNs
4. **Implement**: Roll out Phases 3-4 (CPU and GPU solvers)
5. **Test**: Comprehensive validation and performance testing
6. **Deploy**: Release with documentation

---

## Timeline Estimate (Development Effort)

- **Phase 1-2** (Protobuf + SBML): 2-3 days
- **Phase 3** (CME Solver): 1-2 days
- **Phase 4** (GPU Solver): 3-5 days (most complex)
- **Phase 5** (HDF5 I/O): 1-2 days
- **Phase 6** (Python/Java): 1-2 days
- **Testing & Validation**: 3-4 days

**Total**: ~2-3 weeks of focused development

---

## Conclusion

The integration of sparse matrix methods is:
- **Highly beneficial** for sparse CRNs (typical case)
- **Technically feasible** (CME already uses sparse internally)
- **Backward compatible** (dual format support)
- **Performance-critical** (removes size limits, reduces memory, speeds computation)

**Recommendation**: Proceed with implementation using the phased approach outlined above.
