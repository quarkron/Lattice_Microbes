# GPU Programming Prerequisites & Concepts Guide

**For:** Understanding Sparse Stoichiometry Matrix Implementation in Lattice Microbes
**Audience:** Someone with basic C++ knowledge, no GPU experience
**Goal:** Understand the concepts to read, modify, and optimize CUDA kernels

---

## Table of Contents

1. [Essential C++ Concepts (Review)](#1-essential-c-concepts-review)
2. [Memory & Pointers (Deep Dive)](#2-memory--pointers-deep-dive)
3. [Parallel Computing Fundamentals](#3-parallel-computing-fundamentals)
4. [GPU Architecture Basics](#4-gpu-architecture-basics)
5. [CUDA Programming Model](#5-cuda-programming-model)
6. [Memory Hierarchies in CUDA](#6-memory-hierarchies-in-cuda)
7. [Performance Concepts](#7-performance-concepts)
8. [Sparse Matrix Concepts](#8-sparse-matrix-concepts)
9. [Reading the Lattice Microbes Code](#9-reading-the-lattice-microbes-code)
10. [Learning Path & Resources](#10-learning-path--resources)

---

## 1. Essential C++ Concepts (Review)

### 1.1 Pointers and Arrays

**What you need to know:**

```cpp
// Pointer basics
int value = 42;
int* ptr = &value;        // ptr holds the ADDRESS of value
int result = *ptr;        // Dereference: get the VALUE at that address (42)

// Arrays and pointer arithmetic
int arr[5] = {10, 20, 30, 40, 50};
int* p = arr;             // Array name is pointer to first element
int first = p[0];         // Same as: *p
int second = p[1];        // Same as: *(p+1)
int third = *(p+2);       // Pointer arithmetic

// Multi-dimensional arrays (critical for matrices!)
int matrix[3][4];         // 3 rows, 4 columns
int value = matrix[1][2]; // Row 1, Column 2

// Flattened 2D array (how matrices are ACTUALLY stored)
int* flat = new int[3 * 4];  // 12 elements in 1D array
// Access [row][col] as: flat[row * num_cols + col]
int value2 = flat[1 * 4 + 2]; // Same as matrix[1][2]
```

**Why it matters:** GPU memory is all flat arrays. Matrix `S[species][reaction]` is stored as `S[species * numberReactions + reaction]`.

### 1.2 Memory Allocation

```cpp
// Stack allocation (automatic, limited size)
int stackArray[100];      // OK for small arrays
// int hugeArray[1000000]; // BAD! Stack overflow

// Heap allocation (manual, large size)
int* heapArray = new int[1000000];  // OK, but YOU must free it
delete[] heapArray;       // Don't forget!

// Memory leak example
void badFunction() {
    int* data = new int[1000];
    // ... use data ...
    // FORGOT to delete[] data; <-- MEMORY LEAK
}
```

**Why it matters:** GPUs have separate memory. You'll copy data between CPU (host) and GPU (device).

### 1.3 Const and Restrict

```cpp
// const: promise you won't modify
void processData(const int* input) {
    // input[0] = 5;  // ERROR: can't modify const
    int val = input[0];  // OK: can read
}

// __restrict__: promise no aliasing (C99/C++, GPU-specific)
void compute(int* __restrict__ output,
             const int* __restrict__ input) {
    // Compiler knows: output and input don't overlap
    // Can optimize more aggressively
}
```

**Why it matters:** GPU code uses `__restrict__` heavily for performance (see `byte_reaction_dev.cu:181`).

---

## 2. Memory & Pointers (Deep Dive)

### 2.1 Memory Layout

**How data is actually stored:**

```cpp
// 2D matrix in memory
int S[3][4] = {
    {1, 2, 3, 4},      // Row 0
    {5, 6, 7, 8},      // Row 1
    {9, 10, 11, 12}    // Row 2
};

// Actual memory layout (row-major):
// Address: 0    4    8    12   16   20   24   28   32   36   40   44
// Value:   1    2    3    4    5    6    7    8    9    10   11   12
//          |----Row 0-----|----Row 1-----|----Row 2-----|

// Column-major (Fortran, MATLAB):
// Value:   1    5    9    2    6    10   3    7    11   4    8    12
//          |Col0|Col1|Col2|Col3|
```

**Accessing elements:**

```cpp
// 2D notation: S[row][col]
int val = S[1][2];  // Row 1, Col 2 = 7

// Flattened (what's actually happening):
int* flat = (int*)S;
int val2 = flat[1 * 4 + 2];  // Same as S[1][2]
//              ^ row * num_cols + col

// In Lattice Microbes (see lm_sbml_import.cpp:275):
S[species * numberReactions + reaction]
//  ^row    ^cols per row      ^col
```

**Why it matters:** GPU memory is always flat. Understanding layout is critical for performance.

### 2.2 Cache Lines and Stride

**Sequential access (FAST):**
```cpp
// Read elements 0, 1, 2, 3, 4, 5...
for (int i = 0; i < 1000; i++) {
    sum += array[i];  // Sequential: cache-friendly
}
```

**Strided access (SLOW):**
```cpp
// Read elements 0, 10, 20, 30, 40...
for (int i = 0; i < 1000; i += 10) {
    sum += array[i];  // Strided: cache-unfriendly
}
```

**Why it matters:** GPU kernels must access memory sequentially for "coalesced" access (major performance factor).

---

## 3. Parallel Computing Fundamentals

### 3.1 Sequential vs Parallel

**Sequential (normal CPU code):**
```cpp
// One task at a time
for (int i = 0; i < 1000; i++) {
    result[i] = compute(data[i]);
}
// Time: 1000 × time_per_compute
```

**Parallel (what GPU does):**
```cpp
// 1000 tasks simultaneously!
// Thread 0 computes result[0]
// Thread 1 computes result[1]
// ...
// Thread 999 computes result[999]
// Time: 1 × time_per_compute (ideally)
```

### 3.2 Data Parallelism

**Key idea:** Same operation on different data

```cpp
// CPU: Sequential
for (int i = 0; i < N; i++) {
    output[i] = input[i] * 2;  // Same operation, different data
}

// GPU: Parallel (conceptual)
parallel_for (int i = 0; i < N; i++) {
    output[i] = input[i] * 2;  // All threads do this simultaneously
}
```

**Why it matters:** GPUs excel at data parallelism (same code, many data elements). Perfect for applying reactions across many lattice sites!

### 3.3 Race Conditions

**Problem when threads share data:**

```cpp
int counter = 0;

// Two threads running simultaneously:
// Thread A:                 Thread B:
counter = counter + 1;    counter = counter + 1;

// Expected: counter = 2
// Actual: counter = 1 (sometimes!)
// Why? Both read 0, both write 1
```

**Solution: Atomic operations**
```cpp
atomicAdd(&counter, 1);  // Hardware guarantees correct result
```

**Why it matters:** See `byte_reaction_dev.cu:239` - `atomicAdd(siteOverflowList, 1)` prevents race conditions.

---

## 4. GPU Architecture Basics

### 4.1 CPU vs GPU Philosophy

**CPU: Few powerful cores**
```
[Core 1] [Core 2] [Core 3] [Core 4]
Complex, out-of-order, speculative execution
4-16 cores typical
Good at: Sequential tasks, branching, unpredictable workloads
```

**GPU: Many simple cores**
```
[Core 1] [Core 2] ... [Core 1024] [Core 1025] ... [Core 5120]
Simple, in-order execution
1000s-10000s of cores
Good at: Parallel tasks, same operation on lots of data
```

**Analogy:**
- CPU = 4 PhD students solving hard problems
- GPU = 5000 elementary students doing arithmetic

### 4.2 NVIDIA GPU Structure

**Hierarchy:**
```
GPU (Device)
 ├─ Streaming Multiprocessor (SM) 1
 │   ├─ Core 1, Core 2, ..., Core 128
 │   └─ Shared Memory (fast, 48-96 KB)
 ├─ SM 2
 ├─ ...
 └─ SM 80 (for example)

Global Memory (slow, 8-32 GB)
```

**Execution model:**
- Each SM runs many threads
- Threads grouped into "warps" (32 threads)
- All threads in a warp execute same instruction (SIMD)

### 4.3 Warps (Critical Concept!)

**What is a warp?**
- Group of 32 threads that execute in lockstep
- Hardware executes one instruction for entire warp at once

**Example:**
```cpp
// All 32 threads execute simultaneously:
// Thread 0: result[0] = data[0] * 2;
// Thread 1: result[1] = data[1] * 2;
// ...
// Thread 31: result[31] = data[31] * 2;

// GPU executes as ONE instruction: "multiply by 2" for all 32
```

**Warp divergence (BAD):**
```cpp
if (threadID % 2 == 0) {
    // Even threads do this
    result = data * 2;
} else {
    // Odd threads do this
    result = data * 3;
}

// GPU must execute BOTH branches:
// Step 1: Even threads work, odd threads idle (50% utilization)
// Step 2: Odd threads work, even threads idle (50% utilization)
// Total: 50% efficiency
```

**Why it matters:** The dense matrix code has severe warp divergence (see performance analysis). Sparse fixes this!

---

## 5. CUDA Programming Model

### 5.1 Host vs Device

```cpp
// Host = CPU code (normal C++)
void hostFunction() {
    int* h_data = new int[1000];  // CPU memory
    // ... fill h_data ...
}

// Device = GPU code (CUDA)
__device__ void deviceFunction() {
    // Runs on GPU
}

__global__ void kernelFunction() {
    // Entry point: called from CPU, runs on GPU
}
```

### 5.2 Basic CUDA Kernel

```cpp
// GPU kernel: runs on device
__global__ void addKernel(int* output, const int* input, int N) {
    // Get thread ID
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Bounds check
    if (i < N) {
        output[i] = input[i] + 10;
    }
}

// CPU code: launches kernel
int main() {
    // 1. Allocate device memory
    int* d_input, *d_output;
    cudaMalloc(&d_input, 1000 * sizeof(int));
    cudaMalloc(&d_output, 1000 * sizeof(int));

    // 2. Copy data to GPU
    cudaMemcpy(d_input, h_input, 1000 * sizeof(int), cudaMemcpyHostToDevice);

    // 3. Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (1000 + 255) / 256;  // Round up
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, d_input, 1000);

    // 4. Copy result back
    cudaMemcpy(h_output, d_output, 1000 * sizeof(int), cudaMemcpyDeviceToHost);

    // 5. Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}
```

### 5.3 Thread Indexing

**Key concept:** Each thread needs to know "which data element am I processing?"

```cpp
__global__ void kernel(int* data) {
    // Built-in variables:
    // blockIdx.x  = which block am I in? (0, 1, 2, ...)
    // threadIdx.x = which thread within my block? (0-255 if 256 threads/block)
    // blockDim.x  = how many threads per block? (256)

    // Calculate global thread ID:
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Example:
    // Block 0, Thread 0: i = 0 + 0*256 = 0
    // Block 0, Thread 1: i = 1 + 0*256 = 1
    // Block 1, Thread 0: i = 0 + 1*256 = 256
    // Block 1, Thread 1: i = 1 + 1*256 = 257
}
```

**Visual:**
```
Grid (all blocks)
├─ Block 0: [Thread 0] [Thread 1] ... [Thread 255]  → Process elements 0-255
├─ Block 1: [Thread 0] [Thread 1] ... [Thread 255]  → Process elements 256-511
├─ Block 2: [Thread 0] [Thread 1] ... [Thread 255]  → Process elements 512-767
└─ ...
```

### 5.4 CUDA Qualifiers

```cpp
__global__ void kernelFunc() {
    // Called from: Host (CPU)
    // Executes on: Device (GPU)
    // Entry point for GPU computation
}

__device__ void helperFunc() {
    // Called from: Device (other GPU code)
    // Executes on: Device (GPU)
    // Helper function for kernels
}

__host__ void cpuFunc() {
    // Called from: Host (CPU)
    // Executes on: Host (CPU)
    // Normal C++ function (default if no qualifier)
}

__host__ __device__ void bothFunc() {
    // Can be called from either host or device
    // Compiled for both CPU and GPU
}
```

---

## 6. Memory Hierarchies in CUDA

### 6.1 Memory Types (Speed vs Size)

**From fastest to slowest:**

```
Registers (per-thread)
  ├─ Speed: 1 cycle
  ├─ Size: ~64 KB per SM (divided among threads)
  └─ Scope: Private to each thread

Shared Memory (per-block)
  ├─ Speed: ~5 cycles
  ├─ Size: 48-96 KB per SM
  └─ Scope: Shared among threads in same block

L1/L2 Cache (automatic)
  ├─ Speed: ~100 cycles (L1), ~200 cycles (L2)
  └─ Size: ~128 KB (L1), ~6 MB (L2)

Global Memory (main GPU memory)
  ├─ Speed: ~400 cycles
  ├─ Size: 8-32 GB
  └─ Scope: Entire GPU, all threads

Constant Memory (read-only)
  ├─ Speed: ~5 cycles (if cached), ~400 (if not)
  ├─ Size: 64 KB
  └─ Scope: Read-only, cached efficiently
```

### 6.2 Code Examples

```cpp
__global__ void memoryExample(int* global_data) {
    // REGISTERS (automatic for local variables)
    int local_var = 42;  // Stored in register (FAST)

    // SHARED MEMORY (declared with __shared__)
    __shared__ int shared_data[256];  // Shared by all threads in block
    shared_data[threadIdx.x] = local_var;
    __syncthreads();  // Wait for all threads to write

    // GLOBAL MEMORY (passed as pointer)
    int value = global_data[threadIdx.x];  // Read from global (SLOW)

    // CONSTANT MEMORY (declared at file scope)
    // extern __constant__ int const_data[100];
    // int c = const_data[5];  // Read from constant (FAST if cached)
}
```

### 6.3 Lattice Microbes Memory Usage

**See `src/cuda/constant.cu:82`:**
```cpp
__constant__ int8_t SC[MPD_MAX_S_MATRIX_ENTRIES];
// Stoichiometry matrix in constant memory
// Pros: Fast if cached (all threads read same data)
// Cons: Limited to 64 KB (16,384 entries with int8_t)
```

**Alternative (`MPD_GLOBAL_S_MATRIX` flag):**
```cpp
int8_t* SG;  // Global memory
cudaMalloc(&SG, size);

// In kernel:
#ifdef MPD_GLOBAL_S_MATRIX
    S[i] = SG[index];  // Global memory (slower but unlimited size)
#else
    S[i] = SC[index];  // Constant memory (faster but limited)
#endif
```

---

## 7. Performance Concepts

### 7.1 Memory Bandwidth

**Concept:** How much data can move per second

**CPU-GPU Transfer (PCIe):**
- Speed: ~10-16 GB/s
- Example: 400 KB takes ~40 microseconds
- **Lesson:** Minimize transfers, batch operations

**GPU Global Memory:**
- Speed: ~500-900 GB/s
- Example: Load 400 KB → ~0.5 microseconds
- **Lesson:** Still expensive, prefer shared memory

### 7.2 Coalesced Memory Access

**Good (coalesced):**
```cpp
// Thread 0 reads address 0
// Thread 1 reads address 4
// Thread 2 reads address 8
// ... consecutive addresses
for (int i = threadIdx.x; i < N; i += blockDim.x) {
    data[i] = ...;  // Sequential, coalesced
}

// Hardware combines into one 128-byte transaction
```

**Bad (uncoalesced):**
```cpp
// Thread 0 reads address 0
// Thread 1 reads address 1000
// Thread 2 reads address 2000
// ... scattered addresses
for (int i = 0; i < N; i++) {
    data[threadIdx.x * 1000 + i] = ...;  // Scattered, uncoalesced
}

// Hardware needs 32 separate transactions (32× slower!)
```

**In Lattice Microbes (see `MpdRdmeSolver.cu:317-356`):**
```cpp
// Transpose S for coalesced access on device
for(uint rx = 0; rx < numberReactions; rx++) {
    for (uint p = 0; p < numberSpecies; p++) {
        tmpS[rx * numberSpecies + p] = S[numberReactions * p + rx];
        //    ^column-major              ^row-major
    }
}
```

**Why?** So GPU threads access consecutive elements!

### 7.3 Occupancy

**Concept:** How many threads can actively run on SM

```
SM can run: 2048 threads maximum
Your kernel uses: 256 threads per block

If each thread uses:
- 32 registers → Can fit 8 blocks (2048 threads) ✓ 100% occupancy
- 64 registers → Can fit 4 blocks (1024 threads) ✗ 50% occupancy
- 128 registers → Can fit 2 blocks (512 threads) ✗ 25% occupancy

Higher occupancy = Better at hiding memory latency
```

**Lesson:** Don't use too many registers or shared memory per thread.

### 7.4 Latency Hiding

**Problem:** Memory access takes 400 cycles

**CPU Solution:** Stall and wait (expensive!)

**GPU Solution:** Switch to another thread
```
Cycle 0: Thread 0 requests memory → switch to Thread 1
Cycle 1: Thread 1 requests memory → switch to Thread 2
...
Cycle 400: Thread 0's data ready → resume Thread 0
```

**Lesson:** Need LOTS of threads to hide latency (thousands, not hundreds).

---

## 8. Sparse Matrix Concepts

### 8.1 Dense vs Sparse Storage

**Dense Matrix:**
```cpp
// Stoichiometry matrix: 5 species × 3 reactions
int S[5][3] = {
    {-1,  0,  0},  // Species 0
    { 1, -2,  0},  // Species 1
    { 0,  1,  0},  // Species 2
    { 0,  0, -1},  // Species 3
    { 0,  0,  2},  // Species 4
};

// Storage: 15 integers (5×3)
// Non-zeros: 6
// Sparsity: (15-6)/15 = 60%
```

**Sparse CSR (Compressed Sparse Row):**
```cpp
// Only store non-zeros and their locations
uint rowPtr[4] = {0, 2, 4, 6};     // Offsets for each reaction
uint colIdx[6] = {0, 1, 1, 2, 3, 4};  // Which species
int values[6] = {-1, 1, -2, 1, -1, 2}; // Stoichiometry values

// Storage: 3 arrays (4+6+6 = 16 integers)
// Benefit: For 98% sparse, 50× reduction!
```

### 8.2 CSR Format Details

**How to read CSR:**

```cpp
// Get non-zeros for reaction 1:
uint start = rowPtr[1];      // = 2
uint end = rowPtr[1+1];      // = 4
uint nnz = end - start;      // = 2 non-zeros

for (uint i = start; i < end; i++) {
    uint species = colIdx[i];   // i=2: species=1, i=3: species=2
    int stoich = values[i];     // i=2: stoich=-2, i=3: stoich=1
    // Reaction 1: Species 1 changes by -2, Species 2 changes by +1
}
```

**Visual representation:**
```
Reaction 0: rowPtr[0]=0 to rowPtr[1]=2
            colIdx[0]=0, values[0]=-1  → Species 0: -1
            colIdx[1]=1, values[1]=1   → Species 1: +1

Reaction 1: rowPtr[1]=2 to rowPtr[2]=4
            colIdx[2]=1, values[2]=-2  → Species 1: -2
            colIdx[3]=2, values[3]=1   → Species 2: +1

Reaction 2: rowPtr[2]=4 to rowPtr[3]=6
            colIdx[4]=3, values[4]=-1  → Species 3: -1
            colIdx[5]=4, values[5]=2   → Species 4: +2
```

### 8.3 Why CSR for GPUs?

**Row-wise access pattern:**
```cpp
// Each GPU thread processes one reaction
__global__ void processReactions() {
    int rxn = threadIdx.x;

    // Access all species for this reaction
    uint start = rowPtr[rxn];
    uint end = rowPtr[rxn+1];

    for (uint i = start; i < end; i++) {
        // Process species colIdx[i] with stoich values[i]
    }
}
```

**Benefits:**
- ✓ Consecutive threads access consecutive rowPtr entries (coalesced!)
- ✓ Each thread knows its data bounds instantly
- ✓ No wasted iterations over zeros

---

## 9. Reading the Lattice Microbes Code

Now you can understand the actual code!

### 9.1 Understanding `byte_reaction_dev.cu:181-258`

```cpp
inline __device__ void evaluateReaction(
    const unsigned int latticeIndex,     // Which site in lattice
    const uint8_t siteType,              // Type of site
    uint8_t * __restrict__ particles,    // Particles at this site
    const unsigned int reactionIndex,    // Which reaction to evaluate
    unsigned int * siteOverflowList)     // Overflow tracking
{
    // STEP 1: Load dense S matrix row for this reaction
    int8_t S[256];  // Local array (registers or stack)
    for (uint i=0, index=reactionIndex*numberSpeciesC;
         i<numberSpeciesC; i++, index++) {
        //      ^reaction     ^species     ^iterate all species

#ifdef MPD_GLOBAL_S_MATRIX
        S[i] = SG[index];  // Load from global memory
#else
        S[i] = SC[index];  // Load from constant memory
#endif
    }
    // Problem: Loads ALL 500 species even if reaction only affects 5!

    // STEP 2: Remove reactants (negative stoichiometry)
    int nextParticle = 0;
    for (uint i=0; i<MPD_PARTICLES_PER_SITE; i++) {
        uint8_t particle = particles[i];
        if (particle > 0) {
            if (S[particle-1] >= 0) {
                particles[nextParticle++] = particle;  // Keep it
            } else {
                S[particle-1]++;  // Mark as consumed
            }
        }
    }

    // STEP 3: Add products (positive stoichiometry)
    for (uint i=0; i<numberSpeciesC; i++) {  // Iterate ALL species
        for (uint j=0; j<S[i]; j++) {        // Most S[i]=0, skip
            if (nextParticle < MPD_PARTICLES_PER_SITE) {
                particles[nextParticle++] = i+1;  // Add particle
            } else {
                // Overflow handling
                int exceptionIndex = atomicAdd(siteOverflowList, 1);
                if (exceptionIndex < TUNE_MPD_MAX_PARTICLE_OVERFLOWS) {
                    siteOverflowList[(exceptionIndex*2)+1] = latticeIndex;
                    siteOverflowList[(exceptionIndex*2)+2] = i+1;
                }
            }
        }
    }
    // Problem: Iterates all 500 species, but only 5 have S[i]>0
}
```

**Inefficiencies identified:**
1. Line 191: Loads all species (dense row)
2. Line 226: Iterates all species (even zeros)
3. Warp divergence: Different threads have different S[i] values

### 9.2 Sparse Version (Conceptual)

```cpp
__device__ void evaluateReactionSparse(
    uint8_t * __restrict__ particles,
    const unsigned int reactionIndex,
    const uint* __restrict__ rowPtr,     // CSR format
    const uint* __restrict__ colIdx,     // CSR format
    const int8_t* __restrict__ values)   // CSR format
{
    // STEP 1: Get sparse row bounds
    uint start = rowPtr[reactionIndex];
    uint end = rowPtr[reactionIndex + 1];
    uint nnz = end - start;
    // Only load 5 entries instead of 500!

    // STEP 2: Process only affected species
    for (uint idx = start; idx < end; idx++) {
        uint species = colIdx[idx];      // Which species
        int stoich = values[idx];        // Stoichiometric coefficient

        if (stoich < 0) {
            // Remove reactants (only these species, not all 500)
            for (int i = 0; i < -stoich; i++) {
                removeParticle(particles, species);
            }
        } else if (stoich > 0) {
            // Add products (only these species)
            for (int i = 0; i < stoich; i++) {
                addParticle(particles, species);
            }
        }
    }
    // Much faster: Only 5 iterations instead of 500!
}
```

**Benefits:**
- Memory: Load 5 values instead of 500 (100× less)
- Computation: 5 iterations instead of 500 (100× fewer)
- Warp efficiency: All threads do similar work (minimal divergence)

### 9.3 Understanding `CMESolver.cpp:645-663`

```cpp
// Create the species dependency tables from the S matrix.
for (uint i=0; i<numberReactions; i++) {
    // Count non-zeros for reaction i
    numberDependentSpecies[i] = 0;
    for (uint j=0, index=i; j<numberSpecies; j++, index+=numberReactions) {
        //           ^start at reaction i, stride by numberReactions
        if (S[index] != 0)
            numberDependentSpecies[i]++;  // Found a non-zero
    }

    // Allocate arrays for this reaction's dependencies
    dependentSpecies[i] = new uint[numberDependentSpecies[i]];
    dependentSpeciesChange[i] = new int[numberDependentSpecies[i]];

    // Extract non-zero species and values
    for (uint j=0, index=i, k=0; j<numberSpecies; j++, index+=numberReactions) {
        if (S[index] != 0 && k < numberDependentSpecies[i]) {
            dependentSpecies[i][k] = j;        // Species index (like colIdx)
            dependentSpeciesChange[i][k] = S[index];  // Value (like values)
            k++;
        }
    }
}
```

**This is building CSR format from dense!**
- `dependentSpecies[i]` = colIdx for reaction i
- `dependentSpeciesChange[i]` = values for reaction i
- `numberDependentSpecies[i]` = nnz for reaction i

**With sparse input, this entire loop goes away:**
```cpp
// Just read CSR directly:
for (uint i=0; i<numberReactions; i++) {
    uint start = rowPtr[i];
    uint end = rowPtr[i+1];
    numberDependentSpecies[i] = end - start;
    dependentSpecies[i] = &colIdx[start];         // Point to existing data
    dependentSpeciesChange[i] = &values[start];   // No copy needed!
}
```

---

## 10. Learning Path & Resources

### Phase 1: C++ Fundamentals (1-2 weeks if rusty)

**Topics:**
- Pointers and references
- Dynamic memory allocation
- Arrays and multi-dimensional arrays
- Const correctness

**Resources:**
- [LearnCpp.com](https://www.learncpp.com/) - Chapters 9-12 (pointers, arrays)
- Practice: Write a matrix multiplication in C++ with flat arrays

**Checkpoint:** Can you write this?
```cpp
void matrixMultiply(const int* A, const int* B, int* C,
                   int M, int K, int N) {
    // A: M×K, B: K×N, C: M×N
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}
```

### Phase 2: Parallel Computing Concepts (1 week)

**Topics:**
- Data parallelism
- SIMD vs MIMD
- Race conditions
- Memory hierarchies

**Resources:**
- "Introduction to Parallel Computing" (Blaise Barney, LLNL)
  https://hpc.llnl.gov/documentation/tutorials/introduction-parallel-computing-tutorial
- Video: "Thinking Parallel" series (NVIDIA, YouTube)

**Checkpoint:** Understand why this is bad:
```cpp
// Parallel code (conceptual)
parallel_for (int i = 0; i < 1000; i++) {
    counter++;  // RACE CONDITION!
}
```

### Phase 3: GPU Architecture (1 week)

**Topics:**
- CPU vs GPU design philosophy
- Streaming multiprocessors
- Warps and SIMT execution
- Memory hierarchy

**Resources:**
- "CUDA C Programming Guide" - Chapters 1-2 (NVIDIA docs)
- Video: "How GPU Computing Works" (Mythbusters' Brian Krzanich)
- Blog: "GPU Architecture Overview" (fgiesen.wordpress.com)

**Checkpoint:** Can you explain warp divergence to someone?

### Phase 4: Basic CUDA Programming (2-3 weeks)

**Topics:**
- Kernel syntax
- Thread indexing
- Memory allocation and transfers
- Basic optimization

**Resources:**
- "CUDA by Example" book (Jason Sanders) - Chapters 1-6
- NVIDIA CUDA Training Series (free online)
  https://www.nvidia.com/en-us/on-demand/session/gtcspring21-cwes1083/
- Coursera: "Introduction to Parallel Programming" (Udacity/NVIDIA)

**Hands-on:**
```cpp
// Exercise 1: Vector addition
__global__ void vectorAdd(int* a, int* b, int* c, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) c[i] = a[i] + b[i];
}

// Exercise 2: Matrix transpose
__global__ void transpose(int* in, int* out, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        out[col * M + row] = in[row * N + col];
    }
}
```

**Checkpoint:** Write and run a CUDA vector addition program.

### Phase 5: Memory Optimization (1-2 weeks)

**Topics:**
- Coalesced memory access
- Shared memory usage
- Constant memory
- Memory bandwidth analysis

**Resources:**
- "CUDA C Programming Guide" - Chapter 5 (Performance Guidelines)
- "CUDA Best Practices Guide" (NVIDIA)
- Blog: "CUDA Pro Tip" series (NVIDIA Developer Blog)

**Hands-on:**
```cpp
// Exercise: Matrix multiply with shared memory
__global__ void matMulShared(float* A, float* B, float* C, int N) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Load tile into shared memory
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0;
    for (int tile = 0; tile < N/BLOCK_SIZE; tile++) {
        As[threadIdx.y][threadIdx.x] = A[row * N + tile * BLOCK_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(tile * BLOCK_SIZE + threadIdx.y) * N + col];
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * N + col] = sum;
}
```

### Phase 6: Sparse Matrix Algorithms (1 week)

**Topics:**
- CSR, COO, CSC formats
- Sparse matrix-vector multiply
- SpMV performance considerations

**Resources:**
- "Sparse Matrix Computations" (lecture notes, Berkeley CS267)
- cuSPARSE documentation (NVIDIA)
- Paper: "Implementing Sparse Matrix-Vector Multiplication on GPUs" (Bell & Garland)

**Hands-on:**
```cpp
// Exercise: SpMV in CSR format
__global__ void spMV_CSR(const int* rowPtr, const int* colIdx,
                         const float* values, const float* x,
                         float* y, int numRows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        float sum = 0;
        int start = rowPtr[row];
        int end = rowPtr[row + 1];
        for (int i = start; i < end; i++) {
            sum += values[i] * x[colIdx[i]];
        }
        y[row] = sum;
    }
}
```

### Phase 7: Lattice Microbes Codebase (Ongoing)

**Study these files in order:**

1. `src/protobuf/ReactionModel.proto` - Data structures
2. `src/cmd/lm_sbml_import.cpp:249-303` - How S matrix is built
3. `src/cme/CMESolver.h:235-240` - Simple usage
4. `src/cme/CMESolver.cpp:645-663` - Dependency extraction
5. `src/rdme/MpdRdmeSolver.cu:254-356` - GPU memory transfer
6. `src/cuda/constant.cu:82` - Constant memory declaration
7. `src/rdme/dev/byte_reaction_dev.cu:181-258` - GPU kernel (most complex)

**Practice:**
- Trace through `evaluateReaction` with a debugger
- Modify kernel to print S matrix values
- Profile GPU kernel with `nvprof` or NSight

---

## Quick Reference: Key Concepts Summary

### Essential C++
- **Pointers:** `int* p = &value;` and `*p` (dereference)
- **Arrays:** `arr[i]` same as `*(arr + i)`
- **2D flatten:** `matrix[row][col]` = `flat[row * cols + col]`

### CUDA Basics
- **Kernel:** `__global__ void func()` - runs on GPU
- **Thread ID:** `threadIdx.x + blockIdx.x * blockDim.x`
- **Memory copy:** `cudaMemcpy(dst, src, size, direction)`

### Memory Hierarchy
- **Registers:** Fastest (1 cycle), automatic
- **Shared:** Fast (5 cycles), `__shared__` keyword
- **Global:** Slow (400 cycles), `cudaMalloc`
- **Constant:** Fast if cached, `__constant__`, 64KB limit

### Performance
- **Coalesced access:** Sequential memory reads (fast)
- **Warp divergence:** Threads take different paths (slow)
- **Occupancy:** More threads = hide latency better

### Sparse Matrices
- **CSR:** `rowPtr`, `colIdx`, `values` arrays
- **Access row i:** Loop from `rowPtr[i]` to `rowPtr[i+1]`
- **Why:** Skip zeros, save memory, reduce computation

---

## Estimated Timeline

**Total learning time:** 8-12 weeks part-time

- **Weeks 1-2:** C++ review
- **Weeks 3-4:** Parallel computing + GPU architecture
- **Weeks 5-7:** CUDA programming basics
- **Weeks 8-9:** Memory optimization
- **Week 10:** Sparse matrices
- **Weeks 11-12:** Lattice Microbes codebase

**Minimum to understand sparse matrix code:** ~6 weeks
**Minimum to implement sparse kernels:** ~10 weeks

---

## Next Steps

1. **Assess your C++ level** - Can you do the matrix multiply checkpoint?
2. **If yes:** Start Phase 2 (parallel computing concepts)
3. **If no:** Review Phase 1 (C++ fundamentals)
4. **Get CUDA environment:**
   - NVIDIA GPU (even laptop GPU works for learning)
   - Install CUDA Toolkit
   - Test with sample programs
5. **Work through phases sequentially** - Don't skip!
6. **Return to this guide** - Reference memory hierarchy, performance concepts

**Remember:** GPU programming has a steep learning curve, but becomes intuitive with practice. Start with simple examples, build up complexity gradually.

Good luck! 🚀
