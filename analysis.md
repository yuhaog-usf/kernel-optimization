# v2 (new) vs v2 (old) vs v3: Resource Utilization Analysis

## 1. Configuration

| | New v2 | Old v2 | v3 |
|---|---|---|---|
| **Warps** | 4 (128 threads) | 8 (256 threads) | 8 (256 threads) |
| **WMMA warps** | 4 (all) | 4 (warp 0-3 only) | 4 (warp 0-3) |
| **CC warps** | 4 (all, after GEMM) | 8 (all, after GEMM) | 4 (warp 4-7, concurrent) |
| **Tiles/block** | 1 | 4 | 4 |
| **Grid size** (M=N=1024) | (16, 16) = 256 blocks | (16, 4) = 64 blocks | (16, 4) = 64 blocks |

## 2. GEMM Phase: Who Works, Who Idles

**New v2**: All 4 warps do WMMA, no idle warps.

**Old v2**: Warp 4-7 only help load data to shared memory. During WMMA they idle:
```
warp 0-3: [load A,B] [WMMA] [load A,B] [WMMA] ... -> write smem_C
warp 4-7: [load A,B] [idle] [load A,B] [idle] ... -> wait at __syncthreads
```

**v3**: TC warps load independently from global memory. CC warps process the previous tile:
```
warp 0-3 (TC): [load from global mem + WMMA] -> write smem_out[buf]
warp 4-7 (CC): [simultaneously process previous tile's bias+ReLU]
```

## 3. Data Loading Strategy — Key Difference

**New v2 & Old v2**: Collaborative load to shared memory, then WMMA reads from shared memory.
```cuda
// All threads cooperatively load A[64x16] to shared memory
for (int i = threadIdx.x; i < BLOCK_M * WMMA_K; i += BLOCK_DIM)
    smem_A[r][c] = A[global_r * K + global_c];
__syncthreads();
// WMMA loads from shared memory
wmma::load_matrix_sync(frag_a, &smem_A[warp_row][0], WMMA_K);
```

**v3**: Each TC warp loads independently from global memory, no shared A/B tiles.
```cuda
// Each warp independently loads from global memory (B is read 4x redundantly)
const half* a_ptr = A + (tile_row + warp_row) * K + k_step;
wmma::load_matrix_sync(frag_a, a_ptr, K);
const half* b_ptr = B + k_step * N + (tile_col + j * WMMA_N);
wmma::load_matrix_sync(frag_b, b_ptr, N);
```

v3 does this because CC warps don't participate in GEMM. Using collaborative load would
require `__syncthreads()`, which **blocks CC warps** and breaks the pipeline. The cost:
each warp reads B independently, so the same B tile is read 4 times from global memory.

## 4. bias+ReLU Phase

**Old v2**: After GEMM finishes, all 256 threads do bias+ReLU together.
```
[========= GEMM =========] -> __syncthreads -> [= bias+ReLU (256 threads) =]
```

**New v2**: After GEMM finishes, all 128 threads do bias+ReLU together.
```
[========= GEMM =========] -> __syncthreads -> [= bias+ReLU (128 threads) =]
```

**v3**: CC warps process the previous tile while TC warps compute the next (pipelined).
```
TC: [GEMM tile 0 -> buf[0]] [GEMM tile 1 -> buf[1]] [GEMM tile 2 -> buf[0]]
CC:        idle              [bias+relu 0 <- buf[0]] [bias+relu 1 <- buf[1]]
                              ^ overlapped!           ^ overlapped!
```

## 5. Synchronization

**New v2 / Old v2**: Only `__syncthreads()`, simple and reliable.
```cuda
__syncthreads();  // after load -> WMMA can start
__syncthreads();  // after WMMA -> next load can start
__syncthreads();  // after GEMM -> bias+ReLU can start
```

**v3**: Atomic flags + spin-wait + `__syncthreads()` (complex).
```cuda
// TC warp signals CC warp: tile is ready
__threadfence_block();
atomicExch(&tile_ready[buf], 1);

// CC warp waits:
while (atomicAdd(&tile_ready[prev_buf], 0) == 0) {}  // spin-wait
// After processing, reset flag
atomicExch(&tile_ready[prev_buf], 0);

// End of each iteration: full block sync to prevent buffer overwrite
__syncthreads();
```

## 6. Shared Memory Usage

| | New v2 | Old v2 | v3 |
|---|---|---|---|
| **A tile** | `half[64][16]` = 2 KB | same | None (direct global load) |
| **B tile** | `half[16][64]` = 2 KB | same | None |
| **C/output** | `float[64][64]` = 16 KB | same | `float[2][64][64]` = **32 KB** (double buffer) |
| **Flags** | None | None | `int[2]` = 8 B |
| **Total** | ~20 KB | ~20 KB | ~32 KB |

## 7. Resource Utilization: Does Anyone Use All Resources?

### New v2 (128 threads)

Within one block, 4 warps go through two serial phases:
```
Tensor Cores:  [==== busy ====]  [       idle        ]
CUDA Cores:    [     idle     ]  [==== busy ====]
               <- GEMM phase ->  <- bias+ReLU phase ->
```
**At any moment, only one type of execution unit is working.**

Hidden advantage: 128 threads/block is small. A100 SM can hold many blocks.
With ~8 blocks/SM, different blocks at different phases could naturally overlap TC and CC.
But in practice, blocks tend to progress in sync, so this natural overlap rarely happens.

### v3 (256 threads)

During steady-state pipeline:
```
Tensor Cores:  [==== busy (warp 0-3 WMMA) ====]
CUDA Cores:    [==== busy (warp 4-7 bias+ReLU) ====]
               <- both active simultaneously! ->
```
Looks better, **but three sources of waste**:

1. **CC warp spin-wait waste**: CC warps aren't always doing bias+ReLU, they spin-wait for TC completion.
2. **Register waste**: All threads get same register allocation. CC warps need ~10 regs but get 75 (same as TC). 128 CC threads waste ~8320 registers, reducing blocks/SM.
3. **GEMM quality degradation**: No collaborative loading, B matrix read 4x redundantly from global memory.

## 8. Summary of Trade-offs

```
New v2 approach:
  Use minimum resources (4 warps) for maximum serial efficiency.
  -> More blocks, better SM scheduling, zero resource waste.

Old v2 approach:
  Match v3 configuration (8 warps) for fair comparison.
  -> Fair comparison, but warp 4-7 idle most of the time.

v3 approach:
  Split 8 warps into TC+CC groups, pipeline for overlap.
  -> Uses CC/TC concurrency, but pays:
     1. Redundant global memory reads (no collaborative loading)
     2. Complex synchronization mechanism
     3. Double shared memory for double buffering
     4. Pipeline startup and drain overhead (first/last tile can't overlap)
```

**Key insight**: v3 must not only overcome its own overheads but also beat new v2's
advantage of having 4x more blocks (256 vs 64 at 1024x1024). Whether CC+TC concurrency
helps depends on how large bias+ReLU is relative to GEMM — if GEMM dominates (large K),
the time saved by overlapping bias+ReLU is negligible and v2 may win.
