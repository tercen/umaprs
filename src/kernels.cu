// CUDA C kernels for umaprs GPU acceleration
// Compiled at runtime via NVRTC — no nvcc needed at build time

// Kernel 1: TQ4 packed 4-bit dot products with shared memory codebook
extern "C" __global__ void tq4_dot(
    const unsigned char* __restrict__ A,  // tile: tile_rows × d_half bytes
    const unsigned char* __restrict__ B,  // data: n × d_half bytes
    float* __restrict__ C,                // output: tile_rows × n
    const float* __restrict__ codebook,   // 16 floats
    int tile_rows, int n, int d_half
) {
    __shared__ float cb[16];

    // Load codebook into shared memory
    int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (linear_tid < 16) {
        cb[linear_tid] = codebook[linear_tid];
    }
    __syncthreads();

    int i = blockIdx.y * blockDim.y + threadIdx.y;  // tile row
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // data point

    if (i >= tile_rows || j >= n) return;

    float dot = 0.0f;
    const unsigned char* row_a = A + i * d_half;
    const unsigned char* row_b = B + j * d_half;

    int sign_disagree = 0;
    for (int k = 0; k < d_half; k++) {
        unsigned char a = row_a[k];
        unsigned char b = row_b[k];
        // Each nibble: (3-bit MSE idx << 1) | 1-bit QJL sign
        unsigned char a_hi = (a >> 4) & 0x0F;
        unsigned char b_hi = (b >> 4) & 0x0F;
        unsigned char a_lo = a & 0x0F;
        unsigned char b_lo = b & 0x0F;
        // MSE dot: use top 3 bits as codebook index
        dot += cb[a_hi >> 1] * cb[b_hi >> 1] + cb[a_lo >> 1] * cb[b_lo >> 1];
        // QJL sign disagreement: bottom bit of each nibble
        sign_disagree += ((a_hi ^ b_hi) & 1) + ((a_lo ^ b_lo) & 1);
    }

    // Pack MSE dot + sign_disagree into output
    // C[i*n + j] stores the QJL-corrected dot product
    // Correction: sqrt(pi/2)/d * mse_per_coord * d * (d - 2*sign_disagree)
    //           = sqrt(pi/2) * mse_per_coord * (d - 2*sign_disagree)
    // where d = d_half * 2 (padded_dims)
    // mse_per_coord is passed via codebook[8] (slot after the 8 centroids)
    float padded_d = (float)(d_half * 2);
    float mse_per_coord = cb[8]; // stored in codebook slot 8
    float agreement = padded_d - 2.0f * (float)sign_disagree;
    // QJL correction: (π/2)/d² · ||r_i||·||r_j|| · agreement
    // With ||r||² ≈ d·mse: (π/2)/d² · d·mse · agreement = (π/2)/d · mse · agreement
    // Note: GPU kernel uses sign(residual) directly (no WHT projection yet)
    float qjl = 1.5707964f / padded_d * mse_per_coord * agreement;
    C[i * n + j] = dot + qjl;
}

// Kernel 1b: TQ8 packed 8-bit dot products (7-bit MSE + 1-bit QJL sign)
// Each byte: (7-bit uniform index << 1) | sign_bit
extern "C" __global__ void tq8_dot(
    const unsigned char* __restrict__ A,  // tile: tile_rows × d bytes
    const unsigned char* __restrict__ B,  // data: n × d bytes
    float* __restrict__ C,                // output: tile_rows × n
    int tile_rows, int n, int d,
    float quant_range,   // quantizer range (3.5)
    float mse_per_coord  // for QJL correction
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= tile_rows || j >= n) return;

    float dot = 0.0f;
    int sign_disagree = 0;
    const unsigned char* row_a = A + i * d;
    const unsigned char* row_b = B + j * d;
    float scale = 2.0f * quant_range / 127.0f;

    for (int k = 0; k < d; k++) {
        unsigned char a = row_a[k];
        unsigned char b = row_b[k];
        // byte = (7-bit idx << 1) | sign
        float va = (float)(a >> 1) * scale - quant_range;
        float vb = (float)(b >> 1) * scale - quant_range;
        dot += va * vb;
        sign_disagree += (a ^ b) & 1;
    }

    // QJL correction: (π/2)/d · mse · (d - 2·disagree)
    float fd = (float)d;
    float agreement = fd - 2.0f * (float)sign_disagree;
    float qjl = 1.5707964f / fd * mse_per_coord * agreement;
    C[i * n + j] = dot + qjl;
}

// Kernel 2: Per-row top-k selection
// Each block handles one row. 256 threads cooperate to find k nearest.
extern "C" __global__ void topk(
    const float* __restrict__ dots,   // tile_rows × n (dot products)
    const float* __restrict__ norms_tile,  // tile_rows
    const float* __restrict__ norms_all,   // n
    unsigned int* __restrict__ out_idx,    // tile_rows × k
    int tile_rows, int n, int k,
    int row_offset,  // for self-skip
    float inv_d,     // 1.0 / padded_dims for TQ mode, 0.0 for f32 mode
    int mode         // 0 = f32 (norms are squared, dist = ni+nj-2*dot)
                     // 1 = TQ  (norms are L2, dist = ni²+nj²-2*ni*nj*clamp(dot*inv_d))
) {
    int row = blockIdx.x;
    if (row >= tile_rows) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;  // 256

    float ni = norms_tile[row];
    const float* row_dots = dots + row * n;
    int global_i = row + row_offset;

    // Each thread: find its local best k candidates
    // Simple approach: each thread keeps 1 best per iteration
    // We iterate k times. Each iteration, all threads find the global min.

    __shared__ float s_dist[256];
    __shared__ unsigned int s_idx[256];
    __shared__ unsigned int found[32]; // max k
    __shared__ float found_dist[32];

    // Init found list
    if (tid < k) {
        found[tid] = 0xFFFFFFFF;
        found_dist[tid] = 1e30f;
    }
    __syncthreads();

    for (int iter = 0; iter < k; iter++) {
        // Each thread scans its stripe for the best (excluding found + self)
        float best_d = 1e30f;
        unsigned int best_j = 0xFFFFFFFF;

        for (int j = tid; j < n; j += stride) {
            if (j == global_i) continue;

            // Check if already found
            bool skip = false;
            for (int f = 0; f < iter; f++) {
                if (found[f] == (unsigned int)j) { skip = true; break; }
            }
            if (skip) continue;

            float dot = row_dots[j];
            float nj = norms_all[j];
            float dist;
            if (mode == 0) {
                // f32 mode: norms are squared, dots are raw
                dist = ni + nj - 2.0f * dot;
            } else {
                // TQ mode: norms are L2, dots are in rotated/scaled space
                float cos_val = dot * inv_d;
                if (cos_val > 1.0f) cos_val = 1.0f;
                if (cos_val < -1.0f) cos_val = -1.0f;
                dist = ni * ni + nj * nj - 2.0f * ni * nj * cos_val;
            }
            if (dist < 0.0f) dist = 0.0f;

            if (dist < best_d) {
                best_d = dist;
                best_j = (unsigned int)j;
            }
        }

        s_dist[tid] = best_d;
        s_idx[tid] = best_j;
        __syncthreads();

        // Thread 0: find global best among all threads
        if (tid == 0) {
            float gmin = 1e30f;
            unsigned int gidx = 0xFFFFFFFF;
            for (int t = 0; t < stride; t++) {
                if (s_dist[t] < gmin) {
                    gmin = s_dist[t];
                    gidx = s_idx[t];
                }
            }
            found[iter] = gidx;
            found_dist[iter] = gmin;
        }
        __syncthreads();
    }

    // Write output
    if (tid < k) {
        out_idx[row * k + tid] = found[tid];
    }
}

// Kernel 3: Brute-force f32 distance matrix via dot products (alternative to cuBLAS)
// Useful when cuBLAS is not available
extern "C" __global__ void f32_dot(
    const float* __restrict__ A,  // tile: tile_rows × d
    const float* __restrict__ B,  // data: n × d (row-major)
    float* __restrict__ C,        // output: tile_rows × n
    int tile_rows, int n, int d
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= tile_rows || j >= n) return;

    float dot = 0.0f;
    const float* row_a = A + i * d;
    const float* row_b = B + j * d;

    for (int k = 0; k < d; k++) {
        dot += row_a[k] * row_b[k];
    }

    C[i * n + j] = dot;
}
