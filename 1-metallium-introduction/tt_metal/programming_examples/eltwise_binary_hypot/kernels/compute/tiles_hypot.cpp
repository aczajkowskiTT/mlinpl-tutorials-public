// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_unary/sqrt.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"



namespace NAMESPACE {
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    // We are going to read from these two circular buffers
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    // and write to the output circular buffer
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    // The destination register is a set of 16 tiles. Which the matrix engine (FPU) can output
    // to. For our case, we are going compute the hypotenuse of two tiles and write the result
    // to destination register 0.
    constexpr uint32_t dst0 = 0; // DST register index for operand 0
    constexpr uint32_t dst1 = 1; // DST register index for operand 1

    // Tell the SFPU that we will be using circular buffers c_in0, c_in1 and c_out0
    // to perform the computation.
    binary_op_init_common(cb_in0, cb_in1, cb_out0);

    // Loop over all the tiles and perform the computation
    for (uint32_t i = 0; i < n_tiles; i++) {
        // Wait until there is a tile in both input circular buffers
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);
        // Make sure there is registers we can use and hold it. The register can be being used by other
        // components. So we need to be sure before we use it. Thus even though there is 16 registers, each
        // time acquire a register, we get 8 of them that we can use until released.
        tile_regs_acquire();


        // ----- TODO: -----
        // 1. Load operand 0 from cb_0 and put into dst0 register
        // 2. Load operand 1 from cb_1 and put into dst1 register
        // 3. Square dst0 and store into dst0 register
        // 4. Square dst1 and store into dst1 register
        // 5. Add dst0 and dst1 and store into dst0 register (HINT: use add_binary_tile)
        // 6. Square dst0 and store into dst0 register
        // --- END TODO ---
        
        // Wait for result to be done and data stored back to the circular buffer
        
        // Release the held register
        tile_regs_commit();
        tile_regs_wait();
        // Make sure there is space in the output circular buffer
        cb_reserve_back(cb_out0, 1);
        // Copy the result from adding the tiles to the output circular buffer
        pack_tile(dst0_reg, cb_out0);
        // Mark the output tile as ready and pop the input tiles
        cb_push_back(cb_out0, 1);
        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
        // Release the held register
        tile_regs_release();
    }
}
}  // namespace NAMESPACE
