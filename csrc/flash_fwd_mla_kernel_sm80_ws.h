#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

using namespace cute;

#include "named_barrier.h"
#include "utils.h"
#include "softmax.h"
#include "static_switch.h"
#include "flash_mla.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
// SM80-compatible named barrier helpers using raw PTX.
// bar.sync and bar.arrive are available on all architectures from compute_20+.

namespace flash {
namespace sm80_barrier {

__device__ __forceinline__ void named_barrier_arrive(int barrier_id, int num_threads) {
    asm volatile("bar.arrive %0, %1;" :: "r"(barrier_id), "r"(num_threads));
}

__device__ __forceinline__ void named_barrier_sync(int barrier_id, int num_threads) {
    asm volatile("bar.sync %0, %1;" :: "r"(barrier_id), "r"(num_threads));
}

}  // namespace sm80_barrier

////////////////////////////////////////////////////////////////////////////////////////////////////
// Consumer-only cross-warp reduction: uses named barrier instead of __syncthreads()
// barrier_id and num_threads define the consumer-only barrier.

template<int kNWarpsN, typename Engine0, typename Layout0>
__forceinline__ __device__ void cross_warp_reduce_max_ws(
    Tensor<Engine0, Layout0> &val, float *smem_reduce,
    int n_warp_idx, const int *row_indices, int stride,
    int barrier_id, int num_threads)
{
    #pragma unroll
    for (int mi = 0; mi < size(val); ++mi) {
        smem_reduce[row_indices[mi] * stride + n_warp_idx] = val(mi);
    }
    sm80_barrier::named_barrier_sync(barrier_id, num_threads);
    #pragma unroll
    for (int mi = 0; mi < size(val); ++mi) {
        float result = smem_reduce[row_indices[mi] * stride];
        #pragma unroll
        for (int w = 1; w < kNWarpsN; ++w) {
            result = max(result, smem_reduce[row_indices[mi] * stride + w]);
        }
        val(mi) = result;
    }
    sm80_barrier::named_barrier_sync(barrier_id, num_threads);
}

template<int kNWarpsN, typename Engine0, typename Layout0>
__forceinline__ __device__ void cross_warp_reduce_sum_ws(
    Tensor<Engine0, Layout0> &val, float *smem_reduce,
    int n_warp_idx, const int *row_indices, int stride,
    int barrier_id, int num_threads)
{
    #pragma unroll
    for (int mi = 0; mi < size(val); ++mi) {
        smem_reduce[row_indices[mi] * stride + n_warp_idx] = val(mi);
    }
    sm80_barrier::named_barrier_sync(barrier_id, num_threads);
    #pragma unroll
    for (int mi = 0; mi < size(val); ++mi) {
        float result = 0.f;
        #pragma unroll
        for (int w = 0; w < kNWarpsN; ++w) {
            result += smem_reduce[row_indices[mi] * stride + w];
        }
        val(mi) = result;
    }
    sm80_barrier::named_barrier_sync(barrier_id, num_threads);
}

}  // namespace flash

////////////////////////////////////////////////////////////////////////////////////////////////////
// Consumer-only Softmax: wraps the base Softmax with WS-safe cross-warp reductions.

template <int kNRows>
struct SoftmaxWS : public flash::Softmax<kNRows> {
    using Base = flash::Softmax<kNRows>;
    using TensorT = typename Base::TensorT;

    int barrier_id_;
    int num_threads_;

    __forceinline__ __device__ SoftmaxWS(int barrier_id, int num_threads)
        : barrier_id_(barrier_id), num_threads_(num_threads) {}

    template<bool Is_first, bool Check_inf=false, int kNWarpsN, typename Tensor0>
    __forceinline__ __device__ TensorT softmax_cross_warp_ws(Tensor0 &acc_s, float softmax_scale_log2,
                                                              float *smem_reduce, int n_warp_idx,
                                                              const int *row_indices, int stride) {
        using namespace cute;
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<0>(scores))::value == kNRows);
        TensorT scale_o;
        clear(scale_o);
        if (Is_first) {
            flash::template reduce_max</*zero_init=*/true>(scores, this->row_max);
            flash::cross_warp_reduce_max_ws<kNWarpsN>(this->row_max, smem_reduce, n_warp_idx, row_indices, stride, barrier_id_, num_threads_);
            flash::scale_apply_exp2(scores, this->row_max, softmax_scale_log2);
            flash::reduce_sum</*zero_init=*/true>(scores, this->row_sum);
        } else {
            TensorT scores_max_prev = make_fragment_like(this->row_max);
            cute::copy(this->row_max, scores_max_prev);
            flash::template reduce_max</*zero_init=*/false>(scores, this->row_max);
            flash::cross_warp_reduce_max_ws<kNWarpsN>(this->row_max, smem_reduce, n_warp_idx, row_indices, stride, barrier_id_, num_threads_);
            #pragma unroll
            for (int mi = 0; mi < size(this->row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? this->row_max(mi)
                    : (this->row_max(mi) == -INFINITY ? 0.0f : this->row_max(mi));
                float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                scale_o(mi) = scores_scale;
                this->row_sum(mi) *= scores_scale;
            }
            flash::scale_apply_exp2(scores, this->row_max, softmax_scale_log2);
            flash::reduce_sum</*zero_init=*/false>(scores, this->row_sum);
        }
        return scale_o;
    };

    template<bool Is_dropout=false, bool Split=false, int kNWarpsN, typename Tensor0>
    __forceinline__ __device__ TensorT normalize_softmax_lse_cross_warp_ws(
            Tensor0 &acc_o, float softmax_scale,
            float *smem_reduce, int n_warp_idx,
            const int *row_indices, int stride, float rp_dropout=1.0) {
        using namespace cute;
        flash::SumOp<float> sum_op;
        flash::quad_allreduce_(this->row_sum, this->row_sum, sum_op);
        flash::cross_warp_reduce_sum_ws<kNWarpsN>(this->row_sum, smem_reduce, n_warp_idx, row_indices, stride, barrier_id_, num_threads_);
        TensorT lse = make_fragment_like(this->row_sum);
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            float sum = this->row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
            lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : this->row_max(mi) * softmax_scale + __logf(sum);
            float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scale; }
        }
        return lse;
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename PrecType, int DIM, int DIM2 = DIM>
constexpr auto getSmemLayoutK_ws() {
    constexpr int headSizeBytes = sizeof(PrecType) * DIM;
    constexpr int headSizeBytes2 = sizeof(PrecType) * DIM2;

    if constexpr (headSizeBytes % 128 == 0 && headSizeBytes2 % 128 == 0) {
        return GMMA::Layout_K_SW128_Atom<PrecType>{};
    } else if constexpr (headSizeBytes % 64 == 0 && headSizeBytes2 % 64 == 0) {
        return GMMA::Layout_K_SW64_Atom<PrecType>{};
    } else {
        return GMMA::Layout_K_SW32_Atom<PrecType>{};
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel traits for warp-specialized SM80 MLA kernel
// Consumer (4 warps, tidx 0-127): QK^T GEMM + softmax
// Producer (4 warps, tidx 128-255): global memory loads
// Both: PV GEMM via TiledMmaO

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::bfloat16_t, int kHeadDimV_ = 0>
struct Flash_fwd_kernel_traits_mla_ws {
    using Element = elem_type;
    using ElementAccum = float;
    using index_t = int64_t;

    static constexpr int kNWarps = kNWarps_;         // 8 total
    static constexpr int kNThreads = kNWarps * 32;   // 256
    static constexpr int kNWarpsS = 4;               // consumer warps
    static constexpr int kNThreadsS = kNWarpsS * 32; // 128 consumer threads

    // Consumer QK^T MMA: 2D tiling with 4 warps
    static constexpr int kNWarpsM = 2;               // M-dimension warps in consumer
    static constexpr int kNWarpsN = kNWarpsS / kNWarpsM; // 2 N-dimension warps in consumer
    static_assert(kNWarpsS == kNWarpsM * kNWarpsN);

    // PV GEMM (all 8 warps): 2M x 4N
    static constexpr int kNWarpsN_O = kNWarps / 2;   // 4 N-dimension warps in full TiledMmaO

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kHeadDimV = kHeadDimV_ != 0 ? kHeadDimV_ : kHeadDim;
    static_assert(kHeadDimV % 32 == 0);
    static_assert(kHeadDimV <= kHeadDim);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
    static constexpr int kNumInnerStagesK = 3;
    static_assert(kHeadDim % kNumInnerStagesK == 0, "kHeadDim must be divisible by kNumInnerStagesK");

    // Consumer TiledMma for QK^T: 4 warps in 2M x 2N layout
    // Covers 32x16 per call; for kBlockN=32, CuTe produces MMA_N=2 tiles
    using MMA_Atom_Arch = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
    using TiledMmaS = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<Int<kNWarpsM>, Int<kNWarpsN>, _1>>,
        Tile<Int<16 * kNWarpsM>, Int<8 * kNWarpsN>, _16>>;

    // Full TiledMma for PV GEMM: all 8 warps in 2M x 4N layout
    using TiledMmaO = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<_2, _4, _1>>,
        Tile<_32, _32, _16>>;

    // Smem layouts
    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutSplitQ = decltype(composition(
        SmemLayoutQ{},
        make_layout(Shape<Int<kBlockM>, Int<kHeadDim / kNumInnerStagesK>, Int<kNumInnerStagesK>>{})));

    using kP = Int<2>; // pipeline count (double buffer)
    using SmemLayoutK = decltype(tile_to_shape(
            getSmemLayoutK_ws<Element, kHeadDim, kHeadDimV>(),
            Shape<Int<kBlockN>, Int<kHeadDim>, kP>{}));
    using SmemLayoutSplitK = decltype(composition(
        SmemLayoutK{}, make_layout(Shape<Int<kBlockN>, Int<kHeadDim/kNumInnerStagesK>, Int<kNumInnerStagesK>, kP>{})));

    using SmemLayoutV = decltype(tile_to_shape(
            getSmemLayoutK_ws<Element, kHeadDim, kHeadDimV>(),
            Shape<Int<kBlockN>, Int<kHeadDimV>, kP>{}));
    using SmemLayoutVtransposed = decltype(composition(SmemLayoutK{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>, kP>{}, Stride<Int<kBlockN>,_1, Int<kHeadDim * kBlockN>>{})));

    // P smem layout
    static constexpr int kBlockKSmemP = kBlockN % 64 == 0 ? 64 : 32;
    static constexpr int kSwizzleP = kBlockKSmemP == 32 ? 2 : 3;
    using SmemLayoutAtomP = decltype(
        composition(Swizzle<kSwizzleP, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmemP>>,
                           Stride<Int<kBlockKSmemP>, _1>>{}));
    using SmemLayoutP = decltype(tile_to_shape(
        SmemLayoutAtomP{},
        Shape<Int<kBlockM>, Int<kBlockN>>{}));

    // O smem layout (for epilogue store)
    using SmemLayoutAtomO = decltype(composition(
            Swizzle<kSwizzle, 3, 3>{},
            Layout<Shape<Int<8>, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
            SmemLayoutAtomO{},
            Shape<Int<kBlockM>, Int<kHeadDimV>>{}));

    // Copy atoms
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;
    using SmemCopyAtomB = Copy_Atom<SM75_U32x2_LDSM_N, Element>;
    using SmemCopyAtomBTransposed = Copy_Atom<SM75_U16x4_LDSM_T, elem_type>;
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>;

    // GmemTiledCopy for PRODUCER loads (128 threads)
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    using Gmem_copy_struct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    static constexpr int kNThreadsLoad = kNThreads - kNThreadsS; // 128
    static_assert(kNThreadsLoad % kGmemThreadsPerRow == 0, "kNThreadsLoad must be a multiple of kGmemThreadsPerRow");

    using GmemLayoutAtom = Layout<
            Shape<Int<kNThreadsLoad / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
            Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopy = decltype(make_tiled_copy(
            Copy_Atom<Gmem_copy_struct, Element>{},
            GmemLayoutAtom{},
            Layout<Shape<_1, _8>>{}));

    // O store uses all 256 threads
    using GmemLayoutAtomO = Layout<
            Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
            Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyO = decltype(make_tiled_copy(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
            GmemLayoutAtomO{},
            Layout<Shape<_1, _8>>{}));

    static constexpr int kGmemElemsPerLoadAccum = sizeof(cute::uint128_t) / sizeof(ElementAccum);
    static constexpr int kGmemThreadsPerRowAccum = kBlockKSmem / kGmemElemsPerLoadAccum;
    using GmemLayoutAtomOaccum = Layout<
            Shape<Int<kNThreads / kGmemThreadsPerRowAccum>, Int<kGmemThreadsPerRowAccum>>,
            Stride<Int<kGmemThreadsPerRowAccum>, _1>>;
    using GmemTiledCopyOaccum = decltype(make_tiled_copy(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
            GmemLayoutAtomOaccum{},
            Layout<Shape<_1, _4>>{}));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace flash {

using namespace cute;

template<typename Kernel_traits>
struct SharedStorageMLA_WS {
    union {
        struct {
            cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutQ>> smem_q;
            cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutK>> smem_k;
        };
        struct {
            cute::array_aligned<typename Kernel_traits::ElementAccum, cute::cosize_v<typename Kernel_traits::SmemLayoutO>> smem_o;
        };
    };
    // P intermediate buffer for PV GEMM
    cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutP>> smem_p;
    // Communication buffers between consumer and producer
    cute::array_aligned<typename Kernel_traits::ElementAccum, Kernel_traits::kBlockM> smem_scale;
    cute::array_aligned<typename Kernel_traits::ElementAccum, Kernel_traits::kBlockM> smem_max;
    cute::array_aligned<typename Kernel_traits::ElementAccum, Kernel_traits::kBlockM> smem_sum;
    // Cross-warp reduction buffer: max of (consumer kNWarpsN+1, epilogue kNWarpsN_O+1)
    static constexpr int kReduceStride = Kernel_traits::kNWarpsN_O + 1;  // 5 (for epilogue normalize)
    cute::array_aligned<typename Kernel_traits::ElementAccum, Kernel_traits::kBlockM * kReduceStride> smem_reduce;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Split, typename SharedStorage, typename AccO, typename Softmax>
__forceinline__ __device__ void store_ws(const Flash_fwd_mla_params &params, const int bidb, const int bidh, const int m_block, const int n_split_idx,
                                         SharedStorage &shared_storage, AccO tOrO, Softmax softmax,
                                         float *smem_reduce, int n_warp_idx, const int *row_indices, int reduce_stride) {
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarpsN_O = Kernel_traits::kNWarpsN_O;
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    const int tidx = threadIdx.x;

    typename Kernel_traits::TiledMmaO tiled_mma_o;
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);

    const int split_offset = __ldg(params.num_splits_ptr + bidb);

    // Epilogue: all 256 threads participate. Use base class normalize with __syncthreads().
    Tensor lse = softmax.template normalize_softmax_lse_cross_warp</*Is_dropout=*/false, Split, kNWarpsN_O>(
        tOrO, params.scale_softmax, smem_reduce, n_warp_idx, row_indices, reduce_stride);

    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;
    Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(shared_storage.smem_o.data())), typename Kernel_traits::SmemLayoutO{});
    using SmemTiledCopyO = std::conditional_t<
            !Split,
            typename Kernel_traits::SmemCopyAtomO,
            typename Kernel_traits::SmemCopyAtomOaccum
    >;
    auto smem_tiled_copy_Oaccum = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma_o);
    auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor rO = flash::convert_type<ElementO>(tOrO);
    Tensor taccOrOaccum = smem_thr_copy_Oaccum.retile_S(rO);
    Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(sOaccum);

    cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);

    const index_t row_offset_o = bidb * params.o_batch_stride + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_oaccum = (((split_offset + n_split_idx) * params.h + bidh) * params.seqlen_q + m_block * kBlockM) * params.d_v;
    const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
    const index_t row_offset_lseaccum = ((split_offset + n_split_idx) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;

    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                 Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                 make_stride(Split ? kHeadDimV : params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + (Split ? row_offset_lseaccum : row_offset_lse)),
                                   Shape<Int<kBlockM>>{}, Stride<_1>{});

    using GmemTiledCopyO = std::conditional_t<!Split, typename Kernel_traits::GmemTiledCopyO, typename Kernel_traits::GmemTiledCopyOaccum>;
    GmemTiledCopyO gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

    __syncthreads();

    Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
    cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});
    Tensor taccOcO = thr_mma_o.partition_C(caccO);
    Tensor taccOcO_row = taccOcO(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));
    if (get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < params.seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
        }
    }

    Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));
    Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, params.seqlen_q - m_block * kBlockM
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal, bool Split, typename SharedStorage>
__forceinline__ __device__ void compute_attn_1rowblock_splitkv_mla_ws(const Flash_fwd_mla_params &params,
                                                                      const int bidb, const int bidh, const int m_block,
                                                                      const int n_split_idx, const int seqlen_k,
                                                                      const int n_block_min, const int n_block_max,
                                                                      SharedStorage &shared_storage) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kNThreadsS = Kernel_traits::kNThreadsS;
    constexpr int kNWarpsN = Kernel_traits::kNWarpsN;
    constexpr int kNWarpsM = Kernel_traits::kNWarpsM;
    constexpr int kNumInnerStagesK = Kernel_traits::kNumInnerStagesK;
    constexpr int kReduceStride = kNWarpsN + 1;       // for consumer softmax
    constexpr int kReduceStride_O = Kernel_traits::kNWarpsN_O + 1;  // for epilogue normalize

    // Barrier IDs
    constexpr int kBarrierKReady = 1;
    constexpr int kBarrierSReady = 2;
    constexpr int kBarrierSoftmaxReady = 3;
    constexpr int kBarrierPVisible = 4;

    const int tidx = threadIdx.x;
    const bool is_consumer = (tidx < kNThreadsS);
    const int warp_idx = tidx / 32;
    const int n_warp_idx = is_consumer ? (warp_idx / kNWarpsM) : 0;
    float *smem_reduce = reinterpret_cast<float *>(shared_storage.smem_reduce.data());
    float *smem_scale_ptr = reinterpret_cast<float *>(shared_storage.smem_scale.data());
    float *smem_max_ptr = reinterpret_cast<float *>(shared_storage.smem_max.data());
    float *smem_sum_ptr = reinterpret_cast<float *>(shared_storage.smem_sum.data());

    int n_block = n_block_max - 1;

    // Smem tensors (common)
    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), typename Kernel_traits::SmemLayoutQ{});
    Tensor sQ_split = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), typename Kernel_traits::SmemLayoutSplitQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename Kernel_traits::SmemLayoutK{});
    Tensor sK_split = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename Kernel_traits::SmemLayoutSplitK{});
    Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sP = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), typename Kernel_traits::SmemLayoutP{});

    // PV GEMM setup (all 8 warps via TiledMmaO)
    typename Kernel_traits::TiledMmaO tiled_mma_o;
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);
    Tensor tOrVt_o = thr_mma_o.partition_fragment_B(sVt(_, _, 0));
    Tensor acc_o = partition_fragment_C(tiled_mma_o, Shape<Int<kBlockM>, Int<kHeadDimV>>{});
    clear(acc_o);

    // PV GEMM copy atoms
    auto smem_tiled_copy_PreadA_o = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_o);
    auto smem_thr_copy_PreadA_o = smem_tiled_copy_PreadA_o.get_thread_slice(tidx);
    Tensor tSrP_o = thr_mma_o.partition_fragment_A(sP);
    Tensor tPsP_read_o = smem_thr_copy_PreadA_o.partition_S(sP);

    auto smem_tiled_copy_V_o = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomBTransposed{}, tiled_mma_o);
    auto smem_thr_copy_V_o = smem_tiled_copy_V_o.get_thread_slice(tidx);
    Tensor tOsVt_o = smem_thr_copy_V_o.partition_S(sVt);

    // Row indices for PV GEMM (used by both consumer and producer for scale_o lookup)
    Tensor cO_id = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});
    Tensor tOcO_id = thr_mma_o.partition_C(cO_id);
    auto o_id_rowcol = make_tensor(tOcO_id.data(), flash::convert_layout_acc_rowcol(tOcO_id.layout()));
    constexpr int kNRowsO = decltype(size<0>(o_id_rowcol))::value;
    int row_indices_o[kNRowsO];
    #pragma unroll
    for (int mi = 0; mi < kNRowsO; ++mi) {
        row_indices_o[mi] = int(get<0>(o_id_rowcol(mi, 0)));
    }

    // Consumer-only barrier for softmax cross-warp reduction
    constexpr int kBarrierSoftmaxReduce = 5;
    SoftmaxWS<2 * size<1>(acc_o)> softmax(kBarrierSoftmaxReduce, kNThreadsS);

    // ===== CONSUMER PATH (tidx 0-127) =====
    if (is_consumer) {
        // Consumer QK^T MMA setup (4 warps)
        typename Kernel_traits::TiledMmaS tiled_mma_s;
        auto thr_mma_s = tiled_mma_s.get_thread_slice(tidx);
        Tensor tSrQ = thr_mma_s.partition_fragment_A(sQ_split(_, _, 0));
        Tensor tSrK = thr_mma_s.partition_fragment_B(sK_split(_, _, 0, 0));

        auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_s);
        auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
        Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ_split);

        auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomB{}, tiled_mma_s);
        auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
        Tensor tSsK = smem_thr_copy_K.partition_S(sK_split);

        // P store: consumer writes C-fragment to smem
        auto smem_tiled_copy_PwriteC = make_tiled_copy_C(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_s);
        auto smem_thr_copy_PwriteC = smem_tiled_copy_PwriteC.get_thread_slice(tidx);

        // Row indices for consumer's softmax reduction
        Tensor cS_id = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
        Tensor tScS_id = thr_mma_s.partition_C(cS_id);
        auto scores_id_rowcol = make_tensor(tScS_id.data(), flash::convert_layout_acc_rowcol(tScS_id.layout()));
        constexpr int kSoftmaxRows = decltype(size<0>(scores_id_rowcol))::value;
        int row_indices_s[kSoftmaxRows];
        #pragma unroll
        for (int mi = 0; mi < kSoftmaxRows; ++mi) {
            row_indices_s[mi] = int(get<0>(scores_id_rowcol(mi, 0)));
        }

        int smem_pipe_read = 0;

        // Wait for producer to finish loading Q and first K
        flash::sm80_barrier::named_barrier_sync(kBarrierKReady, kNThreads);

        // Masking iterations
        constexpr int n_masking_steps = !Is_causal ? 1 : cute::ceil_div(kBlockM, kBlockN) + 1;
        #pragma unroll
        for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
            Tensor acc_s = partition_fragment_C(tiled_mma_s, Shape<Int<kBlockM>, Int<kBlockN>>{});
            clear(acc_s);

            // QK^T GEMM (4-warp, 3 inner K-stages)
            #pragma unroll
            for (int slice = 0; slice < kNumInnerStagesK; ++slice) {
                Tensor tSsQ_p = tSsQ(_, _, _, slice);
                Tensor tSsK_p = tSsK(_, _, _, slice, smem_pipe_read);
                flash::gemm_8x<false, false>(acc_s, tSrQ, tSrK, tSsQ_p, tSsK_p,
                    tiled_mma_s, smem_tiled_copy_Q, smem_tiled_copy_K,
                    smem_thr_copy_Q, smem_thr_copy_K);
            }

            // Masking
            const bool is_masking_step = masking_step >= 0;
            const bool is_first_masking_step = masking_step == 0;
            if (is_masking_step) {
                Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
                Tensor tScS = thr_mma_s.partition_C(cS);
#pragma unroll
                for (int i = 0; i < size(acc_s); ++i) {
                    if constexpr (!Is_causal) {
                        if (int(get<1>(tScS(i))) >= int(seqlen_k - n_block * kBlockN)) acc_s(i) = -INFINITY;
                    } else {
                        int row = int(get<0>(tScS(i)));
                        int col_limit_right = seqlen_k - 1 - n_block * kBlockN - (params.seqlen_q - 1 - (m_block * kBlockM + row)) / params.ngroups;
                        if (int(get<1>(tScS(i))) > col_limit_right) acc_s(i) = -INFINITY;
                    }
                }
            }

            // Softmax (cross-warp, kNWarpsN=2)
            Tensor scale_o = is_first_masking_step
                ? softmax.template softmax_cross_warp_ws</*Is_first=*/true, /*Check_inf=*/Is_causal, kNWarpsN>(acc_s, params.scale_softmax_log2, smem_reduce, n_warp_idx, row_indices_s, kReduceStride)
                : is_masking_step ?
                softmax.template softmax_cross_warp_ws</*Is_first=*/false, /*Check_inf=*/Is_causal, kNWarpsN>(acc_s, params.scale_softmax_log2, smem_reduce, n_warp_idx, row_indices_s, kReduceStride)
                : softmax.template softmax_cross_warp_ws</*Is_first=*/false, /*Check_inf=*/false, kNWarpsN>(acc_s, params.scale_softmax_log2, smem_reduce, n_warp_idx, row_indices_s, kReduceStride);

            // Convert P to bf16/fp16, store to smem_p
            Tensor rP = flash::convert_type<Element>(acc_s);
            Tensor tPrP_src = smem_thr_copy_PwriteC.retile_S(rP);
            Tensor tPsP_dst = smem_thr_copy_PwriteC.partition_D(sP);
            cute::copy(smem_tiled_copy_PwriteC, tPrP_src, tPsP_dst);

            // Store scale_o to smem for producer
            #pragma unroll
            for (int mi = 0; mi < kSoftmaxRows; ++mi) {
                smem_scale_ptr[row_indices_s[mi]] = scale_o(mi);
            }

            // Signal producer: P and scale are ready
            flash::sm80_barrier::named_barrier_arrive(kBarrierSReady, kNThreads);

            // Consumer-only barrier to ensure P visibility across consumer warps
            flash::sm80_barrier::named_barrier_sync(kBarrierPVisible, kNThreadsS);

            // Rescale consumer's acc_o
            Tensor scale_o_pv = make_tensor<float>(Shape<Int<kNRowsO>>{});
            #pragma unroll
            for (int mi = 0; mi < kNRowsO; ++mi) {
                scale_o_pv(mi) = smem_scale_ptr[row_indices_o[mi]];
            }
            flash::rescale_o(acc_o, scale_o_pv);

            // PV GEMM (consumer's portion via 8-warp TiledMmaO)
            Tensor tOsVt_p = tOsVt_o(_, _, _, smem_pipe_read);
            flash::gemm_8x<false, false>(acc_o, tSrP_o, tOrVt_o, tPsP_read_o, tOsVt_p,
                tiled_mma_o, smem_tiled_copy_PreadA_o, smem_tiled_copy_V_o,
                smem_thr_copy_PreadA_o, smem_thr_copy_V_o);

            // Wait for next K to be ready
            if (n_block - 1 >= n_block_min) {
                smem_pipe_read = 1 - smem_pipe_read;
                flash::sm80_barrier::named_barrier_sync(kBarrierKReady, kNThreads);
            }

            if (n_masking_steps > 1 && n_block <= n_block_min) {
                --n_block;
                break;
            }
        }

        // Non-masking iterations
        for (; n_block >= n_block_min; --n_block) {
            Tensor acc_s = partition_fragment_C(tiled_mma_s, Shape<Int<kBlockM>, Int<kBlockN>>{});
            clear(acc_s);

            #pragma unroll
            for (int slice = 0; slice < kNumInnerStagesK; ++slice) {
                Tensor tSsQ_p = tSsQ(_, _, _, slice);
                Tensor tSsK_p = tSsK(_, _, _, slice, smem_pipe_read);
                flash::gemm_8x<false, false>(acc_s, tSrQ, tSrK, tSsQ_p, tSsK_p,
                    tiled_mma_s, smem_tiled_copy_Q, smem_tiled_copy_K,
                    smem_thr_copy_Q, smem_thr_copy_K);
            }

            Tensor scale_o = softmax.template softmax_cross_warp_ws</*Is_first=*/false, /*Check_inf=*/false, kNWarpsN>(
                acc_s, params.scale_softmax_log2, smem_reduce, n_warp_idx, row_indices_s, kReduceStride);

            Tensor rP = flash::convert_type<Element>(acc_s);
            {
                Tensor tPrP_src = smem_thr_copy_PwriteC.retile_S(rP);
                Tensor tPsP_dst = smem_thr_copy_PwriteC.partition_D(sP);
                cute::copy(smem_tiled_copy_PwriteC, tPrP_src, tPsP_dst);
            }
            #pragma unroll
            for (int mi = 0; mi < kSoftmaxRows; ++mi) {
                smem_scale_ptr[row_indices_s[mi]] = scale_o(mi);
            }

            flash::sm80_barrier::named_barrier_arrive(kBarrierSReady, kNThreads);
            flash::sm80_barrier::named_barrier_sync(kBarrierPVisible, kNThreadsS);

            Tensor scale_o_pv = make_tensor<float>(Shape<Int<kNRowsO>>{});
            #pragma unroll
            for (int mi = 0; mi < kNRowsO; ++mi) {
                scale_o_pv(mi) = smem_scale_ptr[row_indices_o[mi]];
            }
            flash::rescale_o(acc_o, scale_o_pv);

            Tensor tOsVt_p = tOsVt_o(_, _, _, smem_pipe_read);
            flash::gemm_8x<false, false>(acc_o, tSrP_o, tOrVt_o, tPsP_read_o, tOsVt_p,
                tiled_mma_o, smem_tiled_copy_PreadA_o, smem_tiled_copy_V_o,
                smem_thr_copy_PreadA_o, smem_thr_copy_V_o);

            if (n_block - 1 >= n_block_min) {
                smem_pipe_read = 1 - smem_pipe_read;
                flash::sm80_barrier::named_barrier_sync(kBarrierKReady, kNThreads);
            }
        }

        // Write final softmax stats for producer
        #pragma unroll
        for (int mi = 0; mi < kSoftmaxRows; ++mi) {
            smem_max_ptr[row_indices_s[mi]] = softmax.row_max(mi);
            smem_sum_ptr[row_indices_s[mi]] = softmax.row_sum(mi);
        }
        flash::sm80_barrier::named_barrier_arrive(kBarrierSoftmaxReady, kNThreads);

    // ===== PRODUCER PATH (tidx 128-255) =====
    } else {
        const int *block_table = params.block_table + bidb * params.block_table_batch_stride;
        int cur_block_table = __ldg(&block_table[n_block]);

        const index_t row_offset_q = bidb * params.q_batch_stride + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
        Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                                Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                make_stride(params.q_row_stride, _1{}));
        const index_t row_offset_k = (bidh / params.h_h_k_ratio) * params.k_head_stride;
        Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                                Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                make_stride(params.k_row_stride, _1{}));

        typename Kernel_traits::GmemTiledCopy gmem_tiled_copy;
        auto gmem_thr_copy = gmem_tiled_copy.get_thread_slice(tidx - kNThreadsS);

        Tensor tQgQ = gmem_thr_copy.partition_S(gQ);
        Tensor tQsQ = gmem_thr_copy.partition_D(sQ);
        Tensor tKgK = gmem_thr_copy.partition_S(gK);
        Tensor tKsK = gmem_thr_copy.partition_D(sK);
        auto K_PIPE_MAX = size<3>(tKsK);

        Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
        Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
        Tensor tQcQ = gmem_thr_copy.partition_S(cQ);
        Tensor tKVcKV = gmem_thr_copy.partition_S(cKV);
        Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
        Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

        // Prologue: Load Q
        flash::copy</*Is_even_MN*/false, /*Is_even_K*/true>(gmem_tiled_copy, tQgQ, tQsQ, tQcQ, tQpQ,
                                            params.seqlen_q - m_block * kBlockM);

        // Prologue: Load first K block
        int smem_pipe_read = 0;
        int smem_pipe_write = K_PIPE_MAX - 1;
        const index_t offset_k = cur_block_table * params.k_batch_stride;
        tKgK.data() = tKgK.data() + offset_k;
        Tensor tKsK_p = tKsK(_, _, _, 0);
        flash::copy</*Is_even_MN*/false, /*Is_even_K*/true, /*Clear_OOB_MN=*/true>(gmem_tiled_copy, tKgK, tKsK_p, tKVcKV, tKVpKV,
                                           seqlen_k - n_block * kBlockN);
        tKgK.data() = tKgK.data() + -offset_k;
        cute::cp_async_fence();
        flash::cp_async_wait<0>();

        // Signal consumer: Q and first K are ready
        flash::sm80_barrier::named_barrier_arrive(kBarrierKReady, kNThreads);

        // Main loop
        for (int n_blk = n_block; n_blk >= n_block_min; --n_blk) {
            // Issue next K load (double-buffered)
            if (n_blk - 1 >= n_block_min) {
                cur_block_table = __ldg(&block_table[n_blk - 1]);
                const index_t offset_k_next = cur_block_table * params.k_batch_stride;
                tKgK.data() = tKgK.data() + offset_k_next;
                Tensor tKsK_next = tKsK(_, _, _, smem_pipe_write);
                flash::copy</*Is_even_MN=*/true, /*Is_even_K=*/true>(gmem_tiled_copy, tKgK, tKsK_next, tKVcKV, tKVpKV);
                tKgK.data() = tKgK.data() + -offset_k_next;
                cute::cp_async_fence();
            }

            // Wait for consumer to produce P and scale
            flash::sm80_barrier::named_barrier_sync(kBarrierSReady, kNThreads);

            // Read scale_o from smem and rescale producer's acc_o
            Tensor scale_o_prod = make_tensor<float>(Shape<Int<kNRowsO>>{});
            #pragma unroll
            for (int mi = 0; mi < kNRowsO; ++mi) {
                scale_o_prod(mi) = smem_scale_ptr[row_indices_o[mi]];
            }
            flash::rescale_o(acc_o, scale_o_prod);

            // PV GEMM (producer's portion via 8-warp TiledMmaO)
            Tensor tOsVt_p = tOsVt_o(_, _, _, smem_pipe_read);
            flash::gemm_8x<false, false>(acc_o, tSrP_o, tOrVt_o, tPsP_read_o, tOsVt_p,
                tiled_mma_o, smem_tiled_copy_PreadA_o, smem_tiled_copy_V_o,
                smem_thr_copy_PreadA_o, smem_thr_copy_V_o);

            // Wait for next K load and signal consumer
            if (n_blk - 1 >= n_block_min) {
                flash::cp_async_wait<0>();
                smem_pipe_write = smem_pipe_read;
                smem_pipe_read = 1 - smem_pipe_read;
                flash::sm80_barrier::named_barrier_arrive(kBarrierKReady, kNThreads);
            }
        }

        // Wait for consumer's final softmax stats
        flash::sm80_barrier::named_barrier_sync(kBarrierSoftmaxReady, kNThreads);

        // Read final row_max and row_sum from smem
        #pragma unroll
        for (int mi = 0; mi < kNRowsO; ++mi) {
            softmax.row_max(mi) = smem_max_ptr[row_indices_o[mi]];
            softmax.row_sum(mi) = smem_sum_ptr[row_indices_o[mi]];
        }
    }

    // ===== EPILOGUE (all threads) =====
    // n_warp_idx for TiledMmaO: warp_idx / 2 (2M warps)
    const int n_warp_idx_o = warp_idx / 2;
    if (!Split) {
        store_ws<Kernel_traits, false>(params, bidb, bidh, m_block, n_split_idx, shared_storage, acc_o, softmax,
                                       smem_reduce, n_warp_idx_o, row_indices_o, kReduceStride_O);
    } else {
        store_ws<Kernel_traits, true>(params, bidb, bidh, m_block, n_split_idx, shared_storage, acc_o, softmax,
                                      smem_reduce, n_warp_idx_o, row_indices_o, kReduceStride_O);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal, typename SharedStorage>
__global__ void __launch_bounds__(Kernel_traits::kNThreads)
flash_fwd_splitkv_mla_ws_kernel(__grid_constant__ const Flash_fwd_mla_params params) {
    constexpr int kBlockN = Kernel_traits::kBlockN;
    const int m_block = blockIdx.x;
    const int bidh = blockIdx.y;
    const int partition_idx = blockIdx.z;

    extern __shared__ char shared_memory[];
    auto &shared_storage = *reinterpret_cast<SharedStorage *>(shared_memory);

    int *tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr + partition_idx * TileSchedulerMetaDataSize;
    int4 tile_scheduler_metadata = __ldg(reinterpret_cast<int4 *>(tile_scheduler_metadata_ptr));
    int begin_idx = tile_scheduler_metadata.x;
    int begin_seqlen = tile_scheduler_metadata.y;
    int end_idx = tile_scheduler_metadata.z;
    int end_seqlen = tile_scheduler_metadata.w;
    if (begin_idx >= params.b) return;
    int begin_n_split_idx = __ldg(tile_scheduler_metadata_ptr + 4);

#pragma unroll 1
    for (int batch_id = begin_idx; batch_id <= end_idx; ++batch_id) {
        const int n_split_idx = batch_id == begin_idx ? begin_n_split_idx : 0;
        const int seqlen_k = __ldg(params.cu_seqlens_k + batch_id);
        const int n_block_min = batch_id == begin_idx ? begin_seqlen / kBlockN : 0;
        const int n_block_max = batch_id == end_idx ? cute::ceil_div(end_seqlen, kBlockN) : cute::ceil_div(seqlen_k, kBlockN);
        const bool NoSplit = n_block_min == 0 && n_block_max == cute::ceil_div(seqlen_k, kBlockN);
        if (batch_id > begin_idx) {
            __syncthreads();
        }
        if (!NoSplit) {
            flash::compute_attn_1rowblock_splitkv_mla_ws<Kernel_traits, Is_causal, true>(params, batch_id, bidh, m_block, n_split_idx, seqlen_k, n_block_min, n_block_max, shared_storage);
        } else {
            flash::compute_attn_1rowblock_splitkv_mla_ws<Kernel_traits, Is_causal, false>(params, batch_id, bidh, m_block, n_split_idx, seqlen_k, n_block_min, n_block_max, shared_storage);
        }
    }
}

} // namespace flash
