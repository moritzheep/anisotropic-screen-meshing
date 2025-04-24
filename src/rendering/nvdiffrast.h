#pragma once
#ifndef NVDIFFRAST_TORCH
#define NVDIFFRAST_TORCH
// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <nvdiffrast/torch/torch_common.inl>
#include <nvdiffrast/torch/torch_types.h>
#include <tuple>
// #include <nvdiffrast/common/texture.h>

//------------------------------------------------------------------------
// Op prototypes. Return type macros for readability.

#define OP_RETURN_T     torch::Tensor
#define OP_RETURN_TT    std::tuple<torch::Tensor, torch::Tensor>
#define OP_RETURN_TTT   std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
#define OP_RETURN_TTTT  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
#define OP_RETURN_TTV   std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor> >
#define OP_RETURN_TTTTV std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<torch::Tensor> >

OP_RETURN_TT        rasterize_fwd_cuda                  (RasterizeCRStateWrapper& stateWrapper, torch::Tensor pos, torch::Tensor tri, std::tuple<int, int> resolution, torch::Tensor ranges, int peeling_idx);
OP_RETURN_TT        rasterize_fwd_gl                    (RasterizeGLStateWrapper& stateWrapper, torch::Tensor pos, torch::Tensor tri, std::tuple<int, int> resolution, torch::Tensor ranges, int peeling_idx);
OP_RETURN_T         rasterize_grad                      (torch::Tensor pos, torch::Tensor tri, torch::Tensor out, torch::Tensor dy);
OP_RETURN_T         rasterize_grad_db                   (torch::Tensor pos, torch::Tensor tri, torch::Tensor out, torch::Tensor dy, torch::Tensor ddb);
OP_RETURN_TT        interpolate_fwd                     (torch::Tensor attr, torch::Tensor rast, torch::Tensor tri);
OP_RETURN_TT        interpolate_fwd_da                  (torch::Tensor attr, torch::Tensor rast, torch::Tensor tri, torch::Tensor rast_db, bool diff_attrs_all, std::vector<int>& diff_attrs_vec);
OP_RETURN_TT        interpolate_grad                    (torch::Tensor attr, torch::Tensor rast, torch::Tensor tri, torch::Tensor dy);
OP_RETURN_TTT       interpolate_grad_da                 (torch::Tensor attr, torch::Tensor rast, torch::Tensor tri, torch::Tensor dy, torch::Tensor rast_db, torch::Tensor dda, bool diff_attrs_all, std::vector<int>& diff_attrs_vec);
TextureMipWrapper   texture_construct_mip               (torch::Tensor tex, int max_mip_level, bool cube_mode);
OP_RETURN_T         texture_fwd                         (torch::Tensor tex, torch::Tensor uv, int filter_mode, int boundary_mode);
OP_RETURN_T         texture_fwd_mip                     (torch::Tensor tex, torch::Tensor uv, torch::Tensor uv_da, torch::Tensor mip_level_bias, TextureMipWrapper mip_wrapper, std::vector<torch::Tensor> mip_stack, int filter_mode, int boundary_mode);
OP_RETURN_T         texture_grad_nearest                (torch::Tensor tex, torch::Tensor uv, torch::Tensor dy, int filter_mode, int boundary_mode);
OP_RETURN_TT        texture_grad_linear                 (torch::Tensor tex, torch::Tensor uv, torch::Tensor dy, int filter_mode, int boundary_mode);
OP_RETURN_TTV       texture_grad_linear_mipmap_nearest  (torch::Tensor tex, torch::Tensor uv, torch::Tensor dy, torch::Tensor uv_da, torch::Tensor mip_level_bias, TextureMipWrapper mip_wrapper, std::vector<torch::Tensor> mip_stack, int filter_mode, int boundary_mode);
OP_RETURN_TTTTV     texture_grad_linear_mipmap_linear   (torch::Tensor tex, torch::Tensor uv, torch::Tensor dy, torch::Tensor uv_da, torch::Tensor mip_level_bias, TextureMipWrapper mip_wrapper, std::vector<torch::Tensor> mip_stack, int filter_mode, int boundary_mode);
TopologyHashWrapper antialias_construct_topology_hash   (torch::Tensor tri);
OP_RETURN_TT        antialias_fwd                       (torch::Tensor color, torch::Tensor rast, torch::Tensor pos, torch::Tensor tri, TopologyHashWrapper topology_hash);
OP_RETURN_TT        antialias_grad                      (torch::Tensor color, torch::Tensor rast, torch::Tensor pos, torch::Tensor tri, torch::Tensor dy, torch::Tensor work_buffer);

//----------------------------------------------------------------------------
// Rasterize.
//----------------------------------------------------------------------------

using namespace torch::autograd;

// Macro for 'None' return type
#define NoneTensor torch::Tensor()

class NVDiffRast
{
private:
    struct _RasterizeFunc : public Function<_RasterizeFunc>
    {
        static inline std::vector<torch::Tensor> forward(AutogradContext *ctx, RasterizeCRStateWrapper& stateWrapper, torch::Tensor pos, torch::Tensor tri, std::tuple<int, int> resolution, torch::Tensor ranges, bool grad_db, int peeling_idx)
        {
            // Overloading: RasterizeCRStateWrapper means we want to use CUDA
            auto result = rasterize_fwd_cuda(stateWrapper, pos, tri, resolution, ranges, peeling_idx);

            ctx->save_for_backward({ pos, tri, std::get<0>(result) });
            ctx->saved_data["grad_db"] = grad_db;

            return { std::get<0>(result), std::get<1>(result) };
        }

        static inline std::vector<torch::Tensor> forward(AutogradContext *ctx, RasterizeGLStateWrapper& stateWrapper, torch::Tensor pos, torch::Tensor tri, std::tuple<int, int> resolution, torch::Tensor ranges, bool grad_db, int peeling_idx)
        {
            // Overloading: RasterizeGLStateWrapper means we want to use OpenGL
            auto result = rasterize_fwd_gl(stateWrapper, pos, tri, resolution, ranges, peeling_idx);

            ctx->save_for_backward({ pos, tri, std::get<0>(result) });
            ctx->saved_data["grad_db"] = grad_db;

            //return result;
            return { std::get<0>(result), std::get<1>(result) };        
        }

        static inline std::vector<torch::Tensor> backward(AutogradContext *ctx, std::vector<torch::Tensor>& grads_output)
        {
            auto saved = ctx->get_saved_variables();
            auto pos = saved[0];
            auto tri = saved[1];
            auto out = saved[2];

            torch::Tensor g_pos;

            if(ctx->saved_data["grad_db"].toBool())
            {
                g_pos = rasterize_grad_db(pos, tri, out, grads_output[0], grads_output[1]);
            }
            else
            {
                g_pos = rasterize_grad(pos, tri, out, grads_output[0]);
            }

            return { NoneTensor, g_pos, NoneTensor, NoneTensor, NoneTensor, NoneTensor, NoneTensor };
        }
    };

    struct _InterpolateFunc : public Function<_InterpolateFunc>
    {
        static inline std::vector<torch::Tensor> forward(AutogradContext *ctx, torch::Tensor attr, torch::Tensor rast, torch::Tensor tri)
        {
            auto result = interpolate_fwd(attr, rast, tri);
            ctx->save_for_backward({attr, rast, tri});

            return { std::get<0>(result), std::get<1>(result) };
        }

        static inline std::vector<torch::Tensor> backward(AutogradContext *ctx, std::vector<torch::Tensor>& grads_output)
        {
            auto saved = ctx->get_saved_variables();
            auto result = interpolate_grad(saved[0], saved[1], saved[2], grads_output[0]);

            return { std::get<0>(result), std::get<1>(result), NoneTensor };
        }
    };

public:
    static inline std::vector<torch::Tensor> rasterize(RasterizeCRStateWrapper& stateWrapper, torch::Tensor pos, torch::Tensor tri, std::tuple<int, int> resolution, torch::Tensor ranges = torch::empty({0, 2}, torch::TensorOptions().dtype(torch::kInt32)), bool grad_db = true)
    {
        return _RasterizeFunc::apply(stateWrapper, pos, tri, resolution, ranges, grad_db, -1);
    }

    static inline std::vector<torch::Tensor> rasterize(RasterizeGLStateWrapper& stateWrapper, torch::Tensor pos, torch::Tensor tri, std::tuple<int, int> resolution, torch::Tensor ranges = torch::empty({0, 2}, torch::TensorOptions().dtype(torch::kInt32)), bool grad_db = false)
    {
        return _RasterizeFunc::apply(stateWrapper, pos, tri, resolution, ranges, false, -1);
    }

    static inline std::vector<torch::Tensor> interpolate(torch::Tensor attr, torch::Tensor rast, torch::Tensor tri, torch::Tensor rast_db = NoneTensor, bool diff_attrs_all = false)
    {
        return _InterpolateFunc::apply(attr, rast, tri);
    }
};

#endif
