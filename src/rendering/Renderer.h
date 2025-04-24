#pragma once
#ifndef HEEP_RENDERER
#define HEEP_RENDERER

#include <tuple>
#include <torch/torch.h>
#include "nvdiffrast.h"

// #include "Session.h"

// This class is only intended to attach the render buffer provided by nvdiffrast onto the data.dome_ data structure
class Renderer
{
private:
    static inline Renderer* _renderer;

    RasterizeGLStateWrapper _wrapper;

    Renderer() : _wrapper(true, true, 0)
    {

    }

public:
    static Renderer& getInstance()
    {
        if (!_renderer)
            _renderer = new Renderer();
            
        return *_renderer;
    }

    // Render positions and triangles assuming they are already in clip space
    std::vector<torch::Tensor> render(const torch::Tensor& positions, const torch::Tensor& triangles, const std::tuple<int, int> resolution)
    {
        return NVDiffRast::rasterize(_wrapper, positions.unsqueeze(0), triangles, resolution);
    }
};

#endif
