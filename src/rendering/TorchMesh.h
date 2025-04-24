#pragma once

#include <torch/torch.h>
#include <pmp/SurfaceMesh.h>
// #include "../Utility.h"

struct TorchMesh
{
private:
    TorchMesh(torch::Tensor pos, torch::Tensor tris) : positions(pos), triangles(tris)
    {

    }

public:
    torch::Tensor positions, triangles;

    TorchMesh()
    {

    }

    explicit TorchMesh(const pmp::SurfaceMesh& mesh)
    {
        update_vertices(mesh);
        update_triangles(mesh);
    }

    TorchMesh to(torch::DeviceType device)
    {
        return TorchMesh(positions.to(device), triangles.to(device));
    }

    void to_(torch::DeviceType device)
    {
        positions = positions.to(device);
        triangles = triangles.to(device);
    }

    void update_vertices(const pmp::SurfaceMesh& mesh)
    {
        positions = torch::empty({ mesh.n_vertices(), 4 }).contiguous();

        auto accPositions = positions.accessor<float, 2>();

        for (auto v : mesh.vertices())
        {
            accPositions[v.idx()][0] = mesh.position(v)[0];
            accPositions[v.idx()][1] = mesh.position(v)[1];
            accPositions[v.idx()][2] = mesh.position(v)[2];
            accPositions[v.idx()][3] = 1.;
        }
    }

    void update_triangles(const pmp::SurfaceMesh& mesh)
    {
        triangles = torch::empty({ mesh.n_faces(), 3 }).to(torch::kInt).contiguous();

        auto accTriangles = triangles.accessor<int, 2>();

        int i;

        for (auto f : mesh.faces())
        {
            i = 0;

            for (auto v : mesh.vertices(f))
            {
                accTriangles[f.idx()][i++] = v.idx();
            }
        }
    }
};