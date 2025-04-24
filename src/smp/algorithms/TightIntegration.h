#pragma once

#include <pmp/SurfaceMesh.h>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include "ScreenDifferentialGeometry.h"
#include "../../rendering/TorchMesh.h"
#include "../../rendering/Renderer.h"

namespace pmp
{
	template <typename Camera>
	struct IntegrationPostProcessor
	{
		template <typename T>
		constexpr T operator()(const T& x)
		{
			return x;
		}
	};

	template <>
	struct IntegrationPostProcessor<Perspective>
	{
		template <typename T>
		constexpr T operator()(const T& x)
		{
			return std::exp(-x);
		}
	};

	template <typename T, typename Camera = Orthographic>
	class Integration
	{
	private:
		SurfaceMesh& mesh_;
		ScreenDifferentialGeometry<Camera> geo_;
		cv::Mat normals_, mask_;
		Camera projection_;
	
		T cotan_weight(Halfedge h) const
		{
			return std::clamp<T>(geo_.cotan_weight(h), 0.0524, 19.1); // 19.1 = 3degrees
		}

	public:
		Integration(SurfaceMesh& mesh, const cv::Mat& normals, const cv::Mat& mask = cv::Mat(), Camera camera = Camera()) : mesh_(mesh), geo_(mesh, normals, mask, camera), normals_(normals), mask_(mask), projection_(camera)
		{

		}

		T run()
		{
			mesh_.garbage_collection();
			geo_.update();

			TorchMesh tmesh(mesh_);
			tmesh.to_(torch::kCUDA);

			int height = normals_.rows;
			int width = normals_.cols;

			tmesh.positions.index_put_({ "...", 0 }, 2. * (tmesh.positions.index({ "...", 0 }) + 0.5) / static_cast<float>(width) - 1.);
            tmesh.positions.index_put_({ "...", 1 }, 2. * (tmesh.positions.index({ "...", 1 }) + 0.5) / static_cast<float>(height) - 1.);
            tmesh.positions.index_put_({ "...", 2 }, 0);
            tmesh.positions.index_put_({ "...", 3 }, 1);

            // Triangle ids give one triangle for each pixel
            Renderer& renderer = Renderer::getInstance();

            // Only transfer those ids that concern foreground pixels
            torch::Tensor tIds = renderer.render(tmesh.positions, tmesh.triangles, std::make_tuple(height, width))[0].index({ 0, "...", 3 }).to(torch::kInt).cpu();
			auto accIds = tIds.accessor<int, 2>();

			auto fcoeff = mesh_.add_face_property<Vector<T, 4>>("f:coeff", Vector<T, 4>(0));
			Vector<T, 4> tmp(0, 0, 0, 1);

			Vector<T, 3> du(projection_.du());
			Vector<T, 3> dv(projection_.dv());
			T scale = 1. / std::sqrt(norm(cross(du, dv)));

			Vector<T, 3> dz;
			Vector<T, 3> normal;
			T w;

			for (int v = 0; v != height; ++v)
			{
				for (int u = 0; u != width; ++u)
				{
					const int& id = accIds[v][u];
					
					if (id > 0 && (mask_.empty() || mask_.at<uchar>(v, u) > 127))
					{
						Face f(id - 1);

						normal[0] = normals_.at<cv::Vec3f>(v, u)[2];
						normal[1] = normals_.at<cv::Vec3f>(v, u)[1];
						normal[2] = normals_.at<cv::Vec3f>(v, u)[0];

						dz = Vector<T, 3>(projection_.dz(u, v));

						w = dot(dz, normal);

						// Ignore extremely tilted normals
						if (w > -ScreenDifferentialGeometry<Camera>::cos_max_angle)
						{
							continue;
						}

						tmp[0] = scale * dot(du, normal) * w;
						tmp[1] = scale * dot(dv, normal) * w;
						tmp[2] = scale * w * w;

						fcoeff[f] += tmp;
					}
				}
			}

			int n = mesh_.n_vertices();

			Eigen::SparseMatrix<T> L(n, n);
			Eigen::Vector<T, -1> b = Eigen::Vector<T, -1>::Zero(n);

			 // Assembly:
			std::vector<Eigen::Triplet<T>> coefficients; // list of non-zeros coefficients
			coefficients.reserve(4 * mesh_.n_edges());

			Vertex v0, v1, v2;
			Halfedge h0, h1;
			Face f0, f1;

			T trace = 0;
			T l00, l01, l11;
			T b0, b1;
			T cot;

			T r0n, r1n, r2n;

			Vector<T, 2> dr;


			// #ToDo: This can be parallelized since we have a fixed number of coefficients per edge
			for (auto e : mesh_.edges())
			{
				h0 = mesh_.halfedge(e, 0);
				h1 = mesh_.halfedge(e, 1);

				v0 = mesh_.to_vertex(h0);
				v1 = mesh_.to_vertex(h1);

				f0 = mesh_.face(h0);
				f1 = mesh_.face(h1);

				dr = Vector<T, 2>(mesh_.position(v0)[0] - mesh_.position(v1)[0], mesh_.position(v0)[1] - mesh_.position(v1)[1]);

				l00 = 0; //std::numeric_limits<T>::epsilon(); // Avoid zeros
				b0 = 0;

				if (f0.is_valid() && fcoeff[f0][3] > 0.5)
				{
					cot = cotan_weight(h0);

					Vector<T, 4> coeff = fcoeff[f0];
					coeff /= fcoeff[f0][3];
					
					l00 += cot * coeff[2];
					b0 += cot * (coeff[0] * dr[0] + coeff[1] * dr[1]);
				}
				else // A little regularization on the boundaries
				{
					l00 += 10e-5;
				}

				if (f1.is_valid() && fcoeff[f1][3] > 0.5)
				{
					cot = cotan_weight(h1);

					Vector<T, 4> coeff = fcoeff[f1];
					coeff /= fcoeff[f1][3];
					
					l00 += cot * coeff[2];
					b0 += cot * (coeff[0] * dr[0] + coeff[1] * dr[1]);
				}
				else // A little regularization on the boundaries
				{
					l00 += 10e-5;
				}

				l01 = -l00;
				l11 = l00;

				b1 = -b0;

				coefficients.emplace_back(v0.idx(), v0.idx(), l00);
				coefficients.emplace_back(v0.idx(), v1.idx(), l01);
				
				coefficients.emplace_back(v1.idx(), v1.idx(), l11);
				coefficients.emplace_back(v1.idx(), v0.idx(), l01);

				b(v0.idx()) += b0;
				b(v1.idx()) += b1;

				trace += l00 + l11;
			}

			// Build Linear System
			L.setFromTriplets(coefficients.begin(), coefficients.end());

			T reg = std::numeric_limits<T>::epsilon();

			Eigen::ConjugateGradient<Eigen::SparseMatrix<T>> cg;
			cg.compute(L);
			Eigen::Vector<T, -1> depth = cg.solve(b);

			depth.array() -= depth.mean();

			IntegrationPostProcessor<Camera> processor;

			for (auto v : mesh_.vertices())
			{
				mesh_.position(v)[2] = processor(depth(v.idx()));
			}

			std::cout << "Integration: Residual is " << cg.error() << " after " << cg.iterations() << " iterations\n";

			return cg.error();
		}
	};
}