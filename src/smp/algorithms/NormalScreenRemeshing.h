// Copyright 2011-2020 the Polygon Mesh Processing Library developers.
// Distributed under a MIT-style license, see LICENSE.txt for details.

#pragma once
#include <unordered_set>
#include <random>
#include "../structures/updatable_queue.h"

#include <memory>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "ScreenDifferentialGeometry.h"

namespace pmp 
{
    using Vector2S = Eigen::Matrix<Scalar, 2, 1>;
    using Vector3S = Eigen::Matrix<Scalar, 3, 1>;
    using Vector4S = Eigen::Matrix<Scalar, 4, 1>;
    using Matrix2S = Eigen::Matrix<Scalar, 2, 2>;
    using Matrix3S = Eigen::Matrix<Scalar, 3, 3>;
    using Matrix4S = Eigen::Matrix<Scalar, 4, 4>;

    using EigenOrthographic = projection::Orthographic<Vector3S>;
    using EigenPerspective = projection::Perspective<Vector3S>;

    using Frame = Eigen::Matrix<Scalar, 3, 2>;
    using Quadric3 = Eigen::Matrix<Scalar, 4, 4>;
    using Quadric2 = Eigen::Matrix<Scalar, 3, 3>;

    class DecimationOptions
    {
    private:
        IndexType vertexTarget_ = 0;
        Scalar errorThreshold_ = std::numeric_limits<Scalar>::max();
        Scalar anisotropy_ = 1.;

    public:
        DecimationOptions& setVertexTarget(IndexType target)
        {
            vertexTarget_ = target;
            return *this;
        }

        DecimationOptions& setErrorThreshold(Scalar threshold)
        {
            errorThreshold_ = threshold;
            return *this;
        }

        DecimationOptions& setAnisotropy(Scalar anisotropy)
        {
            anisotropy_ = anisotropy;
            return *this;
        }

        IndexType vertexTarget() const
        {
            return vertexTarget_;
        }

        Scalar errorThreshold() const
        {
            return errorThreshold_;
        }

        Scalar anisotropy() const
        {
            return anisotropy_;
        }
    };

    class RemeshingOptions
    {
    private:
        bool use_flips_ = true;
        bool use_move_ = true;
        int num_iterations_ = 10;
        Scalar alpha_ = 0.5;
        Scalar beta_ = 1;

    public:
        RemeshingOptions& enableFlips()
        {
            use_flips_ = true;
            return *this;
        }

        RemeshingOptions& disableFlips()
        {
            use_flips_ = false;
            return *this;
        }

        RemeshingOptions& setFlips(bool value)
        {
            use_flips_ = value;
            return *this;
        }

        RemeshingOptions& enableMove()
        {
            use_move_ = true;
            return *this;
        }

        RemeshingOptions& disableMove()
        {
            use_move_ = false;
            return *this;
        }

        RemeshingOptions& setMove(bool value)
        {
            use_move_ = value;
            return *this;
        }

        RemeshingOptions& setIterations(int iteration)
        {
            num_iterations_ = iteration;
            return *this;
        }

        RemeshingOptions& setAlpha(Scalar alpha)
        {
            alpha_ = alpha;
            return *this;
        }

        RemeshingOptions& setBeta(Scalar beta)
        {
            beta_ = beta;
            return *this;
        }

        bool useFlips() const { return use_flips_; }
        bool useMove() const { return use_move_; }
        int iterations() const { return num_iterations_; }
        Scalar alpha() const { return alpha_; }
        Scalar beta() const { return beta_; }
    };

    template <typename Projection = EigenOrthographic>
    class NormalScreenRemeshing
    {
    public:

        // Works and converges. For some flips, however, the condition is violated either way
        class ScreenDelaunayCriterion
        {
        private:
            SurfaceMesh& mesh_;
            
            inline Scalar angle(Halfedge h) const
            {
                Vertex previous = mesh_.from_vertex(h);
                Vertex current = mesh_.to_vertex(h);
                Vertex next = mesh_.to_vertex(mesh_.next_halfedge(h));

                Vector2S t0(mesh_.position(previous)[0] - mesh_.position(current)[0], mesh_.position(previous)[1] - mesh_.position(current)[1]);
                Vector2S t1(mesh_.position(next)[0] - mesh_.position(current)[0], mesh_.position(next)[1] - mesh_.position(current)[1]);

                t0.normalize();
                t1.normalize();

                return std::acos(t0.dot(t1));
            }

        public:
            ScreenDelaunayCriterion(SurfaceMesh& mesh, NormalScreenRemeshing&) : mesh_(mesh) { }
            
            Scalar operator()(Edge e) const
            {
                if (mesh_.is_boundary(e))
                {
                    return -1;
                }
                else
                {
                    Halfedge h0 = mesh_.next_halfedge(mesh_.halfedge(e, 0));
                    Halfedge h1 = mesh_.next_halfedge(mesh_.halfedge(e, 1));

                    return angle(h0) + angle(h1) - M_PI; // No flip needed if angle sum is below PI
                }
            }
        };

        class LocalDelaunayCriterion
        {
        private:
            SurfaceMesh& mesh_;
            NormalScreenRemeshing& remeshing_;
            
            inline Scalar angle(Halfedge h, const Matrix2S metric) const
            {
                Vertex previous = mesh_.from_vertex(h);
                Vertex current = mesh_.to_vertex(h);
                Vertex next = mesh_.to_vertex(mesh_.next_halfedge(h));

                Vector2S t0(mesh_.position(previous)[0] - mesh_.position(current)[0], mesh_.position(previous)[1] - mesh_.position(current)[1]);
                Vector2S t1(mesh_.position(next)[0] - mesh_.position(current)[0], mesh_.position(next)[1] - mesh_.position(current)[1]);

                t0 /= std::sqrt(t0.dot(metric * t0));
                t1 /= std::sqrt(t1.dot(metric * t1));

                return std::acos(t0.dot(metric * t1));
            }

        public:
            LocalDelaunayCriterion(SurfaceMesh& mesh, NormalScreenRemeshing& remeshing) : mesh_(mesh), remeshing_(remeshing) { }
            
            Scalar operator()(Edge e) const
            {
                if (mesh_.is_boundary(e))
                {
                    return -1;
                }
                else
                {
                    Halfedge h0 = mesh_.next_halfedge(mesh_.halfedge(e, 0));
                    Halfedge h1 = mesh_.next_halfedge(mesh_.halfedge(e, 1));

                    Face f0 = mesh_.face(e, 0);
                    Face f1 = mesh_.face(e, 1);

                    Matrix2S metric = remeshing_.metric(f0) + remeshing_.metric(f1);

                    return angle(h0, metric) + angle(h1, metric) - M_PI; // No flip needed if angle sum is below PI
                }
            }
        };

        class AnisotropicProjectionCriterion
        {
        private:
            static constexpr Scalar epsilon = 10e-5;

            SurfaceMesh& mesh_;
            NormalScreenRemeshing& remeshing_;
            
            inline Scalar angle(Halfedge h, const Matrix2S metric) const
            {
                Vertex previous = mesh_.from_vertex(h);
                Vertex current = mesh_.to_vertex(h);
                Vertex next = mesh_.to_vertex(mesh_.next_halfedge(h));

                Vector2S t0(mesh_.position(previous)[0] - mesh_.position(current)[0], mesh_.position(previous)[1] - mesh_.position(current)[1]);
                Vector2S t1(mesh_.position(next)[0] - mesh_.position(current)[0], mesh_.position(next)[1] - mesh_.position(current)[1]);

                t0 /= std::sqrt(t0.dot(metric * t0));
                t1 /= std::sqrt(t1.dot(metric * t1));

                return std::acos(t0.dot(metric * t1));
            }

        public:
            AnisotropicProjectionCriterion(SurfaceMesh& mesh, NormalScreenRemeshing& remeshing) : mesh_(mesh), remeshing_(remeshing) { }
            
            Scalar operator()(Edge e) const
            {
                if (mesh_.is_boundary(e))
                {
                    return -1;
                }
                else
                {
                    Halfedge h0 = mesh_.halfedge(e, 0);
                    Halfedge h1 = mesh_.halfedge(e, 1);

                    Vertex v0 = mesh_.to_vertex(h0);
                    Vertex v1 = mesh_.to_vertex(h1);
                    Vertex vv0 = mesh_.to_vertex(mesh_.next_halfedge(h0));
                    Vertex vv1 = mesh_.to_vertex(mesh_.next_halfedge(h1));

                    Vector3S t0 = Vector3S(mesh_.position(vv0)[0] - mesh_.position(v0)[0], mesh_.position(vv0)[1] - mesh_.position(v0)[1], 0);
                    t0.z() = t0.topRows<2>().dot(remeshing_.equadric_[e] * t0.topRows<2>());

                    Vector3S t1 = Vector3S(mesh_.position(vv1)[0] - mesh_.position(v0)[0], mesh_.position(vv1)[1] - mesh_.position(v0)[1], 0);
                    t1.z() = t1.topRows<2>().dot(remeshing_.equadric_[e] * t1.topRows<2>());

                    // We have found the normal of the lifted plane. We check whether z(v1) is above or below that plane. n_x * x + n_y * y = -n_z * z => z = - (n_x * x + n_y * y) / n_z. We flip if the current edge is above
                    Vector3S normal = t0.cross(t1);

                    Vector2S t = Vector2S(mesh_.position(v1)[0] - mesh_.position(v0)[0], mesh_.position(v1)[1] - mesh_.position(v0)[1]);

                    return (t.dot(remeshing_.equadric_[e] * t) + normal.topRows<2>().dot(t) / normal.z());
                }
            }
        };

        NormalScreenRemeshing(SurfaceMesh& mesh, const cv::Mat& normals, const cv::Mat& mask = cv::Mat(), Projection projection = Projection()) : mesh_(mesh), projection_(projection), tmesh_(mesh), A_px(projection_.du().cross(projection_.dv()).norm()), l_px(std::sqrt(A_px))
        {
            tmesh_.to_(torch::kCUDA);
            setNormalMap(normals, mask);
            add_properties();
            update_boundary_quadrics();
        }

        NormalScreenRemeshing(SurfaceMesh& mesh, const cv::Mat& normals, Projection projection = Projection()) : mesh_(mesh), projection_(projection), tmesh_(mesh), A_px(projection_.du().cross(projection_.dv()).norm()), l_px(std::sqrt(A_px))
        {
            tmesh_.to_(torch::kCUDA);
            setNormalMap(normals);
            add_properties();
            update_boundary_quadrics();
        }

        ~NormalScreenRemeshing()
        {
            remove_properties();
        }

        const Quadric3& quadric(Face f) const
        {
            return fquadric_[f];
        }

        const Quadric3& quadric(Vertex v) const
        {
            return vquadric_[v];
        }

        Vector3S normal(Face f) const
        {
            return fnormal_[f].normalized();
        }

        // Run edge- and vertex-alignment
        void remeshing(const RemeshingOptions& options)
        {
            alpha_ = options.alpha();
            beta_ = options.beta();

            for (int i = 0; i != options.iterations(); ++i)
            {
                if (options.useFlips())
                    flip_edges<AnisotropicProjectionCriterion>();

                if (options.useMove())
                    optimizePositions();

                alpha_ *= beta_;
            }
        }

        // Remove duplicate vertices
        void remove_duplicates(Scalar threshold = std::numeric_limits<Scalar>::epsilon())
        {
            bool collapsed = true;
            Vertex v0, v1;
            Halfedge h0, h1;
            bool ok01, ok10;
            Scalar length;
            Edge edge;

            Vector2S uv;

            while(collapsed)
            {
                collapsed = false;
                updatable_queue<Edge, Scalar> queue(mesh_);

                for (auto e : mesh_.edges())
                {
                    v0 = mesh_.vertex(e, 0);
                    v1 = mesh_.vertex(e, 1);
                    
                    length = std::max(
                        std::abs(mesh_.position(v0)[0] - mesh_.position(v1)[0]),
                        std::abs(mesh_.position(v0)[1] - mesh_.position(v1)[1]));

                    if (length < threshold)
                        queue.push(e, length);
                }

                while(!queue.empty())
                {
                    edge = queue.top().first; 
                    queue.pop();

                    h0 = mesh_.halfedge(edge, 0);
                    h1 = mesh_.halfedge(edge, 1);

                    v0 = mesh_.to_vertex(h0);
                    v1 = mesh_.to_vertex(h1);

                    ok10 = mesh_.is_collapse_ok(h0) && (!mesh_.is_boundary(v1) || mesh_.is_boundary(edge));
                    ok01 = mesh_.is_collapse_ok(h1) && (!mesh_.is_boundary(v0) || mesh_.is_boundary(edge));

                    if (ok10)
                    {
                        // Remove to be deleted edges from queue
                        if (!mesh_.is_boundary(h0))
                        {
                            queue.remove(mesh_.edge(mesh_.prev_halfedge(h0)));
                        }

                        if (!mesh_.is_boundary(h1))
                        {
                            queue.remove(mesh_.edge(mesh_.next_halfedge(h0)));
                        }

                        mesh_.collapse(h0);

                        // Update adjacent edges
                        for (auto h : mesh_.halfedges(v0))
                        {
                            auto v = mesh_.to_vertex(h);

                            length = std::max(
                                std::abs(mesh_.position(v0)[0] - mesh_.position(v)[0]),
                                std::abs(mesh_.position(v0)[1] - mesh_.position(v)[1]));

                            if (length < threshold)
                                queue.push(edge, length);
                        }

                        collapsed = true;
                    }
                    else if (ok10)
                    {
                        mesh_.collapse(h1);

                        // Remove to be deleted edges from queue
                        if (!mesh_.is_boundary(h0))
                        {
                            queue.remove(mesh_.edge(mesh_.next_halfedge(h0)));
                        }

                        if (!mesh_.is_boundary(h1))
                        {
                            queue.remove(mesh_.edge(mesh_.prev_halfedge(h0)));
                        }

                        // Update adjacent edges
                        for (auto h : mesh_.halfedges(v1))
                        {
                            auto v = mesh_.to_vertex(h);

                            length = std::max(
                                std::abs(mesh_.position(v1)[0] - mesh_.position(v)[0]),
                                std::abs(mesh_.position(v1)[1] - mesh_.position(v)[1]));

                            if (length < threshold)
                                queue.push(edge, length);
                        }

                        collapsed = true;
                    }
                }

                mesh_.garbage_collection();
            }
        }


        bool decimate(const DecimationOptions& options)
        {
            bool collapsed = false;

            updateFramesAndQuadrics<true, false>();

            updatable_queue<Edge, OptimalEdgeToVertex> queue(mesh_);
            queue.reserve(mesh_.n_edges()); // Avoid reallocations
            Matrix3S quadric;
            Matrix2S metric;

            Eigen::Matrix<Scalar, 4, 3> projection = Eigen::Matrix<Scalar, 4, 3>::Zero(); projection(3, 2) = 1.;
            Eigen::Matrix<Scalar, 3, 2> line = Eigen::Matrix<Scalar, 3, 2>::Zero(); line(2, 1) = 1.;
            Matrix3S translation = Matrix3S::Identity();
   
            Vertex v0, v1;
            Vector2S uv0, uv1;
            Vector3S uv;
            uv.z() = 1.;

            OptimalEdgeToVertex opt;

            // Heapifying while adding is O(n*log(n)), adding and then heapifying is O(n)
            for (auto e : mesh_.edges())
            {
                auto v0 = mesh_.vertex(e, 0);
                auto v1 = mesh_.vertex(e, 1);
                
                if (isCollapseCandidate(e))
                {
                    opt = optimalCollapse(e);

                    if (opt.error < options.errorThreshold())
                    {
                        queue.lazy_push(e, opt);
                    }
                }
            }

            // Heapifying while adding is O(n*log(n)), adding and then heapifying is O(n)
            queue.heapify();

            Edge e;
            Halfedge h0, h1;
            bool ok01, ok10, flip;

            while(mesh_.n_vertices() > options.vertexTarget() && !queue.empty())
            {
                std::tie(e, opt) = queue.top(); queue.pop();

                h0 = mesh_.halfedge(e, 0);
                h1 = mesh_.halfedge(e, 1);

                v0 = mesh_.to_vertex(h0);
                v1 = mesh_.to_vertex(h1);

                // Exclude collapses that cause triangles to flip
                flip = collapseCausesFlip(e, opt.uv);

                // Is collapsing v1 -> v0 allowed?
                // - Topology check
                // - We don't want to collapse away from boundary vertices
                
                ok10 = !flip && mesh_.is_collapse_ok(h0) && (!mesh_.is_boundary(v1) || mesh_.is_boundary(e));
                ok01 = !flip && mesh_.is_collapse_ok(h1) && (!mesh_.is_boundary(v0) || mesh_.is_boundary(e));

                if (ok10) // Collapse v1 into v0
                {
                    // Remove edge that are about to be deleted from the queue
                    if (!mesh_.is_boundary(h0))
                    {
                        queue.remove(mesh_.edge(mesh_.prev_halfedge(h0)));
                    }

                    if (!mesh_.is_boundary(h1))
                    {
                        queue.remove(mesh_.edge(mesh_.next_halfedge(h1)));
                    }

                    collapse(h0, opt.uv);
                    collapsed = true;

                    // Update quadrics
                    for (auto h : mesh_.halfedges(v0))
                    {
                        if (isCollapseCandidate(mesh_.edge(h)))
                        {
                            opt = optimalCollapse(mesh_.edge(h));

                            if (opt.error < options.errorThreshold())
                            {
                                queue.update_or_push(mesh_.edge(h), opt);
                            }
                            else
                                queue.remove(mesh_.edge(h));
                        }
                        else
                            queue.remove(mesh_.edge(h));
                    }
                }
                else if (ok01) // Collapse v0 into v1
                {
                    // Remove edge that are about to be deleted from the queue
                    if (!mesh_.is_boundary(h0))
                    {
                        queue.remove(mesh_.edge(mesh_.next_halfedge(h0)));
                    }

                    if (!mesh_.is_boundary(h1))
                    {
                        queue.remove(mesh_.edge(mesh_.prev_halfedge(h1)));
                    }

                    collapse(h1, opt.uv);
                    collapsed = true;

                    // Update quadrics
                    for (auto h : mesh_.halfedges(v1))
                    {
                        if (isCollapseCandidate(mesh_.edge(h)))
                        {
                            opt = optimalCollapse(mesh_.edge(h));

                            if (opt.error < options.errorThreshold())
                            {
                                queue.update_or_push(mesh_.edge(h), opt);
                            }
                            else
                                queue.remove(mesh_.edge(h));
                        }
                        else
                            queue.remove(mesh_.edge(h));
                    }
                }
            }

            // Free memory of deleted vertices, edges and faces
            mesh_.garbage_collection();

            // Update positions and triangles and transfer to GPU
            tmesh_ = TorchMesh(mesh_);
            tmesh_.to_(torch::kCUDA);

            // cv::waitKey(-1);

            return collapsed;
        }

    private:

        // Datastructure to store collapse candidates
        struct OptimalEdgeToVertex
        {
            Vector2S uv;
            Scalar error;

            OptimalEdgeToVertex() { }

            OptimalEdgeToVertex(Vector2S uv_, Scalar error_) : uv(uv_), error(error_) { }

            bool operator<(const OptimalEdgeToVertex& other) const
            {
                return error > other.error;
            }

            bool operator<=(const OptimalEdgeToVertex& other) const
            {
                return error >= other.error;
            }

            bool operator>(const OptimalEdgeToVertex& other) const
            {
                return error < other.error;
            }

            bool operator>=(const OptimalEdgeToVertex& other) const
            {
                return error <= other.error;
            }
        };

        // Shift the quadric Q'(x) = Q(x-d)
        template <typename QuadricDerived, typename VectorDerived>
        static constexpr Eigen::Matrix<typename QuadricDerived::Scalar, QuadricDerived::RowsAtCompileTime, QuadricDerived::RowsAtCompileTime> translate(const Eigen::MatrixBase<QuadricDerived>& Q, const Eigen::MatrixBase<VectorDerived>& d)
        {
            constexpr int dim = QuadricDerived::RowsAtCompileTime - 1;
            Eigen::Matrix<typename QuadricDerived::Scalar, dim + 1, dim + 1> Qt = Q;

            // c' = c + (Ad, d) - 2(b, d)
            Qt(dim, dim) += d.dot(Q.template topLeftCorner<dim, dim>() * d) - 2. * Q.template topRightCorner<dim, 1>().dot(d);
            
            // Clamp 
            if (Qt(dim, dim) < 0.)
            {
                Qt(dim, dim) = 0.;
            }

            // b' = b - Ad
            Qt.template topRightCorner<dim, 1>() -= Q.template topLeftCorner<dim, dim>() * d;

            // Symmetrize
            Qt.template bottomLeftCorner<1, dim>() = Qt.template topRightCorner<dim, 1>().transpose();

            return Qt;
        }

        void update_boundary_quadrics()
        {
            Vector2S n;
            Vector2S b;
            Matrix2S M;

            for (auto v : mesh_.vertices())
            {
                bquadric_[v].setZero();

                if (mesh_.is_boundary(v))
                {
                    b.setZero();

                    for (auto h : mesh_.halfedges(v))
                    {
                        if (mesh_.is_boundary(mesh_.edge(h)))
                        {
                            b.x() += mesh_.position(mesh_.to_vertex(h))[0];
                            b.y() += mesh_.position(mesh_.to_vertex(h))[1];
                        }
                    }

                    // Normal vector of the outline (pointing away from the curvature center)
                    n.x() = 2. * mesh_.position(v)[0] - b.x();
                    n.y() = 2. * mesh_.position(v)[1] - b.y();

                    // Barycenter of the outline
                    b = -n / 2.;

                    M = n * n.transpose();

                    bquadric_[v].topLeftCorner<2, 2>() = M;
                    bquadric_[v].topRightCorner<2, 1>() = -M * b;
                    bquadric_[v](2, 2) = b.transpose() * M * b;

                    bquadric_[v].bottomLeftCorner<1, 2>() = bquadric_[v].topRightCorner<2, 1>().transpose();
                    bquadric_[v] *= edge_weight;
                }
            }
        }

        static constexpr Scalar degrees = M_PI / 180.;
        static constexpr Scalar cos_max_angle = std::cos(85. * degrees);
        static constexpr Scalar sin_max_angle = std::sqrt(1. - cos_max_angle * cos_max_angle);
        static constexpr Scalar edge_weight = 0.5;

        // Normals should not be perpendicular to the viewing  direction
        Vector3S clamp_normal(const Vector3S& normal, const Vector2S& uv) const
        {
            Vector3S ray = projection_.dz(uv.x(), uv.y()).normalized();

            if (ray.dot(normal) < -cos_max_angle)
            {
                return normal;
            }
            else
            {
                return -cos_max_angle * ray + std::sqrt(1. - cos_max_angle * cos_max_angle) * (normal - ray.dot(normal) * ray).normalized();
            }
        }

        bool isConvex(Edge e) const
        {
            std::array<Halfedge, 4> hs;

            hs[0] = mesh_.next_halfedge(mesh_.halfedge(e, 0));
            hs[1] = mesh_.next_halfedge(hs[0]);
            hs[2] = mesh_.next_halfedge(mesh_.halfedge(e, 1));
            hs[3] = mesh_.next_halfedge(hs[2]);

            Vector2S d, dn, dp;
            Halfedge hn, h, hp;

            for (int i = 0; i != 3; ++i)
            {
                h = hs[i];
                hn = hs[(i + 1) % 4];
                hp = hs[(i + 3) % 4];

                d = Vector3S(mesh_.position(mesh_.to_vertex(h))).topRows<2>() - Vector3S(mesh_.position(mesh_.from_vertex(h))).topRows<2>();
                dn = Vector3S(mesh_.position(mesh_.to_vertex(hn))).topRows<2>() - Vector3S(mesh_.position(mesh_.from_vertex(hn))).topRows<2>();
                dp = Vector3S(mesh_.position(mesh_.to_vertex(hp))).topRows<2>() - Vector3S(mesh_.position(mesh_.from_vertex(hp))).topRows<2>();

                d = Vector2S(d.y(), -d.x());

                if (d.dot(dn) * d.dot(dp) > -std::numeric_limits<Scalar>::epsilon())
                    return false;
            }

            return true;
        }

        Frame frame(Vector3S normal, const Vector2S& uv) const
        {
            Frame frame;

            // Parallel case
            frame.col(0) = projection_.du();
            frame.col(1) = projection_.dv();

            Scalar norm2 = normal.squaredNorm();

            if (norm2 > std::numeric_limits<Scalar>::epsilon()) // No valid normal => Treat as fronto-parallel
            {
                auto dz = projection_.dz(uv.x(), uv.y());

                // Flip if neccessary
                if (normal.dot(dz) > 0)
                {
                    normal *= -1.;
                }

                normal.normalize();

                Scalar n3 = std::min(dz.dot(normal), -cos_max_angle);
                frame -= dz * normal.transpose() * frame / n3;

                // frame(2, 0) = std::clamp(frame(2, 0), -std::abs(frame(0, 0) / cos_max_angle), +std::abs(frame(0, 0) / cos_max_angle));
                // frame(2, 1) = std::clamp(frame(2, 1), -std::abs(frame(1, 1) / cos_max_angle), +std::abs(frame(1, 1) / cos_max_angle));
            }

            return frame / l_px;
        }

        // Collapse Halfedge h and move vertex to uv
        void collapse(Halfedge h0, const Vector2S& uv)
        {
            auto v0 = mesh_.to_vertex(h0);
            auto v1 = mesh_.from_vertex(h0);

            Vector2S offset(uv.x() - mesh_.position(v0)[0], uv.y() - mesh_.position(v0)[1]);

            // Calculate new (boundary) Quadric (in place)
            vquadric_[v0] = translate(vquadric_[v0], -offset);
            bquadric_[v0] = translate(bquadric_[v0], -offset);

            offset = Vector2S(uv.x() - mesh_.position(v1)[0], uv.y() - mesh_.position(v1)[1]);
            vquadric_[v0] += translate(vquadric_[v1], -offset);
            bquadric_[v0] += translate(bquadric_[v1], -offset);

            mesh_.collapse(h0);
            mesh_.position(v0)[0] = uv.x();
            mesh_.position(v0)[1] = uv.y();
        }

        bool isCollapseCandidate(Edge e) const
        {
            auto v0 = mesh_.vertex(e, 0);
            auto v1 = mesh_.vertex(e, 1);

            return mesh_.is_boundary(e) || !(mesh_.is_boundary(v0) && mesh_.is_boundary(v1));
        }

        // Calculate optimal collapse for Edge e
        OptimalEdgeToVertex optimalCollapse(Edge e) const
        {
            auto v0 = mesh_.vertex(e, 0);
            auto v1 = mesh_.vertex(e, 1);
            
            Vector2S uv0(mesh_.position(v0)[0], mesh_.position(v0)[1]);
            Vector2S uv1(mesh_.position(v1)[0], mesh_.position(v1)[1]);
            Scalar s;

            Vector2S line = uv1 - uv0;
            // The optimization problem is Q_0(line * s) + Q_1(line * (1 - s)) = a * s^2 + 2 * b * s + c = 0 => s = -b / a
            Scalar a = line.dot(vquadric_[v0].topLeftCorner<2, 2>() * line) + line.dot(vquadric_[v1].topLeftCorner<2, 2>() * line);
            Scalar b = line.dot(vquadric_[v0].topRightCorner<2, 1>()) + line.dot(vquadric_[v1].topRightCorner<2, 1>()) - line.dot(vquadric_[v1].topLeftCorner<2, 2>() * line);
            Scalar c = vquadric_[v0](2, 2) + vquadric_[v1](2, 2) + line.dot(vquadric_[v1].topLeftCorner<2, 2>() * line) - 2. * line.dot(vquadric_[v1].topRightCorner<2, 1>());

            // Only collapse into boundary vertices
            if (mesh_.is_boundary(v0) && !mesh_.is_boundary(v1))
            {
                s = 0;
            }
            else if (mesh_.is_boundary(v1) && !mesh_.is_boundary(v0))
            {
                s = 1;
            }
            else if (a > std::numeric_limits<Scalar>::epsilon())
            {
                s = std::clamp<Scalar>(-b / a, 0, 1);
            }
            else // Parabola is almost flat/curved downwards
            {
                s = 0.5;
            }

            Scalar error;

            if (std::abs(uv0.x() - uv1.x()) < std::numeric_limits<Scalar>::epsilon() && std::abs(uv0.y() - uv1.y()) < std::numeric_limits<Scalar>::epsilon())
            {
                error = std::numeric_limits<Scalar>::lowest();
            }
            else
            {
                error = 2. * std::max<Scalar>((a * s + 2 * b) * s + c, 0.);
            }

            return OptimalEdgeToVertex(uv0 + s * line, error);
        }

        // Check if collaps causes a triangle to flip
        bool collapseCausesFlip(Edge e, const Vector2S& uv) const
        {
            auto v0 = mesh_.vertex(e, 0);
            auto v1 = mesh_.vertex(e, 1);

            Vertex vv0, vv1;
            Scalar orient;

            // Check for inversions
            for (auto h : mesh_.halfedges(v0))
            {
                if (!mesh_.is_boundary(h))
                {
                    vv0 = mesh_.to_vertex(h);
                    vv1 = mesh_.to_vertex(mesh_.next_halfedge(h));

                    if (vv0 != v1 && vv1 != v1)
                    {
                        // Calculate z component of flat normal
                        orient = (mesh_.position(vv0)[0] - uv[0]) * (mesh_.position(vv1)[1] - uv[1]) - (mesh_.position(vv0)[1] - uv[1]) * (mesh_.position(vv1)[0] - uv[0]);

                        if (orient < std::numeric_limits<Scalar>::min())
                        {
                            return true;
                        }
                    }
                }
            }

            for (auto h : mesh_.halfedges(v1))
            {
                if (!mesh_.is_boundary(h))
                {
                    vv0 = mesh_.to_vertex(h);
                    vv1 = mesh_.to_vertex(mesh_.next_halfedge(h));

                    if (vv0 != v0 && vv1 != v0)
                    {
                        // Calculate z component of flat normal
                        orient = (mesh_.position(vv0)[0] - uv[0]) * (mesh_.position(vv1)[1] - uv[1]) - (mesh_.position(vv0)[1] - uv[1]) * (mesh_.position(vv1)[0] - uv[0]);

                        if (orient < std::numeric_limits<Scalar>::min())
                        {
                            return true;
                        }
                    }
                }
            }

            return false;
        }

        Scalar area(Face f) const
        {
            return fnormal_[f].norm();
        }

        static constexpr Scalar orientation(const Vector2S& t1, const Vector2S& t2)
        {
            return t1.x() * t2.y() - t1.y() * t2.x();
        }

        Scalar orientation(Face f) const
        {
            auto h = mesh_.halfedge(f);
            auto v0 = mesh_.to_vertex(h);
            h = mesh_.next_halfedge(h);
            auto v1 = mesh_.to_vertex(h);
            h = mesh_.next_halfedge(h);
            auto v2 = mesh_.to_vertex(h);

            Vector2S t1 = Vector3S(mesh_.position(v1)).topRows<2>() - Vector3S(mesh_.position(v0)).topRows<2>();
            Vector2S t2 = Vector3S(mesh_.position(v2)).topRows<2>() - Vector3S(mesh_.position(v0)).topRows<2>();

            return orientation(t1, t2);
        }

        void add_properties()
        {
            update_ = mesh_.add_vertex_property<Vector2S>("v:update");

            // Frames and normals
            fnormal_ = mesh_.face_property<Vector3S>("f:normal");
            fframe_ = mesh_.add_face_property<Frame>("f:frame");
            
            // Quadrics
            fquadric_ = mesh_.add_face_property<Matrix4S>("f:quadric");
            vquadric_ = mesh_.add_vertex_property<Quadric2>("v:quadric");
            bquadric_ = mesh_.add_vertex_property<Quadric2>("v:boundary quadric");
            equadric_ = mesh_.add_edge_property<Matrix2S>("e:quadric");
        }

        void remove_properties()
        {
            mesh_.remove_vertex_property(update_);

            // Tangents and normals
            mesh_.remove_face_property(fnormal_);
            mesh_.remove_face_property(fframe_);
            

            // Quadrics
            mesh_.remove_face_property(fquadric_);
            mesh_.remove_vertex_property(vquadric_);
            mesh_.remove_vertex_property(bquadric_);
            mesh_.remove_edge_property(equadric_);
        }


        // The flip criterion must implement the () operator accepting an edge e and return a scalar whether (> 0) and how badly this edge needs to be flipped
        template <typename FlipCriterion>
        void flip_edges()
        {
            updateFramesAndQuadrics<false, true>();

            constexpr Scalar threshold = std::numeric_limits<Scalar>::epsilon();

            int counter = 0;
            FlipCriterion criterion(mesh_, *this);
            updatable_queue<Edge, Scalar> queue(mesh_);

            Scalar c;

            for (auto e : mesh_.edges())
            {
                if (!mesh_.is_boundary(e))
                {
                    c = criterion(e);

                    if (c > threshold)
                    {
                        queue.push(e, c);
                    }
                }
            }

            Edge edge;
            Face face;

            while(!queue.empty())
            {
                edge = queue.top().first;
                queue.pop();

                if (mesh_.is_flip_ok(edge) && isConvex(edge))
                {
                    mesh_.flip(edge);

                    counter++;

                    face = mesh_.face(edge, 0);

                    for (auto h : mesh_.halfedges(face))
                    {
                        if (!mesh_.is_boundary(mesh_.edge(h)) && mesh_.edge(h) != edge) // Don't add the current edge again
                        {
                            c = criterion(mesh_.edge(h));

                            if (c > threshold)
                            {
                                // Edge might not have been on the queue before
                                queue.update_or_push(mesh_.edge(h), c);
                            }
                            else
                            {
                                // If it was on the queue and the flip fixed it, remove it
                                queue.remove(mesh_.edge(h));
                            }
                        }
                    }

                    face = mesh_.face(edge, 1);

                    for (auto h : mesh_.halfedges(face))
                    {
                        if (!mesh_.is_boundary(mesh_.edge(h)) && mesh_.edge(h) != edge) // Don't add the current edge again
                        {
                            c = criterion(mesh_.edge(h));

                            if (c > threshold)
                            {
                                // Edge might not have been on the queue before
                                queue.update_or_push(mesh_.edge(h), c);
                            }
                            else
                            {
                                // If it was on the queue and the flip fixed it, remove it
                                queue.remove(mesh_.edge(h));
                            }
                        }
                    }
                }
            }

            // Update triangles and transfer to GPU
            tmesh_.update_triangles(mesh_);
            tmesh_.triangles = tmesh_.triangles.to(torch::kCUDA);
        }

        void setNormalMap(const cv::Mat& normals, const cv::Mat& mask = cv::Mat())
        {
            uvs_.clear();
            normals_.clear();

            std::size_t pixels;

            int height = normals.rows;
            int width = normals.cols;

            if (mask.empty())
            {
                umin_ = 0; vmin_ = 0;
                umax_ = normals.cols; vmax_ = normals.rows;
                pixels = umax_ * vmax_;
            }
            else
            {
                assert(mask.cols == width && mask.rows == height && "Mask and Normalmap must have same shape!");

                pixels = 0;
                umax_ = 0; vmax_ = 0;
                umin_ = mask.cols;
                vmin_ = mask.rows;

                for (int v = 0; v != height; ++v)
                {
                    for (int u = 0; u != width; ++u)
                    {
                        if (mask.at<uchar>(v, u) > 127)
                        {
                            pixels++;
                            umin_ = std::min(umin_, u);
                            vmin_ = std::min(vmin_, v);
                            umax_ = std::max(umax_, u + 1);
                            vmax_ = std::max(vmax_, v + 1);
                        }
                    }
                }
            }

            // We only store normals for foreground pixels. This improves performance of updateFramesAndQuadrics
            foreground_ = torch::empty({ pixels }).to(torch::kInt);
            auto accFG = foreground_.accessor<int, 1>();
            normals_.reserve(pixels);
            uvs_.reserve(pixels);
            pixels = 0;

            for (int v = vmin_; v != vmax_; ++v)
            {
                for (int u = umin_; u != umax_; ++u)
                {
                    if (mask.empty() || mask.at<uchar>(v, u) > 127)
                    {
                        // BGR to RGB and Normal Clamping
                        normals_.push_back(clamp_normal(Vector3S(normals.at<cv::Vec3f>(v, u)[2], normals.at<cv::Vec3f>(v, u)[1], normals.at<cv::Vec3f>(v, u)[0]), Vector2S(u, v)));
                        
                        uvs_.emplace_back(u, v);
                        
                        accFG[pixels++] = ((v - vmin_) * (umax_ - umin_) + (u - umin_));
                    }
                }
            }

            foreground_ = foreground_.cuda().to(torch::kLong);
        }

        // Update the local frames for each triangle
        template <bool need_vertices, bool need_edges, bool use_regularization = true>
        void updateFramesAndQuadrics()
        {
            torch::Tensor positions = tmesh_.positions.clone();

            positions.index_put_({ "...", 0 }, 2. * (positions.index({ "...", 0 }) - static_cast<float>(umin_) + 0.5) / static_cast<float>(umax_ - umin_) - 1.);
            positions.index_put_({ "...", 1 }, 2. * (positions.index({ "...", 1 }) - static_cast<float>(vmin_) + 0.5) / static_cast<float>(vmax_ - vmin_) - 1.);
            positions.index_put_({ "...", 2 }, 0);
            positions.index_put_({ "...", 3 }, 1);

            // Triangle ids give one triangle for each pixel
            Renderer& renderer = Renderer::getInstance();

            // Only transfer those ids that concern foreground pixels
            torch::Tensor tIds = torch::reshape(renderer.render(positions, tmesh_.triangles, std::make_tuple(vmax_ - vmin_, umax_ - umin_))[0], {-1, 4}).index({ foreground_, 3 }).to(torch::kInt).to(torch::kCPU, true);

            // Reset frames
            #pragma omp parallel for
            for (int i = 0; i != mesh_.n_faces(); ++i)
            {
                Face f(i);
                fnormal_[f].setZero();
                fquadric_[f].setZero();
            }

            torch::cuda::synchronize();

            // Calculcate Face normal as average of pixel normals
            auto accIds = tIds.accessor<int, 1>();

            for (int p = 0; p != uvs_.size(); ++p)
            {
                const int& id = accIds[p];
                
                if (id > 0)
                {
                    Face f(id - 1);
                    
                    fnormal_[f] += normals_[p];
                }
            }

            // Calculcate the local coordinates frames for the faces and
            Point c;
            Vector2S uv;
            Scalar norm2;

            #pragma omp parallel for private(c, uv, norm2)
            for (int i = 0; i != mesh_.n_faces(); ++i)
            {
                Face f(i);
                c = centroid(mesh_, f);
                uv.x() = c[0];
                uv.y() = c[1];
                
                fframe_[f] = frame(fnormal_[f], uv);

                // Area weighted normals
                norm2 = fnormal_[f].squaredNorm();

                if (norm2 > std::numeric_limits<Scalar>::epsilon())
                {
                    // fnormal_[f].normalize();
                    fnormal_[f] *= triangle_area(mesh_, f) * std::sqrt(std::abs((fframe_[f].transpose() * fframe_[f]).determinant()) / norm2);
                }
                else
                {
                    fnormal_[f] = -std::numeric_limits<Scalar>::epsilon() * Vector3S::Unit(2); // * 10e-4; //.setZero();
                }
            }

            // Calculate quadrics
            Scalar weight;
            Matrix3S M;
            Vector3S normal, tangent;
            Vector2S bary2;

            for (int p = 0; p != uvs_.size(); ++p)
            {
                const int& id = accIds[p];
                
                if (id > 0)
                {
                    Face f(id - 1);
                    
                    bary2 = Vector3S(centroid(mesh_, f)).topRows<2>();

                    normal = fnormal_[f].normalized();

                    if constexpr(use_regularization)
                    {
                        M = (normals_[p] * normals_[p].transpose() + REGULARIZATION * Matrix3S::Identity());
                    }
                    else
                    {
                        M = normals_[p] * normals_[p].transpose();
                    }

                    tangent = fframe_[f] * (uvs_[p] - bary2);

                    fquadric_[f].topLeftCorner<3, 3>() += M;
                    fquadric_[f].topRightCorner<3, 1>() -= M * tangent;
                    fquadric_[f](3, 3) += tangent.dot(M * tangent);
                }
            }

            // Finalize quadrics on the faces and ...
            Scalar trace;

            #pragma omp parallel for private(trace)
            for (int i = 0; i != mesh_.n_faces(); ++i)
            {
                Face f(i);
                // Symmetrize
                fquadric_[f].bottomLeftCorner<1, 3>() = fquadric_[f].topRightCorner<3, 1>().transpose();

                // Since tr(A) = sum_p n_p^2 = sum_p 1 = Number of Pixels
                trace = fquadric_[f].topLeftCorner<3, 3>().trace();

                if (trace > 0.5) // At least one pixel
                {
                    fquadric_[f] *= area(f) / trace;
                }
                else
                {
                    fquadric_[f].setZero();
                }
            }

            Vector3S offset;
            Quadric3 quadric;
            Frame frm;

            if constexpr (need_vertices)
            {
                // Quadrics for the vertices
                #pragma omp parallel for private(normal, quadric, bary2, uv, offset, frm)
                for (int i = 0; i != mesh_.n_vertices(); ++i)
                {
                    Vertex v(i);

                    normal.setZero();
                    quadric.setZero();

                    uv.x() = mesh_.position(v)[0];
                    uv.y() = mesh_.position(v)[1];

                    for (auto f : mesh_.faces(v))
                    {
                        // Normals
                        normal += fnormal_[f];

                        // Quadrics
                        bary2 = Vector3S(centroid(mesh_, f)).topRows<2>();
                        offset = -fframe_[f] * (uv - bary2);
                        quadric += translate(fquadric_[f], offset);
                    }

                    // Normals
                    uv.x() = mesh_.position(v)[0];
                    uv.y() = mesh_.position(v)[1];
                    frm = frame(normal, uv);

                    vquadric_[v] = bquadric_[v];

                    vquadric_[v].topLeftCorner<2, 2>() += frm.transpose() * quadric.topLeftCorner<3, 3>() * frm;
                    vquadric_[v].topRightCorner<2, 1>() += frm.transpose() * quadric.topRightCorner<3, 1>();
                    vquadric_[v](2, 2) += quadric(3, 3);

                    // Symmetrize
                    vquadric_[v].bottomLeftCorner<1, 2>() = vquadric_[v].topRightCorner<2, 1>().transpose();
                }
            }

            if constexpr (need_edges)
            {
                // ... move the quadrics to the edges
                #pragma omp parallel for private(uv, M, normal, frm)
                for (int i = 0; i != mesh_.n_edges(); ++i)
                {
                    Edge e(i);

                    if (!mesh_.is_boundary(e))
                    {
                        auto v0 = mesh_.vertex(e, 0);
                        auto v1 = mesh_.vertex(e, 1);

                        auto f0 = mesh_.face(e, 0);
                        auto f1 = mesh_.face(e, 1);

                        uv.x() = 0.5 * (mesh_.position(v0)[0] + mesh_.position(v1)[0]);
                        uv.y() = 0.5 * (mesh_.position(v0)[1] + mesh_.position(v1)[1]);

                        M = fquadric_[f0].topLeftCorner<3, 3>() + fquadric_[f1].topLeftCorner<3, 3>();
                        normal = fnormal_[f0] + fnormal_[f1];

                        frm = frame(normal, uv);

                        equadric_[e] = frm.transpose() * M * frm;
                    }
                }
            }
        }

        void optimizePositions()
        {
            updateFramesAndQuadrics<true, false>();

            #pragma omp parallel for
            for (int i = 0; i != mesh_.n_vertices(); ++i)
            {
                Vertex v(i);
            
                if (!mesh_.is_boundary(v))
                {
                    update_[v] = -vquadric_[v].topLeftCorner<2, 2>().ldlt().solve(vquadric_[v].topRightCorner<2, 1>());
                }
            }

            Vertex v1, v2;
            Vector2S t1, t2, dt1, dt2;
            Scalar m, c;

            // Make step
            for (auto v : mesh_.vertices())
            {
                if (!mesh_.is_boundary(v))
                {
                    // Make sure that triangles do not flip
                    Scalar alpha = alpha_;
                    Scalar s;

                    dt1 = -update_[v];
                    dt2 = -update_[v];

                    for (auto h : mesh_.halfedges(v))
                    {
                        v1 = mesh_.to_vertex(h);
                        v2 = mesh_.to_vertex(mesh_.next_halfedge(h));

                        t1 = Vector3S(mesh_.position(v1)).topRows<2>() - Vector3S(mesh_.position(v)).topRows<2>();
                        t2 = Vector3S(mesh_.position(v2)).topRows<2>() - Vector3S(mesh_.position(v)).topRows<2>();

                        m = orientation(dt1, t2) + orientation(t1, dt2);
                        c = orientation(t1, t2);

                        if (m < -std::numeric_limits<Scalar>::min() && c > std::numeric_limits<Scalar>::epsilon())
                        {
                            s = -c/m;
                            
                            if (s > 0)
                            {
                                alpha = std::fmin<Scalar>(alpha, 0.75 * s);
                            }
                        }
                        else if (c < 0)
                        {
                            alpha = 0;
                        }
                    }

                    update_[v] *= alpha;

                    mesh_.position(v)[0] += update_[v][0];
                    mesh_.position(v)[1] += update_[v][1];
                }
            }

            // Update positions and transfer to cuda
            tmesh_.update_vertices(mesh_);
            tmesh_.positions = tmesh_.positions.to(torch::kCUDA);
        }

        static constexpr Scalar REGULARIZATION = 10e-5;

        SurfaceMesh& mesh_;
        Projection projection_;
        TorchMesh tmesh_;

        // Step width for the tangential smoothing
        Scalar alpha_ = 0.5;

        // Factor by which to change the step-width after each iteration
        Scalar beta_ = 1.;

        // Area of a (fronto-parallel) pixel
        const Scalar A_px;

        // Length of a (fronto-parallel) pixel
        const Scalar l_px;

        VertexProperty<Vector2S> update_;

        // Normals and Tangents
        FaceProperty<Vector3S> fnormal_;
        FaceProperty<Frame> fframe_;

        // Quadrics
        FaceProperty<Matrix4S> fquadric_;
        VertexProperty<Quadric2> vquadric_;
        VertexProperty<Quadric2> bquadric_;
        EdgeProperty<Matrix2S> equadric_;

        std::vector<Vector3S> normals_;
        std::vector<Vector2S> uvs_;
        torch::Tensor foreground_;

        int umax_, umin_, vmax_, vmin_;
    };
} // namespace pmp
