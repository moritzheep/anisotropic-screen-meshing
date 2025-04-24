#include <iostream>
#include <pmp/io/io.h>
#include <cmath>
#include <pmp/algorithms/Triangulation.h>
#include "smp/algorithms/TightIntegration.h"
#include "smp/algorithms/NormalScreenRemeshing.h"
#include "utility/ImageLoader.h"
#include "utility/ArgumentParser.h"
#include "smp/algorithms/ScreenMeshing.h"

#include <boost/timer/timer.hpp>

// Macros
boost::timer::auto_cpu_timer t; // Use boost::timer::auto_cpu_timer(stream) to reroute the output into stream
std::string label;
#ifndef BEGIN_SECTION
#define BEGIN_SECTION(X) label = X; t.start();
#endif
#ifndef END_SECTION
#define END_SECTION std::cout << label << ":"; t.stop(); t.report(); std::cout << std::flush;
#endif

int steps = 5;

void print_help_message()
{
    std::cout << "Options\n";
    std::cout << "  -n <path-to-normal-map>      = Path to the normal maps\n";
    std::cout << "  -t <path-to-triangle-mesh>   = Path to write the output mesh to (we recommend using .obj files)\n"; 
    std::cout << "  -m <path-to-foreground-mask> = Path to foreground mask (as b/w .png file, optional) \n";

    std::cout << "\n";
    std::cout << "  -e <approximation-error>     = Desired approximation error (optional, in pixels^4)\n";
    std::cout << "  -c <number-of-vertices>      = Number of Vertices in the finisehd mesh (optional)\n";
    std::cout << "  Please either\n    - set '-c' > 0 and omit '-e' or\n    - set '-e' to the desired value and set '-c' to '0'\n\n";
    

    std::cout << "\nYou can switch from orthographic to perspective projection by supplying intrinsics\n";
    std::cout << "  | -x  0 -u |\n";
    std::cout << "  |  0 -y -v |\n";
    std::cout << "  |  0  0  1 |\n";
    std::cout << "e.g. '-x 1000 -y 1000 -u 512 -v 512' for f=1000 and a resolution of 1024x1024.\n";
}

enum class Strategy { Linear, Exponential };

std::vector<int> reduction_strategy(int start, int end, int steps, Strategy strategy)
{
    std::vector<int> counts(steps);
    counts[0] = start;

    switch (strategy)
    {
        case Strategy::Linear:
            for (int i = 1; i != steps - 1; ++i)
            {
                float s = static_cast<float>(i) / static_cast<float>(steps - 1);
                counts[i] = static_cast<float>(start) * (1. - s) + static_cast<float>(end) * s;
            }
            break;
        case Strategy::Exponential:
            for (int i = 1; i != steps - 1; ++i)
            {
                float s = static_cast<float>(i) / static_cast<float>(steps - 1);
                counts[i] = std::pow(static_cast<float>(start), 1. - s) * std::pow(static_cast<float>(end), s);
            }
            break;
    }

    counts[steps - 1] = end;
    return counts;
}

int main(int argc, char *argv[])
{
    ArgumentParser parser(argc, argv);

    if (!parser.has_arguments())
    {
        print_help_message();
        return 0;
    }

    if (!parser.has_argument('n'))
    {
        std::cout << "Error: No Normal Map provided!\n\n";
        print_help_message();
        return 1;
    }

    if (!parser.has_argument('t'))
    {
        std::cout << "Warning: No output Mesh provided!\n\n";
    }

    bool use_decimate = false;

    pmp::RemeshingOptions roptions;
    roptions.disableFlips().disableMove();

    if (parser.has_argument('b'))
    {
        std::string s = parser.get_argument('b');

        for (int i = 0; i < s.length(); ++i)
        {
            switch (s[i])
            {
                case 'd':
                    use_decimate = true;
                    break;
                case 'f':
                    roptions.enableFlips();
                    break;
                case 'm':
                    roptions.enableMove();
                    break;
            }
        }
    }
    else
    {
        use_decimate = true;
        roptions.enableFlips();
        roptions.enableMove();
    }

    std::string path_to_normals = parser.get_argument('n');
    std::string path_to_mask = parser.has_argument('m') ? parser.get_argument('m') : "";

    bool flip = (parser.has_argument('f') && (parser.get_argument('f') == "1"));

    cv::Mat normals, mask;
    std::tie(normals, mask) = load_images(path_to_normals, path_to_mask);

    int height = normals.rows;
    int width = normals.cols;

    std::cout << "Height  : "  << mask.rows << "\n";
    std::cout << "Width   : "  << mask.cols  << "\n";
    std::cout << "Channels: "  << mask.channels() << std::endl;

    // Remap normals
    for (int v = 0; v != height; ++v)
    {
        for (int u = 0; u != width; ++u)
        {
            if (mask.at<uchar>(v, u) > 127)
            {
                normals.at<cv::Vec3f>(v, u) = cv::Vec3f(1. - 2. * normals.at<cv::Vec3f>(v, u)[0], 1. - 2. * normals.at<cv::Vec3f>(v, u)[1], 1. - 2. * normals.at<cv::Vec3f>(v, u)[2]);
            }
        }
    }

    if (flip)
    {
        std::cout << "Applying Normal Flip\n";
        for (int v = 0; v != height; ++v)
        {
            for (int u = 0; u != width; ++u)
            {
                if (mask.at<uchar>(v, u) > 127)
                {
                    // Flip x-Coordinate and y-Coordinate (still BGR)
                    normals.at<cv::Vec3f>(v, u)[1] *= -1.;
                    normals.at<cv::Vec3f>(v, u)[2] *= -1.;
                }
            }
        }
    }
    
    if (parser.has_argument('j'))
    {
        float jitter = std::stof(parser.get_argument('j')) / 180. * M_PI; // Noise in degrees

        std::cout << "Adding " << jitter << " rads of noise\n";

        std::default_random_engine generator;
        std::normal_distribution<float> distribution(0., jitter);

        for (int v = 0; v != height; ++v)
        {
            for (int u = 0; u != width; ++u)
            {
                cv::Vec3f normal = normals.at<cv::Vec3f>(v, u);
                float norm = 0;

                // Add noise
                for (int i = 0; i != 3; ++i)
                {
                    normal[i] += distribution(generator);
                    norm += std::pow(normal[i], 2);
                }

                // Renormalize
                normals.at<cv::Vec3f>(v, u) = normal / std::sqrt(norm);
            }
        }
    }

    BEGIN_SECTION("Meshing")

    // Create Quadmesh
    std::function<bool(int, int)> indicator = [&](int v, int u){ return (u != 0) && (v != 0) && (u != width - 1) && (v != height - 1) && mask.at<uchar>(v, u) > 127; };

    pmp::SurfaceMesh mesh = pmp::from_indicator(indicator, height, width);

    int pixels = mesh.n_faces();
    std::cout << " - " << pixels << " Pixels\n";

    // Triangulate Quad-Mesh
    pmp::Triangulation triangulator(mesh);
    triangulator.triangulate(pmp::Triangulation::Objective::MAX_ANGLE); // Objective doesn't matter in the particular case

    END_SECTION

    // Decimation strategy
    int targetCount = parser.has_argument('c') ? std::stoi(parser.get_argument('c')) : mesh.n_vertices();
    int originalCount = mesh.n_vertices();
    float anisotropy = parser.has_argument('a') ? std::stof(parser.get_argument('a')) : 1;

    float approx_error = parser.has_argument('e') ? std::stof(parser.get_argument('e')) : 1.;
    pmp::DecimationOptions options;

    if (parser.has_argument('e'))
    {
        options.setErrorThreshold(approx_error).setVertexTarget(0);
    }

    // Perspective Mode
    if (parser.has_argument('x'))
    {
        std::cout << "Perspective Mode\n";

        float ax = std::stod(parser.get_argument('x'));
        float ay = parser.has_argument('y') ? std::stod(parser.get_argument('y')) : ax;

        float u0 = parser.has_argument('u') ? std::stod(parser.get_argument('u')) : static_cast<float>(width) / 2.;
        float v0 = parser.has_argument('v') ? std::stod(parser.get_argument('v')) : static_cast<float>(height) / 2.;

        float scale = 1. / std::sqrt(std::abs(ax * ay)); // Approximately the size of one pixel at distance 1

        {
            BEGIN_SECTION("Remeshing")

            pmp::NormalScreenRemeshing<pmp::EigenPerspective> remesher(mesh, normals, mask, pmp::EigenPerspective(ax, -ay, u0, v0));

            for (auto target : reduction_strategy(std::min<int>(10. * targetCount, mesh.n_vertices()), targetCount, steps, Strategy::Exponential))
            {
                if (parser.has_argument('c'))
                {
                    std::cout << "\nVertex Target: " << target << "\n";
                    options.setVertexTarget(target);
                }

                if (use_decimate)
                {
                    if (target < mesh.n_vertices() && !remesher.decimate(options))
                    {
                        // No collapse occured, the mesh has converged
                        break;
                    }
                }

                std::cout << "Vertex Actual: " << mesh.n_vertices() << "\n" << std::endl;

               
                remesher.remeshing(roptions);
            }

            remesher.remove_duplicates(0.001);
            std::cout << mesh.n_vertices() << " unique Vertices\n\n";

            END_SECTION
        }

        BEGIN_SECTION("Integration")
    
        // Perform integration
        pmp::Integration<double, pmp::Perspective> integrator(mesh, normals.clone(), mask, pmp::Perspective(ax, -ay, u0, v0));
        integrator.run();
    
        END_SECTION

        // Transform mesh to world-coordinates
        // for (auto v : mesh.vertices())
        // {
        //     auto pos = remesher.mesh().position(v);

        //     pos[0] -= u0;
        //     pos[0] /= ax;

        //     pos[1] -= v0;
        //     pos[1] /= ay;

        //     remesher.mesh().position(v) = pos;
        // }
        for (auto v : mesh.vertices())
        {
            mesh.position(v)[2] /= scale;
        }

        if (parser.has_argument('t'))
        {
            pmp::write(mesh, parser.get_argument('t'));
        }
    }
    // Orthographic Mode
    else
    {
        std::cout << "Orthographic Mode\n";

        {
            BEGIN_SECTION("Remeshing")

            pmp::NormalScreenRemeshing<pmp::EigenOrthographic> remesher(mesh, normals, mask);

            for (auto target : reduction_strategy(std::min<int>(10. * targetCount, mesh.n_vertices()), targetCount, steps, Strategy::Exponential))
            {
                if (parser.has_argument('c'))
                {
                    std::cout << "\nVertex Target: " << target << "\n";
                    options.setVertexTarget(target);
                }

                if (use_decimate)
                {
                    if (target < mesh.n_vertices() && !remesher.decimate(options))
                    {
                        // No collapse occured, the mesh has converged
                        break;
                    }
                }

                std::cout << "Vertex Actual: " << mesh.n_vertices() << "\n" << std::endl;

                remesher.remeshing(roptions);
            }

            END_SECTION

            remesher.remove_duplicates(0.001);
            std::cout << mesh.n_vertices() << " unique Vertices\n\n";
        }

        BEGIN_SECTION("Integration")

        // Perform integration
        pmp::Integration<double, pmp::Orthographic> integrator(mesh, normals.clone(), mask);
        integrator.run();

        END_SECTION

        if (parser.has_argument('t'))
        {
            if (flip)
            {
                for (auto v : mesh.vertices())
                {
                    mesh.position(v)[2] *= -1;
                }
            }

            pmp::write(mesh, parser.get_argument('t'));
        }
    }

    int vertices = mesh.n_vertices();

    std::cout << " - " << vertices << " Vertices\n";
}
