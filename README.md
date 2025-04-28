# Feature-Preserving Mesh Decimation for Normal Integration
### [Project Page](https://moritzheep.github.io/anisotropic-screen-meshing/) | [Paper](https://arxiv.org/abs/2504.00867)


[Moritz Heep](https://moritzheep.github.io/) (PhenoRob, University of Bonn),

[Sven Behnke](https://www.ais.uni-bonn.de/behnke/) (Autonomous Intelligent Systems, University of Bonn),

[Eduard Zell](http://www.eduardzell.com/) (Independent Researcher)

# Getting Started
To clone the repository with all its submodules run
```Shell
$ git clone --recursive https://github.com/moritzheep/anisotropic-screen-meshing.git
```

## Prerequisites
Our method uses [nvdiffrast](https://github.com/NVlabs/nvdiffrast) to translate between the triangle mesh and the pixel grid. Please make sure that all dependencies of nvdiffrast are met, especially torch. Furthermore, we require [OpenCV](https://opencv.org/) to be installed.
### Docker
We prepared a Docker image to take care of these dependencies and facilitate testing. We still need the nvidia-container-runtime. It can be installed via
```Shell
$ cd docker
$ ./nvidia-container-runtime.sh
```
To build the image, run
```Shell
$ cd docker
$ ./build.sh
```
Finally, run
```Shell
$ ./run.sh
```
to get a list of options. You can then mount a volume and point towards your input files. All arguments can be appended to the above command and are passed through.
## Building
To build the project, run
```Shell
$ mkdir build
$ cd build
$ cmake ..
$ make
```

## Running
After the build has been completed successfully, you can run
```sh
$ src/main \
    -n <path-to-the-normal-map> \
    -m <path-to-the-foreground-mask> \
    -t <path-to-save-the-mesh> \
    -c 1000
```
for a quick test. The mask can be any 8Bit grayscale format supported by OpenCV. The mesh can be saved to any format supported by the [pmp-library](https://github.com/pmp-library/pmp-library). We recommend `.obj`.

You can run `src/main` to get of full list of all options.

# Troubleshooting
If you get meshes that curve in the wrong direction, try flipping the x or y coordinate of your normal map.

# Citation
This work has been accepted for IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Nashville TN, USA, June 2025.
```
@article{heep2025feature,
  title={Feature-Preserving Mesh Decimation for Normal Integration},
  author={Heep, Moritz and Behnke, Sven and Zell, Eduard},
  journal={arXiv preprint arXiv:2504.00867},
  year={2025}
}
```
