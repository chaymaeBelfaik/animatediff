# OpenPose GPU Setup for Linux

This repository contains a comprehensive setup script for installing OpenPose with GPU acceleration on Linux systems.

## 🚀 Quick Start

1. **Make the script executable:**
   ```bash
   chmod +x openpose_gpu_setup.sh
   ```

2. **Run the setup script:**
   ```bash
   ./openpose_gpu_setup.sh
   ```

## 📋 Prerequisites

### Required Software
- **CUDA Toolkit** (version 11.0 or higher, recommended: 11.8+)
- **Git** for cloning the repository
- **CMake** (will be installed automatically if missing)

### Recommended Software
- **cuDNN** for optimal performance
- **NVIDIA GPU** with compute capability 6.0 or higher

### System Requirements
- Linux distribution (Ubuntu, CentOS, RHEL, Fedora, etc.)
- At least 10GB free disk space
- 8GB+ RAM recommended
- NVIDIA GPU with CUDA support

## 🔧 What the Script Does

### 1. System Checks
- Verifies CUDA installation and version
- Checks for cuDNN availability
- Ensures required tools are present

### 2. Dependency Installation
- Installs system packages via package manager (apt-get, yum, or dnf)
- Downloads OpenPose 3rd-party dependencies
- Sets up build environment

### 3. OpenPose Configuration
- Clones/updates OpenPose repository
- Configures CMake with GPU support
- Sets appropriate CUDA architecture flags
- Enables OpenCL, cuDNN, and OpenCV support

### 4. Build Instructions
- Provides clear next steps for building
- Includes testing commands
- Offers performance optimization tips

## 🎯 GPU Configuration Details

The script configures OpenPose with:
- **GPU_MODE**: OPENCL (supports both NVIDIA and AMD GPUs)
- **CUDA Support**: Enabled with automatic architecture detection
- **cuDNN Support**: Enabled (if available)
- **OpenCV**: Version 4.x and 5.x support
- **OpenCL**: Enabled for cross-platform GPU support

## 🚨 Troubleshooting

### Common Issues

1. **CUDA not found**
   - Install CUDA Toolkit from NVIDIA's website
   - Ensure CUDA is in your PATH

2. **Build fails with memory errors**
   - Reduce parallel build jobs: `make -j2` instead of `make -j$(nproc)`
   - Close other applications to free memory

3. **Dependencies not found**
   - Run the script with sudo privileges
   - Check your package manager is supported

4. **GPU not detected**
   - Ensure NVIDIA drivers are installed
   - Check `nvidia-smi` output

### Performance Tips

- **Build with all cores**: `make -j$(nproc)` for fastest build
- **Use Release mode**: Always build in Release configuration
- **Enable cuDNN**: Install cuDNN for significant performance boost
- **Monitor GPU usage**: Use `nvidia-smi` to monitor GPU utilization

## 📚 Additional Resources

- [OpenPose Official Repository](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [cuDNN Installation](https://docs.nvidia.com/deeplearning/cudnn/install-guide/)
- [OpenPose Documentation](https://cmu-perceptual-computing-lab.github.io/openpose/web/html/doc/)

## 🔄 Updating

To update an existing installation:
1. Run the script again - it will update the repository
2. Rebuild: `cd build && make -j$(nproc)`

## 📝 Notes

- The build process typically takes 1-3 hours depending on your system
- GPU acceleration provides 5-10x performance improvement over CPU-only
- The script automatically detects your GPU architecture for optimal CUDA compilation
- All dependencies are downloaded from official OpenPose sources

## 🤝 Contributing

Feel free to submit issues or improvements to this setup script. The goal is to make OpenPose installation as painless as possible on Linux systems.

## 📄 License

This setup script is provided as-is for educational and development purposes. OpenPose itself is licensed under the Apache License 2.0.
