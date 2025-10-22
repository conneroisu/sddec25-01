# Project Context

## Purpose

This project focuses on optimizing semantic segmentation algorithms for real-time eye tracking in assistive technology applications, specifically for individuals with mobility impairments and underlying conditions such as cerebral palsy. The system aims to detect early warning signs of medical distress through eye movement and body posture analysis, enabling autonomous response capabilities.

### Goals

- **Performance Optimization**: Improve processing speed from 160ms per frame to approximately 33.2ms per frame for 4 frames simultaneously, achieving 60 FPS real-time processing
- **Accuracy Maintenance**: Maintain 99.8% Intersection over Union (IoU) accuracy for semantic segmentation after optimization
- **Resource Efficiency**: Implement efficient resource scheduling to ensure fair DPU (Deep Processing Unit) access across all algorithms without starvation
- **Real-Time Medical Assistance**: Enable proactive detection and response to medical episodes for wheelchair users with disabilities
- **Edge Computing**: Leverage edge computing capabilities for real-time health monitoring without cloud dependency

## Tech Stack

### Hardware Platform
- **AMD Kria KV260 Development Board**
  - Zynq UltraScale+ MPSoC
  - 4GB DDR memory
  - Deep Processing Unit (DPU) for neural network inference
  - Multi-core ARM Cortex-A53 processors
  - FPGA fabric for custom acceleration

### Software & Frameworks
- **Programming Languages**: C++ (primary), Python (tooling)
- **Neural Network Framework**: ONNX (Open Neural Network Exchange)
- **AI Framework**: Vitis-AI and Vitas-Runtime for model optimization and deployment
- **Threading**: POSIX threads for multi-core parallelism
- **Containerization**: Docker for development and deployment environments
- **Version Control**: Git/GitHub
- **Architecture**: U-Net semantic segmentation model

### Development Tools
- Xilinx Vitis IDE
- GCC/G++ compiler for ARM
- Docker containerization
- CLI-based interface for configuration and monitoring

## Project Conventions

### Code Style
- **Language**: Strict C++17/C++20 standards with modern best practices
- **Memory Management**: Manual memory management with careful allocation strategies for embedded constraints
- **Naming Conventions**:
  - Classes: PascalCase (e.g., `SemanticSegmentationPipeline`)
  - Functions/Methods: camelCase (e.g., `processPipelineStage()`)
  - Constants: UPPER_SNAKE_CASE (e.g., `MAX_BUFFER_SIZE`)
  - Variables: snake_case for local variables
- **Threading**: Use POSIX thread primitives with explicit synchronization
- **Error Handling**: Robust error handling for pipeline management, frame drops, and data corruption
- **Comments**: Document complex algorithmic divisions and synchronization points

### Architecture Patterns
- **Pipelined Architecture**: Four-stage pipeline for parallel U-Net semantic segmentation processing
- **Resource Scheduling**: Fair scheduling strategy for DPU access across multiple algorithms
- **Multi-threaded Processing**: Concurrent execution across multiple ARM cores
- **Memory Optimization**: Buffer management with careful consideration of 4GB DDR limitation
- **Modular Design**: Separation of concerns between:
  - Image preprocessing
  - Semantic segmentation (split into 4 parallel stages)
  - Blink detection
  - Eye tracking
  - Medical distress detection

### Testing Strategy
- **Performance Metrics**:
  - Throughput: Target <33.2ms per frame processing time
  - Accuracy: Maintain 99.8% IoU for semantic segmentation
  - Resource utilization: Monitor DPU, CPU, and memory usage
  - Latency: Measure pipeline stage delays and synchronization overhead
- **Testing Approach**:
  - Unit tests for individual pipeline stages
  - Integration tests for end-to-end pipeline
  - Performance benchmarks on actual hardware
  - Stress testing with concurrent frame processing
- **IEEE Standards Compliance**:
  - IEEE 3129-2023: AI-based image recognition robustness testing
  - IEEE 2802-2022: AI-based medical device performance evaluation
- **Validation**:
  - Real-world testing with eye-tracking datasets
  - Edge case handling (lighting conditions, user movement)
  - Error recovery and fault tolerance testing

### Git Workflow
- **Branching Strategy**: Hybrid Waterfall + Agile approach
  - `main`: Production-ready code
  - `develop`: Integration branch
  - `feature/*`: Individual feature development
  - `bugfix/*`: Bug fixes
  - `experiment/*`: Algorithm optimization experiments
- **Commit Conventions**:
  - Descriptive commit messages following conventional commits
  - Reference issue/task numbers in commits
  - Atomic commits for each logical change
- **Code Review**: Client has read access to GitHub for real-time progress tracking
- **Documentation**: Keep implementation docs in sync with code changes

## Domain Context

### Assistive Technology for Mobility-Impaired Individuals
This project addresses critical safety needs for wheelchair users with underlying medical conditions (cerebral palsy, epilepsy, cardiovascular disorders). The system must:
- Detect early physiological warning signs through eye movement analysis
- Provide real-time response with minimal latency
- Operate autonomously without constant caregiver supervision
- Integrate with wheelchair control systems for safety positioning

### Semantic Segmentation for Medical Applications
- **U-Net Architecture**: Convolutional neural network specifically designed for biomedical image segmentation
- **Eye Tracking Indicators**: Analyze pupil position, blink patterns, and gaze direction
- **Real-Time Constraints**: Processing must occur at 60 FPS minimum for responsive assistance
- **Edge Computing**: All processing occurs on-device for privacy and low-latency response

### Embedded Systems Constraints
- **Limited Resources**: 4GB DDR memory shared across all pipeline stages
- **FPGA Resource Allocation**: Efficient sharing of DPU between multiple algorithms
- **Power Constraints**: Battery-operated wheelchair systems require power efficiency
- **Thermal Considerations**: Sustained processing must not overheat embedded hardware

### User Groups
1. **Primary Users**: Individuals with mobility impairments using wheelchairs
2. **Secondary Users**: Caregivers and family members requiring alert systems
3. **Tertiary Users**: Healthcare providers and emergency responders needing health data integration

## Important Constraints

### Technical Constraints
- **Hardware**: Limited to Xilinx Kria KV260 board specifications
- **Memory**: 4GB DDR memory must accommodate all pipeline stages and buffers
- **Processing Power**: Multi-core ARM Cortex-A53 + DPU shared resource
- **FPGA Resources**: Limited logic elements for custom acceleration
- **Real-Time Requirements**: Strict 33.2ms per frame deadline
- **Accuracy Requirements**: Must maintain 99.8% IoU (currently at 98.8%)

### Physical Constraints
- **Deployment Environment**: Wheelchair-mounted system subject to vibration and movement
- **Power Budget**: Battery-operated with power efficiency requirements
- **Size/Weight**: Must fit within wheelchair mounting constraints
- **Environmental**: Operating temperature and humidity ranges for medical devices

### Economic Constraints
- **Cost-Effectiveness**: Minimize additional hardware requirements
- **Maintenance**: Future updates must remain economical
- **Scalability**: Design for potential future hardware upgrades

### Regulatory/Standards Constraints
- **IEEE 2952-2023**: Secure Computing Based on Trusted Execution Environment
- **IEEE 2802-2022**: Performance and Safety Evaluation of AI-Based Medical Devices
- **IEEE 7002-2022**: Data Privacy Process
- **IEEE 3129-2023**: Robustness Testing of AI-Based Image Recognition Services
- **IEEE 3156-2023**: Privacy-Preserving Computation Integrated Platforms
- **IEEE 2842-2021**: Secure Multi-Party Computation
- **IEEE 1484.1-2003**: Learning Technology Systems Architecture

### Safety Constraints
- **Data Privacy**: Strict privacy measures for sensitive user health data
- **Security**: Secure data handling between pipeline stages
- **Reliability**: Robust error handling and fault tolerance
- **Medical Safety**: False positive/negative rates must meet medical device standards

## External Dependencies

### Hardware Dependencies
- **AMD Kria KV260 Board**: Primary deployment platform
- **Deep Processing Unit (DPU)**: Hardware accelerator for neural network inference
- **Wheelchair Integration**: Interface with wheelchair control systems (implementation-specific)

### Software Dependencies
- **Vitis-AI SDK**: Model quantization, compilation, and optimization tools
- **Vitas-Runtime**: Inference runtime for DPU
- **ONNX Runtime**: Neural network model interchange format
- **Docker**: Container runtime for development environment
- **Xilinx Platform**: Board support packages and drivers

### Data Dependencies
- **Eye-Tracking Dataset**: Training and validation data for semantic segmentation
- **Medical Episode Data**: Ground truth data for distress detection validation
- **Calibration Data**: User-specific calibration for eye-tracking accuracy

### Communication Dependencies
- **Telegram**: Primary communication channel with client and previous team members
- **GitHub**: Code repository and project tracking (client has read access)

### Standards & Frameworks
- **IEEE Standards Suite**: Compliance with medical device and AI safety standards
- **ONNX Format**: Industry-standard neural network interchange
- **POSIX Threading**: Standard threading API for portability
