# Engineering Standards

Engineering standards are present in everything that involves the internet. They are important because of how interconnected our lives have become after the internet was adapted to just about everything in our lives.

We were tasked with investigating existing engineering standards in similar domains as our project.

## Primary Standards

### 1. ISO/IEC/IEEE International Standard - Software and systems engineering - Software testing -- Part 2: Test processes

This is a standard for the engineers that will be governing, managing and testing our own equipment. We will be testing our code constantly to make sure it runs properly.

### 2. ISO/IEC/IEEE International Standard - Systems and software engineering -- Life cycle processes -- Risk management

We added this standard because our project will be a risk factor for the user. The equipment itself won't harm, but not used right or not working could cause communication with the user. Example the user has a heart attack and our equipment wasn't working properly to notify or predict the outcome.

### 3. IEEE Recommended Practice on Software Reliability

This standard is about the reliability of our software and the predictions.

- Users are relying on our equipment to work so if not used properly the outcome will be on us.
- From high level programming model to FPGA machines
- Improving the FPGA design process through determining and applying logical-to-physical design mappings
- New FPGA architecture for bit-serial pipeline datapath
- On sparse matrix-vector multiplication with FPGA-based system
- PAM-Blox high performance FPGA design for adaptive computing
- Performance and area modeling of complete FPGA designs in the presence of loop transformations
- Pin assignment for multi-FPGA systems
- Scalable network based FPGA accelerators for an automatic target recognition application
- Task-level partitioning and RTL design space exploration for multi-FPGA architectures
- Techniques for FPGA Implementation of Video Compression Systems
- The Wave Pipeline Effect on LUT-Based FPGA Architectures

## Individual Summaries

### 1. Improving the FPGA Design Process Through Determining and Applying Logical-to-Physical Design Mappings

A method to connect the logical description of a design with its physical FPGA setup. By looking at files like netlists, bitstreams, and descriptions, the authors show how this mapping can help with debugging, estimating power usage, and making quick changes to the design without having to redo everything from scratch.

### 2. New FPGA Architecture for Bit-Serial Pipeline Datapath

A new FPGA design aimed at improving bit-serial pipeline systems. By changing how lookup tables (LUTs) are built and adding a special routing system, the new design reduces routing problems and gets close to 100% use of the logic, overcoming issues seen in older bit-parallel FPGA designs.

### 3. On Sparse Matrix-Vector Multiplication with FPGA-Based System

FPGAs for sparse matrix-vector multiplication, which is important for many scientific tasks. By using techniques like distributed arithmetic and smart scheduling, the authors make the best use of FPGAs' flexibility to handle irregular data and speed up computation.

### 4. PAM-Blox: High Performance FPGA Design for Adaptive Computing

Focused on adaptive computing, this paper introduces PAM-Blox, a system that uses object-oriented design to make FPGA programming more flexible. By breaking down designs into smaller, customizable parts, this approach helps designers optimize resource use and improve performance.

### 5. Performance and Area Modeling of Complete FPGA Designs in the Presence of Loop Transformations

Present models to predict the performance and size of FPGA designs that use loop transformations (like unrolling or tiling). These models help designers balance the benefits of more parallelism with the available resources, particularly in image processing.

### 6. Pin Assignment for Multi-FPGA Systems

The challenge of connecting multiple FPGAs together. Instead of random connections, the authors suggest an optimized method for assigning pins, which improves the speed of mapping and the quality of signal connections—important for applications like logic emulation.

### 7. Scalable Network Based FPGA Accelerators for an Automatic Target Recognition Application

FPGAs for real-time automatic target recognition (ATR) in radar images. The authors describe a system where FPGAs work together over a network, demonstrating that they can handle the high data processing needed for fast image analysis in defense applications.

### 8. Task-Level Partitioning and RTL Design Space Exploration for Multi-FPGA Architectures

The SPADE system is introduced, which helps divide tasks across multiple FPGAs. Starting from a task graph, SPADE looks at how to best split up tasks while considering architectural limits (like area and memory), helping designers optimize throughput while managing resources.

### 9. Techniques for FPGA Implementation of Video Compression Systems

Two approaches for video compression on FPGAs. One is a simple, single-FPGA method that uses algorithmic tricks to meet real-time demands, while the other pairs the FPGA with an external video processor to support multiple compression methods. The paper looks at the trade-offs between system complexity, image quality, and throughput.

### 10. The Wave Pipeline Effect on LUT-Based FPGA Architectures

Wave pipelining can improve FPGA performance. By adjusting delay times in logic paths, multiple data "waves" can be processed at once, boosting performance without adding extra registers. Their experiment with an array multiplier shows that this method can improve speed and reduce latency.

## Collective Overview

Together, these papers address a broad spectrum of challenges and innovations in FPGA design and implementation. Common themes include:

- **Design Process Enhancement**: Several works (e.g., Paper 1 and Paper 5) emphasize methodologies and modeling techniques that provide deeper insights into the mapping from high-level descriptions to physical implementations, facilitating debugging, power estimation, and rapid design iterations.

- **Architectural Innovations**: Novel architectures—such as bit-serial datapath FPGAs (Paper 2) and wave pipelined designs (Paper 10)—demonstrate how rethinking the conventional FPGA design can lead to significant improvements in logic utilization, routing efficiency, and overall performance.

- **Multi-FPGA and Partitioning Challenges**: With increasing design complexity, effective partitioning (Paper 8) and optimized pin assignments (Paper 6) become essential for multi-FPGA systems, ensuring that inter-chip communication is efficient and that resource constraints are managed effectively.

- **Application-Specific Implementations**: A number of papers apply these innovations to real-world problems—from sparse matrix computations (Paper 3) and adaptive computing (Paper 4) to video compression (Paper 9) and automatic target recognition (Paper 7)—showing the versatility and impact of advanced FPGA methodologies in both high-performance and resource-constrained environments.

## Conclusion

### Reflection

Some of the standards are so ubiquitous and commonplace that we can just accept them without having to factor them very strongly into our design.

The technical details of the three published standards are of course, relevant to our project as they are in some ways.

During this investigation, we chose not only software standards that we will use for the machine learning portions of our design project, but also some for the hardware implementation since we will be using a FPGA board with multiple processors. These also go into the standards that apply to the multi threading and managing processes in the Linux OS environment we will be using.

### Changes

We will review the codebase given to us and make sure that the applied standards we have selected match what has already been produced. We will also keep these standards in mind while we move to optimize the given codebase so that we do not violate any of the standards that have been put in place.

---

## Tags
#engineering-standards #fpga #software-testing #risk-management #reliability #ieee-standards #iso-standards

## Related Documents
- [[DesignDocSemester1]]
- [[TestingStrategy]]
- [[06-implementation]]
