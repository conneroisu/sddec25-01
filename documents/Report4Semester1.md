# EE/CprE/SE 491 WEEKLY REPORT 04

**Date Range:** 03/12/2025 – 03/25/2025
**Group Number:** sddec25-01
**Project Title:** Semantic Segmentation Optimization
**Client/Advisor:** JR Spidell/Namrata Vaswani

## Team Members/Role

- **Joseph Metzen** – Kria Board Manager
- **Tyler Schaefer** – ML Algorithm Analyst
- **Conner Ohnesorge** – ML Integration HWE
- **Aidan Perry** – Multithreaded Program Developer

---

## Weekly Summary

In this week, each team member advanced their learning in each category that was assigned to them. We individually worked on becoming more familiar with our own specialties for the project. Furthermore, as the hardware has arrived, we began having some more hands-on learning with our specific application platform.

---

## Past Week Accomplishments

### Joseph Metzen
- Solidified what components the team will use for the project
- Watched online videos that taught me to read the data coming in and out of the Kria Board

### Tyler Schaefer
- Finished setting up the Vitis-AI docker container
- Trained the existing model to prove to the client that I have the Vitis-AI environment properly set up

### Conner Ohnesorge
- Created a test application where I split a segmentation model defined in pytorch into two ONNX models: encoder and decoder
- Decreased latency as required data storage in between encoding and decoding stages, but increased throughput for multicore processing
- Used GPU however

### Aidan Perry
- Worked on developing a matrix equation to be fed into the multithreaded pipeline
- Maintained contact with the previous owner of the codebase to discuss how to approach feeding in the equation and reconstructing the current codebase to complement the kind of equation requested

---

## Pending Issues

### Joseph Metzen
- Need to get my hands on the Kria Board and familiarize myself with it

### Tyler Schaefer
- I need to begin mathematically dividing the model
- Understanding how much the model may change will affect how the model should be divided

### Conner Ohnesorge
- Waiting for the Wednesday meeting to do a presentation of a data solution
- Wait for the team at Oregon State to finish the development environment for the kernel and overall Linux build

### Aidan Perry
- Becoming familiarized with the cv::mat class and how to create a formula in that format to be able to feed in the data that we would need to test in order to see how the threads interact and feed into one another

---

## Individual Contributions

| Team Members | Individual Contributions | Hours this week | Hours Cumulative |
|--------------|-------------------------|-----------------|------------------|
| **Joseph Metzen** | - Learned how data comes in and out of the Kria Board<br>- Figured out what components needed to increase throughput | 5 | 24 |
| **Tyler Schaefer** | - Setting up Vitis-AI<br>- Training model | 13 | 25 |
| **Conner Ohnesorge** | - ONNX model splitting examples/learning<br>- Finishing touches on the data presentation slides<br>- Found external demo videos to include with my presentation<br>- Custom build of Linux using petalinux-tools that uses USB announcements exploration | 4 | 20 |
| **Aidan Perry** | - Messaging with Dylan, the previous owner of the codebase<br>- Theorizing what kind of equation to feed into the algorithm<br>- Testing the current compiled codebase<br>- Testing out/playing with the hardware sent to us | 5 | 20 |

---

## Plans for the Upcoming Week

### Joseph Metzen
- Work with the Kria Board that was sent in and do small testing

### Tyler Schaefer
- Speak with the client about how much the model may/has to change or be optimized before we begin the division for pipelining

### Conner Ohnesorge
- Experiment with priority execution and further

### Aidan Perry
- Continue communication with previous codebase owners to understand more how to interact and test the threading
- Experiment with what might need to be modified for updated pipelined functionality and optimization with thread timings for this pipelined structure
