# EE/CprE/SE 491 WEEKLY REPORT 03

**Report Period:** 03/05/2025 – 03/11/2025
**Group Number:** sddec25-01
**Project Title:** Semantic Segmentation Optimization
**Client/Advisor:** JR Spidell/Namrata Vaswani

## Team Members/Role

- **Joseph Metzen** – Kria Board Manager
- **Tyler Schaefer** – ML Algorithm Analyst
- **Conner Ohnesorge** – ML HW Integration Engineer
- **Aidan Perry** – Multithreaded Program Developer

---

## Weekly Summary

This week our team has consumed more knowledge of the codebase with the comments and diagrams provided. Creating slide decks to further our education for this project. Also meeting with old team members of the project to explain the small details.

---

## Past Week Accomplishments

### Joseph Metzen
- Figured out what components to use for our upcoming assignment
- Created a Slide Deck on the Kria Board
- Met up with the team to tell them the best options we have on our Kria Board

### Tyler Schaefer
- Had a second meeting with Mason Inman and the Client to discuss setting up the environment for the algorithm and the quantization of the algorithm for the Kria board
- Continuing to get familiar with the existing codebase

### Conner Ohnesorge
- Built Petalinux OS and began writing nix derivations of the dockerfile to allow for fully multithreaded and deterministic builds of software dependencies
- With assistance from lab experience, began further designing and experimenting with the interactions between OS and underlying memory hierarchy of the board

### Aidan Perry
- Connected with last semester's multithreaded programming developer to find out more information about the project
- Discussed options for memory options to use for the Kria Board

---

## Pending Issues

- **Joseph Metzen:** Still waiting on the Kria Board
- **Tyler Schaefer:** Need to set up the environment and train/test the existing model
- **Conner Ohnesorge:** _(no pending issues listed)_
- **Aidan Perry:** Still figuring out how the algorithm completely works with a mathematical problem that is fed into the algorithm

---

## Individual Contributions

| Team Members | Individual Contributions | Hours this week | Hours Cumulative |
|--------------|-------------------------|-----------------|------------------|
| **Joseph Metzen** | • Spent majority of my time looking through the datasheet to find different memory components<br>• Read up on some information about how to use Xilinx Vitis | 7 | 19 |
| **Tyler Schaefer** | • Meeting with Mason and JR<br>• Began setting up the environment for the algorithm | 3 | 12 |
| **Conner Ohnesorge** | • Created Slide deck detailing data version control management within a git repo while storing large files in an s3-compatible bucket<br>• Gave lightning talk to senior project class. Covered Unet architecture and scheduling of tasks | 3 | 16 |
| **Aidan Perry** | • Got in contact with Dylan to ask questions about the current implementation of the threading programs<br>• Reviewed presentations and videos from the last team describing how they developed and proved their design<br>• Tested some matrices to try and connect what I learned to the actual running process | 4 | 15 |

---

## Plans for the Upcoming Week

- **Joseph Metzen:** Familiarize myself more with the Kria Board once it arrives and run some code on it
- **Tyler Schaefer:** Finish setting up the environment for the algorithm and complete training and testing to validate that I have the model working correctly
- **Conner Ohnesorge:** Meet with JR to give the dvc presentation. Create a customized build of Petalinux
- **Aidan Perry:** Continue to connect with the previous member that had worked on my portion of the project to further understand how to test and optimize the multithreading

---

## Tags
#weekly-report #sddec25-01 #semantic-segmentation #kria-board #report-03

## Related
- [[Report2Semester1]]
- [[Report4Semester1]]
