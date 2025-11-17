# EE/CprE/SE 491 Weekly Report 02

**Report Period:** 02/25/2025 - 03/4/2025
**Group Number:** sddec25-01
**Project Title:** Semantic Segmentation Optimization
**Client & Advisor:** JR Spidell and Namrata Vaswani

## Team Members

- **Joseph Metzen** - Firmware Engineer
- **Tyler Schaefer** - ML Algorithm Engineer
- **Conner Ohnesorge** - ML Integration HWE
- **Aidan Perry** - Multi-Threading Specialist

---

## Weekly Summary

Team members began onboarding into their specialized areas of the project. Grouped up to see who was familiarized with the resources given and shared what work was done and learned so far.

---

## Past Week Accomplishments

### Joseph Metzen
Started with getting a good grasp of the Kria board and the understanding of it. I've looked through multiple datasheets and watched some videos on the Kria board. Also understanding which memory is best to use for the team.

### Tyler Schaefer
Met with Mason Inman from the ISU team currently working on the algorithm. Began familiarizing myself with the ONNX files and the current state of the algorithm.

### Conner Ohnesorge
Worked on communications with Oregon State Team, ran a docker build of the environment, learned more about how we can split up the Onnx defined ai model onto parts of the MPU, and prepared a branch and pull request for the Kria repo with some improvements to environment and preparations for versioned data sharing with dvc. Created github repo and project for task management and invited all team members.

### Aidan Perry
Shifting focus from operating system and docker functionality to how the multi-threading is already implemented. Reviewing that codebase to understand how a matrix is sent through the first thread and outputted into one another to provide, in the end, the complete solution after 4 threads have worked on simplifying that problem. This is in theory how our threads will work on processing the image at hand and we will in the future need to optimize these threads to function at a higher efficiency.

---

## Pending Issues

- **Joseph Metzen:** Still need the Kria board to be shipped to me.
- **Tyler Schaefer:** Need access to the GitHub repo for the actual source code of the algorithm.
- **Conner Ohnesorge:** Need notice of whether we are using proposed data vendor, tigris.
- **Aidan Perry:** A little unfamiliar with C++ Object Oriented Programming so the understanding of the processes has been a little slow, and understanding how to feed an equation into that pre-existing codebase.

---

## Individual Contributions

| NAME | Individual Contributions | Hours this week | Hours cumulative |
|------|-------------------------|-----------------|------------------|
| Joseph Metzen | Talked with client and team members, Studied the Kria Board, Getting to know Kria Board videos. Also dabbled in AMD Vivado. | 6 | 12 |
| Tyler Schaefer | Met with Mason, the client, and the team. Studying ONNX, U-Nets, and convolutions. | 6 | 9 |
| Conner Ohnesorge | Met with client and team. Further investigated the MPU on the Kria Board, started a documentation collection repository. Created a github project for task management. | 6 | 13 |
| Aidan Perry | Met with client and team. Analyzing and understanding the C++ codebase and trying to run small tests. | 6 | 11 |

---

## Plans for the Upcoming Week

- **Joseph Metzen:** Ask Client for more details and figure out more efficient ways to use the Kria board. Familiarize myself more with AMD Vivado.
- **Tyler Schaefer:** Study more on convolutions and get more familiar with the model. Meet with Mason again.
- **Conner Ohnesorge:** Finish presentation slides for the data storage solution if the solution is picked as viable.
- **Aidan Perry:** Further investigation and testing of the codebase in order to successfully see data transferring thread by thread.

---

## Tags

#weekly-report #sddec25-01 #semantic-segmentation #week-02

## Related Documents

- [[Report1Semester1]]
- [[Report3Semester1]]
