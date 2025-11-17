# EE/CprE/SE 491 WEEKLY REPORT 05

**Period:** 03/26/2025 – 04/01/2025

**Group Number:** sddec25-01
**Project Title:** Semantic Segmentation Optimization
**Client/Advisor:** JR Spidell/Namrata Vaswani

## Team Members/Role

- Joseph Metzen – Kria Board Manager
- Tyler Schaefer – ML Algorithm Analyst
- Conner Ohnesorge– ML Integration HWE
- Aidan Perry – Multithreaded Program Developer

---

## Weekly Summary

The team met together and discussed options with a new data version control system. Progress has been made in downloading software in order to properly run the Kria board on each of our systems to visualize how data is being passed through and how frames are processed.

---

## Past Week Accomplishments

- **Joseph Metzen:** Downloaded the right version of Vitis Ai for the Kria Board. Ran the Kria the board and login requirements.

- **Tyler Schaefer:** Researched optimization techniques for U-Nets and CNNs for embedded deployment.

- **Conner Ohnesorge:** Fixed broken docker dev environment built. Made more configurable. Made PR for it. Presented with a data version control system that I proposed, I also made a PR to add that slide deck to the repository.

- **Aidan Perry:** Formulated equation to be passed through threads to start testing. Met with previous codebase owner of multi threads to discuss options and viability of my proposition.

---

## Pending Issues

- **Joseph Metzen:** Camara would be nice to visually see the output produced.

- **Tyler Schaefer:** Picking optimization techniques as necessary and beginning to edit the existing model.

- **Conner Ohnesorge:** Fixed past teams dockerfile and tested it on windows, macos, and linux using github actions, but still the issue of downloading the installer each time is still an issue.

- **Aidan Perry:** Conflicting options with how progress have been approached to how the client and seniors have advised to move forward with testing and passing through equations through the threads.

---

## Individual Contributions

| Team Members | Individual Contributions | Hours this week | Hours Cumulative |
|--------------|-------------------------|-----------------|------------------|
| Joseph Metzen | Got the Kria Board up and going. | 5 | 29 |
| Tyler Schaefer | Researching optimizations | 5 | 30 |
| Conner Ohnesorge | 2 PRs. Created fix for Dockerfile for dev environment making it more configurable. Also presented and added slide deck to repo. | 10 | 30 |
| Aidan Perry | Met with previous multithreading developer and discussed options for testing purposes with codebase | 5 | 25 |

---

## Plans for the Upcoming Week

- **Joseph Metzen:** Run some test code with Conner and see the benchmarks of previous teams.

- **Tyler Schaefer:** Continuing research on optimizations and analyzing the computational complexity of the existing model.

- **Conner Ohnesorge:** Running the old model on the board without the quantization changes.

- **Aidan Perry:** Reformulate my options in how I will be testing due to discrepancies with the class that we are using in the multi threading. Research heavily into cv::mat.
