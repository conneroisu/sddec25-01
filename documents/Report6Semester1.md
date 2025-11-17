# EE/CprE/SE 491 WEEKLY REPORT 06

**Report Period:** 04/02/2025 – 04/08/2025

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

The Kria board was further optimized, and research was conducted on splitting the model for improved computational efficiency. Petalinux configurations were developed to run the previous team's code, while the input structure for the algorithm was re-theorized. A slide deck was created to present a proposed model division, and efforts were made to optimize memory flow and performance.

---

## Past Week Accomplishments

- **Joseph Metzen:** Continued looking through the Kria Board on how to optimize memory flow and performance.

- **Tyler Schaefer:** Continued researching computational complexity and began making a slide deck to propose a split in the model.

- **Conner Ohnesorge:** Built some petalinux builds to try to run the old teams code this thursday.

- **Aidan Perry:** Continued research into cv::mat class, and began re-theorizing developing input to the algorithm to break down. Completed one on one with the client and past developers.

---

## Pending Issues

- **Conner Ohnesorge:** Still waiting for the finished product for Oregon State, have not received petalinux configurations for the image and boot configurations to put on the board.

- **Aidan Perry:** Previous understanding of the threads was incorrect, and thus the theory I had developed needed to be scrapped. Minor setback but understanding still stands.

---

## Individual Contributions

| Team Members | Individual Contributions | Hours this week | Hours Cumulative |
|--------------|-------------------------|-----------------|------------------|
| Joseph Metzen | • Studied the memory flow and optimize performance | 5 | 34 |
| Tyler Schaefer | • Continuing research | 6 | 36 |
| Conner Ohnesorge | - Built petalinux builds to run old code on thursday<br>- Experimented with OpenCV c++ "edition" to better aid in development | 4 | 34 |
| Aidan Perry | • Continued research and theorizing | 4 | 29 |

---

## Plans for the Upcoming Week

- **Joseph Metzen:** Finish testing and benchmarking the old code.

- **Tyler Schaefer:** Finishing this round of research and presenting to the team and client my proposed division of the model.

- **Conner Ohnesorge:** Test the petalinux images that I made this week with the previous team's code.

- **Aidan Perry:** Have a proposed input in the cv::mat format to begin examination of the threads and look for flaws within the current system.
