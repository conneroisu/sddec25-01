# EE/CprE/SE 491 Weekly Report 08

**Report Period:** April 15, 2025 – April 22, 2025
**Group Number:** sddec25-01
**Project Title:** Semantic Segmentation Optimization
**Client/Advisor:** JR Spidell/Namrata Vaswani

## Team Members/Roles

- **Joseph Metzen** – Kria Board Manager
- **Tyler Schaefer** – ML Algorithm Analyst
- **Conner Ohnesorge** – ML Integration HWE
- **Aidan Perry** – Multithreaded Program Developer

---

## Weekly Summary

The team continued hardware performance evaluation with additional RAM benchmarks and tests on the Kria Board. Research findings were presented to both the team and client, followed by the initiation of model division coding work. Comprehensive benchmarking results for RAM and on-chip memory were produced and documented in the main repository, along with the code used to generate and visualize these benchmarks. The team is now beginning more focused testing on Onnx runtime inference of the u-net model based on these findings. Work also progressed on theoretical aspects of implementing cv within the codebase, with an improved understanding of how the board operates. A meeting with an advisor has been finally scheduled to discuss potential data structure options.

---

## Past Week Accomplishments

- **Joseph Metzen:** Continued looking over benchmarks of the RAM and ran more tests on the Kria Board.

- **Tyler Schaefer:** Presented research to the team and client, began coding division of the model.

- **Conner Ohnesorge:** Produced benchmarking results for ram and other on-chip memory. Documented these in our main repository along with the code used to generate and graph the benchmarks. Am beginning to run more focused tests focusing on the Onnx runtime inference of our u-net model leveraging our findings.

- **Aidan Perry:** Continued theorizing how to use cv:matt within the codebase with an enhanced sense of how the board runs. Scheduled 1 on 1 with advisor to discuss data structure options.

---

## Pending Issues

- **Conner Ohnesorge:** Need access to the actual Onnx running code as I have been making personal models and inference code to run on the board.

- **Aidan Perry:** Previous person who I had contacted never got back to me. Had to use resources to outreach another specialist in order to come to a decision on what kind of data structure to focus on for our needs in multithreading.

---

## Individual Contributions

| Team Members | Individual Contributions | Hours this week | Hours Cumulative |
|--------------|-------------------------|-----------------|------------------|
| Joseph Metzen | - Looked through the benchmarks of the RAM | 3 | 43 |
| Tyler Schaefer | - Presentation<br>- Programming division of model | 6 | 53 |
| Conner Ohnesorge | - Onnx Model testing on the Kria board | 4 | 51 |
| Aidan Perry | - Continued research into feeding algorithm with cv:mat<br>- One on one meeting | 4 | 37 |

---

## Plans for the Upcoming Week

- **Joseph Metzen:** Help Conner run tests with the Onnx model and look into the benchmarks.

- **Tyler Schaefer:** Continuing to divide the model, working on the presentation for the semester final.

- **Conner Ohnesorge:** Starting more rigorous Onnx model testing and evals.

- **Aidan Perry:** Talk with specialist and if they are not able to give me the right direction, schedule and continue conversation with another who is more fitted to our needs.
