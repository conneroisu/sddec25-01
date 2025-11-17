# EE/CprE/SE 491 WEEKLY REPORT 07

**Date Range:** 04/09/2025–04/15/2025
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

This week, the team made progress on benchmarking and optimizing memory performance, researching computational complexity, and exploring data structures for implementation. Memory benchmarking was conducted to analyze data transfers between different cache levels and RAM. Documentation was created, detailing allocation strategies and performance trade-offs, including a comprehensive 36-page report with graphs and code examples. Research on computational complexity was finalized, with findings prepared for presentation to both teammates and clients. Outreach was made to a professor for insights on optimal data structures, while further theoretical work was done to test and visualize threaded inputs. Overall, the team focused on refining performance analysis and preparing deliverables for review.

---

## Past Week Accomplishments

- **Joseph Metzen:** Meet up with Conner to help benchmark the memory of the Kria board. For example, the data transfers between the L1 and L2 caches and the RAM. Looked at more ways of transferring data throughout the board.

- **Tyler Schaefer:** Finished the current round of research and computational complexity analysis. Prepared to present to teammates and clients this week.

- **Conner Ohnesorge:** Created memory benchmark documentation. Showing how and where we can allocate memory with the resulting slowdowns as a result. Made a report for this and created a pull request to add the report files to the repo with code, markdown, and pdf (Python for plotting, C for memory benchmarking, Markdown for documentation). Report is 36 pages long with ~12 graphs.

- **Aidan Perry:** Sent out a message to a professor to discuss possible data structures to look into as an optimal solution to our implementation. Continued theorizing fed inputs to test and visualize threads.

---

## Pending Issues

- **Conner Ohnesorge:** Play around with Onnx model on chip after reaching out to chris from oregon. Probably want to talk Wednesday about setting up the fpga board as an ssh server.

- **Aidan Perry:** Continuing to understand and theorize new fed input for the algorithm, also waiting for confirmation on tylers proposed separation of the segmentation to start threading.

---

## Individual Contributions

| Team Members | Individual Contributions | Hours this week | Hours Cumulative |
|--------------|--------------------------|-----------------|------------------|
| **Joseph Metzen** | - Studied data transfers within the Kria Board<br>- Benchmarked the RAM | 6 | 40 |
| **Tyler Schaefer** | - Finished research and analysis<br>- Preparing a Slide Deck to summarize the division | 11 | 47 |
| **Conner Ohnesorge** | - Finished the creation of memory benchmarks<br>- Created pull request to add observations and interpretations of the results (including source code used to produce results) | 13 | 47 |
| **Aidan Perry** | - Made contact with a professor to discuss data structure options<br>- Continued analysis of feeding algorithm and cv:mat | 4 | 33 |

---

## Plans for the Upcoming Week

- **Joseph Metzen:** Continue to Benchmark data within the Kria Board.

- **Tyler Schaefer:** Present my research and division of the algorithm to my teammates and client. Begin programming the division (assuming I get approval) or rework the division.

- **Conner Ohnesorge:** Play around with multicore Onnx model loading on the actual fpga board. Set up the ssh server for the team wherever we determine at the meeting on wednesday.

- **Aidan Perry:** Hopefully get a meeting with the professor who I sent a message out to in order to sit down and pick his brain on possible solutions apart from theorizing the equation I'm working on in cv::mat.
