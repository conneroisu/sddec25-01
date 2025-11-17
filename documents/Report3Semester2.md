# EE/CprE/SE 492 Weekly Report 03

**Report Period:** 09/19/2025 – 10/02/2025
**Group Number:** sddec25-01
**Project Title:** Semantic Segmentation Optimization
**Client/Advisor:** JR Spidell/Namrata Vaswani

---

## Team Members/Roles

- **Joseph Metzen** – Kria Board Manager
- **Tyler Schaefer** – ML Algorithm Analyst
- **Conner Ohnesorge** – ML Integration HWE
- **Aidan Perry** – Multithreaded Program Developer

---

## Weekly Summary

The team focused on several key areas this week. Joseph finalized his slide deck and assisted with getting the segments running on the board. Conner made significant progress on optimizing the unet model, implementing multiple splitting techniques, and ensuring reproducible execution using pinned Docker images. He also created Python scripts to verify the behavior of the 4 split Onnx model segments and developed a Nix dev environment for the new Gitlab repository.

---

## Accomplishments

### Joseph Metzen
- Doing some final touches on slide deck
- Helped get the segments running on the board

### Tyler Schaefer
- Shared the split model with the team and client
- Worked with the team to verify the split model retained the proper accuracy

### Conner Ohnesorge
- Made multiple implementations for splitting after/before quantizing and optimizing the unet model
- Aimed for reproducible execution using pinned docker images that are easily run and interacted with by running a simple python script
- Deployed a few versions to the board
- Made python scripts that verified that the split Onnx model had/has the same behavior as the unsplit model when inputs/outputs are "stringed" together

### Aidan Perry
- Finished up the C++ code for theorizing the confirmation of the split up multi-threaded model
- Helped with the implementation for quantizing and optimizing the unet model

---

## Pending Issues

*(No pending issues reported)*

---

## Individual Contributions

| Team Members | Individual Contributions | Hours this week | Hours Cumulative |
|--------------|-------------------------|-----------------|------------------|
| **Joseph Metzen** | - Finishing up on a data management slide deck for the client to see<br>- Helped get the segments on the board | 8 | 13 |
| **Tyler Schaefer** | - Meetings with the client and team members<br>- Verifying the accuracy of the split model | 8 | 12 |
| **Conner Ohnesorge** | - Informed JR of how he setup his new gitlab repo was incorrect if he wanted us to be able to merge prs<br>- Created verification python scripts and environment for verifying that the split unet model segments have the same output as the whole model ran by itself<br>- Created docker interfacing quantized aware hardware optimization python script which optimize the onnx models to xmodels specifically optimized for working on our kv260 board<br>- Created Nix dev environment for our new gitlab repository | 7 | 11 |
| **Aidan Perry** | - Personal model work for client<br>- Confirmation of team's model quantization as well as confirming the board was accessible from ETG | 6 | 12 |

---

## Plans for the Upcoming Weeks

### Joseph Metzen
Do some data testing and compare from the previous model to our new model.

### Tyler Schaefer
Further verification of the model and checking the performance of each segment. We need to ensure that the computational complexity estimated earlier is accurate and the segments are roughly equal sizes.

### Conner Ohnesorge
**Run my prototyped basic program that uses the quantized and optimized segments of our unet model to perform inference over an input image.**

### Aidan Perry
Help with data testing and running/verifying prototypes that we have theorized.

---

## Tags
#weekly-report #semester2 #report03 #semantic-segmentation #ml-optimization

## Related Documents
- [[Report2Semester2]]
- [[Report4Semester2]]
- [[DesignDocSemester1]]
