# EE/CprE/SE 492 Weekly Report 04

**Report Period:** 10/02/2025 – 10/16/2025
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

As a group, we successfully relocated the Kria Board from ETG to the Senior Design lab for easier access and resetting. We also worked together to debug issues with getting models into XModels and the UNET algorithm segments. Going forward, we plan to create benchmark scripts for blink and eye tracking, and train the model for a visually appealing final presentation.

---

## Accomplishments

### Joseph Metzen
- Helped move the Kria Board from ETG to the Senior Design lab to reset the board easier
- Worked with Conner in debugging some problems with getting the models into XModels

### Tyler Schaefer
- Working to sort out various models inherited from past teams
- Helping to interface with client on compiling the models into xModels for the Kria board

### Conner Ohnesorge
- Model compilation created and verified optimized/compiled model works correctly on the board
- Planned out our implementation pattern for optimizing the inference of our three models (blink detection, eye tracking, and pupil identification - our split up U-net model)

### Aidan Perry
- Helped relocate the Kria board from ETG to Senior design lab to avoid having to ask ETG to keep resetting the board and for ease of use to us students over the weekend
- Debugged our 4 segments with Joey and Conner over the UNET algorithm
- Worked on reorganization of the new git repo for the Eye tracker and Blink Algorithms so that we can isolate each algorithm if problems within them occur

---

## Pending Issues

- **GitLab repository write permissions** are still not fully complete. Accidentally pushed changes straight to main and we can't fix them because JR has history protection on the main branch. Would have preferred that JR make the main branch protected and require a PR/MR to make changes to it, but to allow us even to merge into main with PR/MR's took weeks of effort consistently telling JR that this was an issue.

---

## Individual Contributions

| Team Member | Individual Contributions | Hours This Week | Hours Cumulative |
|-------------|-------------------------|-----------------|------------------|
| **Joseph Metzen** | - Helped Move the Kria Board from ETG to Senior design lab<br>- Debugged the segments to convert to Xmodels | 7 | 20 |
| **Tyler Schaefer** | - Training and sharing models with the team<br>- Interfacing with the client on compiling into xModels | 8 | 20 |
| **Conner Ohnesorge** | • Uploaded/Created Eye Track, Semantic Segmentation Segments, and Blink Model to Gitlab Model Registry for our Repository (Released v1.0.0 for each)<br>• Finally got permission to use the correct process (Pytorch) for the dpu optimized/compiled .xmodel generation process and used it and successfully ran the split segments on board!<br>• Documented the generation process for creating optimized split segments of the Unet algorithm inside of a properly defined python project and module system with tests. Used uv for environment management. Big picture we do this in the following order: parse output from training which is stored as an .onnx model and convert to pytorch using onnx2torch -> quantize using pytorch -> convert to .onnx to do the splitting -> split the model on our decided nodes -> reparse each of the new segments back into pytorch using onnx2torch again -> output to .xmodel -> compile/optimize using vai_c_xir. | 12 | 23 |
| **Aidan Perry** | - Debugging session over the segments for the UNET algorithm<br>- Kria board relocation<br>- Eye tracker and blink section revamping<br>- Started making scripts for eye tracking algorithm | 7 | 19 |

---

## Plans for the Upcoming Weeks

- **Joseph Metzen:** Need to make benchmark scripts for blink.
- **Tyler Schaefer:** Train the non-quantization aware model (QAT currently not compatible with xModels) to segment onto the board
- **Conner Ohnesorge:** Train the model so that we have visually nice to show visually for the final presentation/poster. Make individual inference script/benchmark projects for each of the segments.
- **Aidan Perry:** Need to make benchmark scripts for running eye tracking algorithm on its own.

---

## Tags
#weekly-report #semester2 #kria-board #model-compilation #xmodels #unet #benchmarking
