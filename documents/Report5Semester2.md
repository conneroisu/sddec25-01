# EE/CprE/SE 492 WEEKLY REPORT 05

**Date Range:** 10/17/2025 – 10/30/2025
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

### Accomplishments

- **Joseph Metzen:** Successfully created a Blink Detection script to output the chances of blink or not for one inference. Also helped out plan out our design poster.

- **Tyler Schaefer:** Working on coalescing information to hand off to future teams and to present for the end of the semester.

- **Conner Ohnesorge:** Improved training script performance >10x by configuring to more efficiently use gpu and upload metrics and model artifacts to gitlab's mlflow implementation. Planned our live/interactive demo for when presenting our project.

![Training Performance Screenshot](assets/report5-semester2-training.png)

- **Aidan Perry:** Created a script for accuracy in eye tracking for isolated analysis as well as helped with finishing up our analysis of the split Unet algorithm.

### Pending Issues

_None listed._

---

## Individual Contributions

| Team Members | Individual Contributions | Hours this week | Hours Cumulative |
|--------------|--------------------------|-----------------|------------------|
| Joseph Metzen | - Created a blink detection script<br>- Helped out plan our poster | 7 | 27 |
| Tyler Schaefer | - Wrote out report on splitting up the model to share with future teams<br>- Planning presentations/design doc for end of semester | 7 | 27 |
| Conner Ohnesorge | • Presented work from last period again for all members of the group<br>• Fully implemented inference for all four segment together and each segment individually<br>• Improved training etiquette and proficiency by >10x through reducing trips from cpu to gpu for data, adding better tracking for live training runs and more<br>• Planned our poster/demo for our table:<br>&nbsp;&nbsp;&nbsp;&nbsp;○ HP Display (one of the extras from Conner)<br>&nbsp;&nbsp;&nbsp;&nbsp;○ Kv260 connected to usb webcam<br>&nbsp;&nbsp;&nbsp;&nbsp;○ Telescoping tripod for usb webcam<br>&nbsp;&nbsp;&nbsp;&nbsp;○ Usb webcam | 8 | 31 |
| Aidan Perry | - Created EyeTracking script<br>- Isolated EyeTracking cpp models and Makefiles to run standalone without other ML algorithms<br>- Helped with the poster planning mentioned above. | 7 | 26 |

---

## Plans for the Upcoming Weeks

- **Joseph Metzen:** Do more planning for the poster. Run more tests to ensure high accuracy is met.

- **Tyler Schaefer:** Get the design doc and final presentation rough drafts in place and

- **Conner Ohnesorge:** Evaluate performance loss from split up inference of the UNet model.

- **Aidan Perry:** Compare performance loss of old model compared to the split model. Continue to analyse eye tracking scripts for issues when running all algorithms together. Prepare for design documentation and final poster.

---

## Tags

#weekly-report #semester2 #report5 #semantic-segmentation #ml-optimization #kria-board #unet #eye-tracking #blink-detection

## Related Documents

- [[Report4Semester2]]
- [[Report6Semester2]]
- [[DesignDocSemester1]]
