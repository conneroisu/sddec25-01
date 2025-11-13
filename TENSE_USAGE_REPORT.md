# Tense Usage Analysis Report
## VisionAssist Senior Design Document (SDDEC25-01)

**Generated:** November 13, 2025
**Document Analyzed:** Real-Time Eye Tracking Through Optimized Semantic Segmentation for Medical Assistive Technology

---

## Executive Summary

This report provides a comprehensive analysis of verb tense usage across all sections of the VisionAssist senior design documentation. The analysis reveals generally strong consistency in most sections, with some areas requiring attention to maintain uniformity and clarity. Overall, the document demonstrates good adherence to academic writing conventions for technical documentation.

**Key Findings:**
- **Abstract:** Consistent present tense (appropriate)
- **Introduction:** Consistent present tense (appropriate)
- **Requirements:** Appropriate use of imperatives and present tense
- **Project Plan:** **INCONSISTENT** - mixed present perfect, present, and future tenses
- **Design:** Mostly consistent present tense with appropriate past tense for prior work
- **Testing:** Consistent present tense and imperatives (appropriate)
- **Implementation:** Appropriate mix of present perfect and present tense
- **Conclusion:** Consistent present perfect and present tense (appropriate)

---

## Section-by-Section Analysis

### 1. Abstract (abstract.tex)

**Primary Tense:** Present tense
**Consistency Rating:** ✓ Excellent

**Analysis:**
The abstract consistently uses present tense to describe what the research presents, implements, and demonstrates. This is the standard convention for academic abstracts.

**Examples:**
- "This research **presents** an innovative approach..."
- "The proposed system **implements** efficient sequential DPU scheduling..."
- "Performance optimization **targets** a reduction from 160ms..."
- "The system **leverages** the AMD Kria KV260 development board..."
- "This research directly **enhances** quality of life..."

**Recommendation:** No changes needed. The abstract follows IEEE and academic conventions appropriately.

---

### 2. Chapter 1: Introduction (01-introduction.tex)

**Primary Tense:** Present tense
**Consistency Rating:** ✓ Excellent

**Analysis:**
The introduction maintains consistent present tense throughout all subsections (Problem Statement, Intended Users, Summary). This is appropriate for:
- Describing current problems and challenges
- Explaining project objectives
- Characterizing user groups
- Outlining the project's approach

**Examples:**
- "Handicapped individuals **face** the critical challenge..."
- "Current healthcare solutions **are** reactive..."
- "Our project **focuses** on leveraging semantic segmentation..."
- "The primary clients **are** individuals with mobility impairments..."
- "Caregivers and family members **form** the secondary user group..."
- "This product **alleviates** some of the stress..."

**Recommendation:** No changes needed. Consistent and appropriate use of present tense throughout.

---

### 3. Chapter 2: Requirements & Standards (02-requirements.tex)

**Primary Tenses:** Imperative mood (for requirements), Present tense (for descriptions)
**Consistency Rating:** ✓ Excellent

**Analysis:**
This chapter appropriately uses:
- **Imperative mood** for functional requirements ("Optimize", "Implement", "Ensure", "Achieve")
- **Present tense** for describing standards, constraints, and existing conditions
- **Modal verbs** ("must", "will") for expressing obligations and future constraints

**Examples - Requirements (Imperatives):**
- "**Optimize** the U-Net semantic segmentation algorithm..."
- "**Implement** a pipelined architecture..."
- "**Ensure** the pipeline maintains data consistency..."
- "**Achieve** a system throughput of less than 33.2 ms..."

**Examples - Descriptions (Present Tense):**
- "The Xilinx Kria K26 board **has** 4GB of DDR memory..."
- "This standard **provides** clear terms and definitions..."
- "The following IEEE standards **are** applicable to this project..."

**Minor Note:**
One instance of future tense: "The system **will continue** to be deployed..." (line 102)

**Recommendation:** Consider changing "will continue to be deployed" to "continues to be deployed" or "is deployed" for consistency with the present tense used elsewhere in constraints. Otherwise, this section is well-structured.

---

### 4. Chapter 3: Project Plan (03-project-plan.tex)

**Primary Tenses:** Mixed (Present Perfect, Present, Future)
**Consistency Rating:** ⚠ **NEEDS IMPROVEMENT**

**Analysis:**
This chapter shows the most significant tense inconsistency in the document. It mixes:
- **Present perfect** ("has adopted", "have been identified")
- **Present tense** ("involves", "is", "means")
- **Future tense** ("will utilize", "will be held", "will ensure", "will span")

**Examples of Inconsistency:**

*Project Management section (lines 5-21):*
- "Our team **has adopted** a hybrid Waterfall + Agile project management approach..." (present perfect)
- "This methodology **provides** us with both..." (present tense)
- "For project tracking, the team **will utilize** the following tools..." (future tense)
- "Weekly team meetings **will be held**..." (future tense)

*Task Decomposition section (line 25):*
- "Our project **involves** optimizing the semantic segmentation..." (present tense)
- "The key objective **is** to increase system throughput..." (present tense)

*Milestones section (lines 76-129):*
- "The following key milestones **have been identified**..." (present perfect)
- "For each milestone, our team **will track** progress..." (future tense)

*Timeline section (lines 131-151):*
- "The project **will span** approximately 16 weeks..." (future tense)
- "The project timeline **follows** a structured approach..." (present tense)

**Problem:**
The mixing of tenses creates ambiguity about whether the document describes:
1. A plan that was made (past/present perfect)
2. A plan that currently exists (present)
3. A plan for future activities (future)

**Recommendation - Option A (Present Tense - Recommended):**
Convert the entire chapter to present tense to describe the current project plan:
- "Our team **adopts** a hybrid approach..." or "Our team **uses** a hybrid approach..."
- "For project tracking, the team **utilizes** the following tools..."
- "Weekly team meetings **are held**..." or "Weekly team meetings **occur**..."
- "The project **spans** approximately 16 weeks..."
- "Our team **tracks** progress using the following quantifiable metrics..."

**Recommendation - Option B (Consistent Future Tense):**
Alternatively, use future tense consistently for all planned activities:
- "Our team **will adopt** a hybrid approach..."
- "Weekly team meetings **will be held**..."
- "The project **will span** approximately 16 weeks..."

**Preferred Approach:** Option A (present tense) is more aligned with standard technical documentation practices, where project plans are described as current frameworks rather than future intentions.

---

### 5. Chapter 4: Design (04-design.tex)

**Primary Tense:** Present tense (with appropriate past tense for prior work)
**Consistency Rating:** ✓ Good (with minor observations)

**Analysis:**
The design chapter predominantly uses present tense to describe the current design, which is appropriate. Past tense is correctly used when discussing prior work and previous implementations.

**Examples - Present Tense (Current Design):**
- "Our Semantic Segmentation Optimization project **is situated** in the healthcare..."
- "Our project directly **improves** the safety and well-being..."
- "Our optimization approach **provides** significant performance improvements..."
- "The key innovation in our design **is** the approach to pipelined processing..."

**Examples - Past Tense (Prior Work):**
- "Wang et al. (2021) **proposed** 'EfficientEye'..."
- "Their approach **reduced** model size but processing speed **remained** at approximately 120 ms..."
- "Previous Project Iteration **implemented** a standard U-Net architecture..."
- "...but **could only process** a single frame every 160ms..."

**Observation - Decision-Making Section (lines 150-161):**
This subsection uses past tense to describe design decisions that were made:
- "Our client **required** absolutely no decrease in accuracy..."
- "We **could not let** semantic segmentation starve the other algorithms..."
- "Our design **prioritized** balanced system performance..."
- "The approach **maintained** critical timing needs..."
- "We **selected** a scheduling approach that minimizes memory transfer overhead..."

This shift to past tense suggests these are completed decisions. However, the following sentences return to present tense:
- "This approach **is** feasible because our model uses fixed memory access patterns..."
- "The system **allocates** appropriate DPU time slices..."

**Minor Issue:**
The transition between past tense (describing decision-making process) and present tense (describing the resulting design) is somewhat abrupt.

**Recommendation:** Consider one of the following approaches for consistency:

**Option A - Present Tense Throughout:**
Reframe the decision-making narrative in present tense:
- "Our client **requires** absolutely no decrease in accuracy..."
- "We **cannot let** semantic segmentation starve the other algorithms..."
- "Our design **prioritizes** balanced system performance..."
- "The approach **maintains** critical timing needs..."
- "We **select** a scheduling approach that minimizes memory transfer overhead..."

**Option B - Maintain Past Tense with Clearer Transition:**
Keep past tense for decision-making but add transitional language:
- "Based on these decisions, the current design **is** feasible because..."
- "The resulting system **allocates** appropriate DPU time slices..."

**Preferred Approach:** Option A (present tense) for better consistency with the rest of the design chapter.

---

### 6. Chapter 5: Testing (05-testing.tex)

**Primary Tenses:** Present tense, Imperatives (for test procedures)
**Consistency Rating:** ✓ Excellent

**Analysis:**
The testing chapter maintains consistent present tense throughout for describing the testing strategy and philosophy. Test procedures appropriately use imperatives or infinitives.

**Examples - Strategy Description (Present Tense):**
- "Testing **is** key to our Semantic Segmentation project."
- "We **need** to make sure our system meets our goals..."
- "We **test** early and often."
- "This **helps** us catch problems quickly..."

**Examples - Test Procedures (Imperatives/Infinitives):**
- "**Verify** system stability under concurrent processing demands"
- "**Run** system under load with varying periodic collection requirements"
- "**Feed** many eye images continuously"
- "**Test** with images in different lighting"

**Observation:**
The informal, accessible tone ("We test early and often", "Our project has some tough testing challenges") is somewhat different from the more formal tone in other chapters. While this makes the testing strategy more approachable, it creates slight stylistic inconsistency with other sections.

**Recommendation:** The tense usage is appropriate. If desired, you could elevate the formality slightly to match other chapters, but this is a stylistic choice rather than a tense consistency issue.

---

### 7. Chapter 6: Implementation (06-implementation.tex)

**Primary Tenses:** Present Perfect (for completed work), Present tense (for descriptions)
**Consistency Rating:** ✓ Excellent

**Analysis:**
The implementation chapter appropriately uses:
- **Present perfect** to describe completed milestones and achievements
- **Present tense** to describe how systems and processes work
- **Past tense** occasionally for specific completed actions

This combination is ideal for an implementation chapter that reports progress while explaining technical details.

**Examples - Present Perfect (Completed Work):**
- "Our project **has completed** comprehensive performance evaluation..."
- "We **have conducted** extensive benchmarking..."
- "The U-Net semantic segmentation algorithm **has been thoroughly analyzed**..."
- "Training experiments **are tracked** using MLflow..."

**Examples - Present Tense (How It Works):**
- "The baseline U-Net implementation **demonstrates** the following performance characteristics..."
- "The split model approach **incurs** additional overhead..."
- "The training implementation **prioritizes** GPU-native operations..."
- "The optimized PyTorch training script **achieved** remarkable performance improvements..."

**Minor Observation - Future Implementation Plans (lines 162-187):**
The subsections use imperative forms (which can also function as infinitives):
- "**Complete** implementation of pipelined U-Net processing..."
- "**Optimize** sequential DPU scheduling system..."
- "**Achieve** target pipelined performance..."

**Alternative Consideration:**
These could be reframed with future tense for clarity:
- "**Will complete** implementation of pipelined U-Net processing..."
- "**Will optimize** sequential DPU scheduling system..."
- "**Will achieve** target pipelined performance..."

However, the current imperative/infinitive form is acceptable for listing planned tasks.

**Recommendation:** No changes needed. The tense usage effectively distinguishes between completed work (present perfect) and technical descriptions (present tense). The future plans section is acceptable as-is but could optionally be made more explicit with future tense.

---

### 8. Chapter 7: Conclusion (07-conclusion.tex)

**Primary Tenses:** Present Perfect (accomplishments), Present tense (ongoing impact and reflections)
**Consistency Rating:** ✓ Excellent

**Analysis:**
The conclusion appropriately uses:
- **Present perfect** to summarize what has been achieved
- **Present tense** to describe ongoing significance and impact
- **Modal verbs** to discuss potential future outcomes

**Examples - Present Perfect (Accomplishments):**
- "This project **has successfully addressed** the critical challenge..."
- "Our approach **has demonstrated** significant progress..."
- "The journey from concept to implementation **has provided** valuable insights..."

**Examples - Present Tense (Impact and Significance):**
- "Our project **contributes** to the field of embedded AI systems..."
- "This approach **has** broader implications for..."
- "The potential social impact **extends** beyond technical achievements..."
- "The success of this project **is not only measured** in technical metrics..."

**Examples - Modal Verbs (Potential):**
- "...improvements **can be achieved** through intelligent sequential DPU scheduling..."
- "...that **can overcome** significant hardware limitations..."
- "...technology **empowers** individuals with disabilities..."

**Recommendation:** No changes needed. The conclusion effectively uses tense to distinguish between completed work and ongoing/future impact.

---

## Overall Tense Consistency Summary

### Sections with Excellent Consistency
✓ **Abstract** - Consistent present tense
✓ **Introduction** - Consistent present tense
✓ **Requirements** - Appropriate imperatives and present tense
✓ **Testing** - Consistent present tense and imperatives
✓ **Implementation** - Appropriate present perfect and present tense
✓ **Conclusion** - Appropriate present perfect and present tense

### Sections Needing Improvement
⚠ **Project Plan** - Inconsistent mixing of present perfect, present, and future tenses
⚠ **Design (Decision-Making subsection)** - Abrupt transition between past and present tense

---

## Detailed Recommendations

### Priority 1: Project Plan Chapter (03-project-plan.tex)

**Current Issue:** Inconsistent mixing of tenses creates ambiguity about the temporal status of the plan.

**Recommended Changes:**

1. **Project Management section (lines 5-21):**
   - Change "has adopted" → "adopts" or "uses"
   - Change "will utilize" → "utilizes"
   - Change "will be held" → "are held" or "occur"
   - Change "will ensure" → "ensures"

2. **Timeline section (lines 131-151):**
   - Change "will span" → "spans"
   - Change "will track" → "tracks"

3. **Task Decomposition section (line 74):**
   - Change "will be further broken down" → "are further broken down"

4. **Milestones section (line 77):**
   - Change "have been identified" → "are" or "include"
   - Change "will track" → "tracks"

**Alternative:** If you prefer to emphasize future activities, consistently use future tense throughout the chapter. However, present tense is more standard for technical project plans.

### Priority 2: Design Chapter - Decision-Making Subsection (04-design.tex, lines 150-161)

**Current Issue:** Abrupt transition between past tense (decision narrative) and present tense (design description).

**Recommended Changes:**

Convert past tense decision narrative to present tense:
- "required" → "requires"
- "could not let" → "cannot let"
- "prioritized" → "prioritizes"
- "maintained" → "maintains"
- "enabled" → "enables"
- "ensured" → "ensures"
- "selected" → "select"

**Rationale:** This maintains consistency with the rest of the design chapter and presents the design rationale as current reasoning rather than historical narrative.

### Priority 3: Minor Refinements

**Chapter 2 (Requirements), line 102:**
- Change "will continue to be deployed" → "continues to be deployed" or "is deployed"

**Chapter 6 (Implementation), Future Plans sections (optional):**
- Consider changing imperative forms to explicit future tense for clarity:
  - "Complete" → "Will complete"
  - "Optimize" → "Will optimize"
  - "Achieve" → "Will achieve"

---

## Conventions for Technical Documentation

For reference, here are standard tense conventions for senior design and technical documentation:

### Abstract
- **Present tense** for what the paper presents/demonstrates/describes
- Example: "This research presents...", "The system implements..."

### Introduction
- **Present tense** for current problems and project objectives
- Example: "Individuals face challenges...", "Our project focuses on..."

### Background/Related Work
- **Past tense** for previous research
- **Present tense** for established facts and current state
- Example: "Smith et al. (2020) proposed..." vs. "Current systems use..."

### Requirements
- **Imperative mood** or **modal verbs** for requirements
- **Present tense** for constraints and conditions
- Example: "The system shall/must..." or "Implement...", "The board has 4GB memory..."

### Design
- **Present tense** for proposed/current design
- **Past tense** when referencing prior work or previous designs
- Example: "The system uses...", "Previous implementations employed..."

### Implementation
- **Present perfect** for completed work
- **Present tense** for how things work
- **Future tense** for planned work
- Example: "We have implemented...", "The system operates...", "We will complete..."

### Testing
- **Present tense** for test strategy
- **Imperative** or **future tense** for test procedures
- Example: "Our strategy is...", "Test the system..." or "We will test..."

### Results
- **Past tense** for what was done and observed
- **Present tense** for current analysis
- Example: "The system achieved...", "These results indicate..."

### Conclusion
- **Present perfect** for accomplishments
- **Present tense** for current significance
- **Modal verbs** for future potential
- Example: "This project has demonstrated...", "The results show...", "This approach can enable..."

---

## Conclusion

The VisionAssist senior design document demonstrates strong tense consistency in most sections, with clear and appropriate tense choices that align with academic and technical writing conventions. The primary area requiring attention is the **Project Plan** chapter, which would benefit from consistent present tense usage throughout.

**Summary of Action Items:**

1. **High Priority:** Revise Chapter 3 (Project Plan) for consistent tense usage
2. **Medium Priority:** Smooth transition in Chapter 4 (Design - Decision-Making section)
3. **Low Priority:** Minor adjustments in Chapters 2 and 6 as noted above

**Overall Assessment:** The document is well-written with professional technical communication. Addressing the identified inconsistencies will enhance clarity and maintain the high quality of the documentation.

---

**Report Prepared By:** Claude Code Analysis
**Date:** November 13, 2025
**Files Analyzed:** main.tex, abstract.tex, sections/01-introduction.tex through sections/07-conclusion.tex
