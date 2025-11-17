# EE/CprE/SE 491 Weekly Report 01

**Date Range:** 2/18/2025 - 2/25/2025
**Group Number:** sddec25-01
**Project Title:** Pipelining Semantic Segmentation Algorithm for Machine Learning
**Client/Advisor:** JR Spidell / Prof. Vaswani

## Team Members/Role

- **Joseph Metzen** – Kria board Specialist Engineer
- **Tyler Schaefer** – TBD
- **Conner Ohnesorge** – Machine Learning Integration Engineer
- **Aidan Perry** – DevOps Engineer

---

## Weekly Summary

The goal for this week was to continue making progress on getting familiar with the material that previous design groups have put together while developing this project. The group focused on onboarding and dividing the project into specific roles. Tasks completed included 1 on 1 meetings with the client and each team member.

---

## Past Week Accomplishments

### Joseph Metzen
Started off with a 1 on 1 meeting with a client. Studied a few hours on how the Kria board works. Reached out to Prof. Vaswani about the project.

### Tyler Schaefer
Had a 1 on 1 meeting with a client. This included trying to determine an area to specialize in.

### Conner Ohnesorge
Determining role and meeting with client.

### Aidan Perry
Getting familiarized with the Operating System by looking into the Docker applications and implementation. Had a 1 on 1 meeting with JR to discuss current skill set and interests within the project.

### Other Collective Efforts

Defining our scheduling problem that must be tackled during the course of the project. We will need to pipeline and separate the processes running in order to increase throughput of our algorithms and in hopes of optimizing and speeding up the processing of frames coming. The team has also put together a defined high-level diagram of the end goal of the u-net semantic segmentation pipeline:

```
┌─────────────────┐
│  UNet x (frame) │──────────┐
└─────────────────┘          │
                             ▼
                   ┌─────────────────┐
                   │   UNet seg 1    │──────────┐
                   └─────────────────┘          │
                             │                  │
                             ▼                  │
                   ┌─────────────────┐          │
                   │   UNet seg 2    │──────────┤
                   └─────────────────┘          │
                             │                  │         ┌─────────────────────┐
                             ▼                  ├────────▶│  Math Accel. (DPU) │
                   ┌─────────────────┐          │         └─────────────────────┘
                   │   UNet seg 3    │──────────┤
                   └─────────────────┘          │
                             │                  │
                             ▼                  │
                   ┌─────────────────┐          │
                   │   UNet seg 4    │──────────┘
                   └─────────────────┘
                             │
                             ▼
                   ┌─────────────────┐
                   │     Answer      │
                   └─────────────────┘
```

---

## Pending Issues

### Joseph Metzen
Still waiting on hardware to be sent. Slow start and still finding our specific roles.

### Tyler Schaefer
Trying to specialize and divide out roles.

### Conner Ohnesorge
Results from role 1 and role 2 are very tied together and seems like the majority of the work of role 2 is very reliant on the progress made in role 1. Thus, thinking that it would be beneficial to combine the roles. Going to ask in my one on one this week if this is an acceptable arrangement.

### Aidan Perry
Slow start to understand and get to know our client's needs and expectations. Still waiting to receive codebase and NDA so that we can have further understanding on what needs to be matured and optimized.

**Note:** [All waiting for NDA's to arrive to sign and for hardware to be delivered].

---

## Individual Contributions

### Joseph Metzen

- 1 on 1 meeting with the client
- Looked over the Kria board data sheet
- Also looked over past groups that lead to what the project is now and how to keep moving forward.

**Hours this week:** 6
**Hours cumulative:** 6

### Tyler Schaefer

- Had a 1 on 1 meeting with the client
- Led team meeting with the client

**Hours this week:** 3
**Hours cumulative:** 3

### Conner Ohnesorge

- Worked on further learning Unet through some toy programs and some attempts at breaking up algorithms. Met with team and advisor receiving more information about possible roles and the underlying problem solved by the project.
- Created document covering data collection and distribution regarding the data-tigris platform for advisor.
- Started communications with the team from Oregon State University, specifically Christopher Poon.
- Started collecting relevant documents into my Obsidian Vault.

**Hours this week:** 7
**Hours cumulative:** 7

### Aidan Perry

- Had a 1 on 1 meeting with our client to get to know each other personally to further understand my skill set.
- Put research into my interests within the project and what I believe I will be focusing on for the development of the material given to us.
- Put research into what past groups have contributed to the project to further my understanding on where my role is picking up from.

**Hours this week:** 5
**Hours cumulative:** 5

---

## Plans for the Upcoming Week

- **Joseph Metzen:** Signing the NDA and looking through the Kria Data Sheet more.
- **Tyler Schaefer:** Signing the NDA and getting access to the existing codebase
- **Conner Ohnesorge:** Signing NDA and explore the codebases once given access.
- **Aidan Perry:** Signing NDA for the upcoming week, Exploring the Operating System environment and how to deploy docker images and containers.

---

## Tags
#weekly-report #sddec25-01 #semester1 #week01 #onboarding
