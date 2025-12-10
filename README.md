# Assistive Eye Tracking System

Eduroam Accessible Site: <https://sddec25-01.sd.ece.iastate.edu/>

## Real-Time Semantic Segmentation Optimization for Medical Assistance Applications

**Project Focus**: Optimizing AI-powered eye tracking systems for real-time medical monitoring and assistive technology deployment on edge computing platforms.

---

> **Note on Project Description:** The initial project description submitted to Iowa State University by our client ([see approved projects list](https://sddec25-01.sd.ece.iastate.edu/documents/approved-projects-sddec25.pdf)) stated: *"This project will break up an existing U-Net model into code segments that can then be pipelined. The result will be slightly higher latency but also higher throughput of the algorithm."* This description represents the client's original proposal to Iowa State and has not been edited or construed by our team. Our actual research findings differ significantly from this initial description—our analysis demonstrated that model splitting on the DPU architecture does not provide the performance benefits originally anticipated by our client. Additionally, we found the project description misleading and incorrect in terms of the hardware and restrictions imposed by the client causing this very note :)

## Development

Have the choice of a few different development environments:

Overleaf (remote), Github Copilot Agents, Local Code Editor, Local AI Agent, etc.

## Academic Poster

The VisionAssist project includes a professional A0 poster for symposium presentations. The poster is generated from the same LaTeX source and assets as the main document, ensuring consistency.

**Quick Start:**
```bash
nix develop -c ltx-compile poster/poster.tex
```

For detailed poster documentation, including customization and printing guidelines, see [poster/README.md](poster/README.md).
