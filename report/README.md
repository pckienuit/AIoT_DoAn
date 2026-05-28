# Report LaTeX Template

This folder contains the LaTeX report template for the AIoT Flight Face Lookup project.

## Installation Guide

To compile the LaTeX files, you need to install the following components:

### 1. Install MiKTeX

MiKTeX is the most popular LaTeX distribution for Windows.

1. Download MiKTeX from [https://miktex.org/download](https://miktex.org/download)
2. Choose the `net` installer to automatically download required packages.
3. During installation, select the option **"Install missing packages on-the-fly"** so MiKTeX can automatically download LaTeX packages when needed.
4. After installation, verify by opening PowerShell and running:

```powershell
pdflatex --version
```

If LaTeX version information is displayed, the installation was successful.

### 2. Install Strawberry Perl

Strawberry Perl is a Perl compiler required to run `latexmk` and `BibTeX`.

1. Download Strawberry Perl from [https://strawberryperl.com/](https://strawberryperl.com/)
2. Choose the version compatible with your system (64-bit is usually the default).
3. Run the installer and follow the prompts (just click next -> next).
4. Verify by opening PowerShell and running:

```powershell
perl --version
```

### 3. Install Visual Studio Code + LaTeX Workshop

If you use VS Code for editing LaTeX:

1. **Install Visual Studio Code** if not already installed:
   - Download from [https://code.visualstudio.com/](https://code.visualstudio.com/)

2. **Install LaTeX Workshop extension**:
   - Open VS Code, press `Ctrl+Shift+X` to open the Extensions panel.
   - Search for "LaTeX Workshop".
   - Select the extension by `James Turner` (ID: `James-Yu.latex-workshop`).
   - Click **Install**.

3. **Configure LaTeX Workshop** (optional, add to `settings.json`):

```json
{
  "latex-workshop.latex.tools": [
    {
      "name": "latexmk",
      "command": "latexmk",
      "args": [
        "-synctex=1",
        "-interaction=nonstopmode",
        "-file-line-error",
        "-pdf",
        "-outdir=%OUTDIR%",
        "%DOC%"
      ],
      "env": {}
    }
  ],
  "latex-workshop.latex.recipes": [
    {
      "name": "latexmk (pdflatex)",
      "tools": ["latexmk"]
    }
  ],
  "latex-workshop.view.pdf.viewer": "tab"
}
```

## Compilation

The template uses pdfLaTeX with `extarticle`, `T5`, `inputenc` and `lmodern`, following the standard course report format.

### Method 1: Using latexmk (recommended)

Open PowerShell in the `report` directory and run:

```powershell
cd report
latexmk -pdf main.tex
```

### Method 2: Manual compilation

If you don't have `latexmk`, run the following commands sequentially:

```powershell
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Output PDF file: `report/main.pdf`.

## Recommended Structure

- `main.tex`: main file, declares packages and includes chapters.
- `chapters/00_frontmatter.tex`: cover page, acknowledgments, abstract.
- `chapters/01_introduction.tex`: project introduction, objectives, scope.
- `chapters/02_background_requirements.tex`: theoretical basis and system requirements.
- `chapters/03_system_architecture.tex`: overall Web - Server - Edge architecture.
- `chapters/04_design_implementation.tex`: design and implementation of each component.
- `chapters/05_evaluation.tex`: testing, benchmarks, evaluation.
- `chapters/06_conclusion.tex`: conclusion and future work.
- `chapters/appendix.tex`: appendix: installation, API, configuration.
- `references.bib`: bibliography.
- `figures/`: place for images, diagrams, demo screenshots.

## Items to Add

- School, department, course, instructor, and student group information.
- Web interface screenshots, MaixCAM images, recognition pipeline demo images.
- Final benchmark data if changed.
- Bibliography following course requirements.

## Report Content Checklist

### 1. Introduction

- [ ] Present the background of the flight information lookup problem at airports/kiosks.
- [ ] Explain the reasons for choosing face recognition and AIoT processing at edge devices.
- [ ] Identify main users: passengers, admin, MaixCAM device.
- [ ] Describe project objectives and prototype scope.
- [ ] Summarize the end-to-end demo flow: book ticket -> register face -> sync edge -> recognize -> display info.

### 2. Theoretical Background and Technologies

- [ ] Explain AIoT and edge computing.
- [ ] Explain the face recognition problem: detection, landmark, alignment, embedding, matching.
- [ ] Include cosine similarity/distance formula.
- [ ] Explain vector databases and why Qdrant is used.
- [ ] Present main technologies: FastAPI, SQLite/MySQL, Qdrant, MaixCAM, Web Crypto API.

### 3. Data and Preprocessing

- [ ] List datasets used: CelebA, CASIA-WebFace, calibration images.
- [ ] Describe image preprocessing: crop, resize, normalize, face alignment.
- [ ] Describe how calibration data is created for model export/compile.
- [ ] Discuss data issues encountered: face misalignment, poor lighting, wrong crop, unstable landmarks.
- [ ] Add sample data images/diagrams if available.

### 4. Model Training Process

- [ ] Describe objectives of each model: YOLO/face detector, V9 landmark, ArcFace P3.
- [ ] Describe architecture or role of each model in the pipeline.
- [ ] Record training configuration: loss, optimizer, epoch, batch size, learning rate if applicable.
- [ ] Describe fine-tuning process and reasons for fine-tuning.
- [ ] Present ONNX -> cvimodel export process.
- [ ] Add model benchmark table on PC/browser/MaixCAM if available.
- [ ] Discuss difficulties when models work well on PC but differ when deployed to MaixCAM.

### 5. Face Recognition Pipeline

- [ ] Describe pipeline: camera frame -> detect face -> crop/adaptive padding.
- [ ] Describe V9 landmark -> alignment.
- [ ] Describe ArcFace P3 -> embedding 128D.
- [ ] Explain L2 normalize.
- [ ] Explain cosine threshold for match/no match.
- [ ] Describe multi-frame averaging/quality gate on web if applicable.
- [ ] Add recognition pipeline diagram.

### 6. Face Registration Flow on Web

- [ ] Describe upload/webcam flow.
- [ ] Describe quality gates: brightness, face score, pose/landmark.
- [ ] Describe how vectors are created and sent to server.
- [ ] Describe AES-GCM Web -> Server.
- [ ] Link to endpoint `/api/face/register`.
- [ ] Add face registration UI screenshot.

### 7. Server, Database and Qdrant

- [ ] Describe main tables/features: passenger, flight, booking, payment, face registration.
- [ ] Describe how metadata is stored in SQL.
- [ ] Describe how vectors are stored in Qdrant.
- [ ] Describe link between `booking_id` and `qdrant_point_id`.
- [ ] Describe face register/match/sync API.
- [ ] Explain why Qdrant is used instead of SQL only.
- [ ] Add important endpoint table.

### 8. Edge Sync Flow

- [ ] Describe admin/server trigger sync.
- [ ] Describe Edge calling `/api/sync/{flight_id}`.
- [ ] Describe server filtering vectors by flight.
- [ ] Describe XTEA-CTR encryption before sending to MaixCAM.
- [ ] Describe encrypted cache on MaixCAM and only decrypt in RAM during match.
- [ ] Describe TTL/cache lifecycle/fallback server on cache miss.
- [ ] Add sync flow sequence diagram.

### 9. MaixCAM Edge Deployment

- [ ] Present hardware limitations: 128MB RAM, RISC-V C906, TPU, camera.
- [ ] Describe role of files: `main.py`, `sync_cache.py`, `display.py`, `mjpeg_server.py`.
- [ ] Describe realtime pipeline on device.
- [ ] Describe MJPEG stream instead of physical LCD if used in demo.
- [ ] Describe handling face occlusion, camera freeze, app restart.
- [ ] Add device/demo MaixCAM images.

### 10. Security and Privacy

- [ ] Note if face vectors are sensitive biometric data.
- [ ] Describe AES-GCM Web -> Server.
- [ ] Describe XTEA-CTR Server -> Edge.
- [ ] Explain why plaintext cache is not saved on SD card.
- [ ] Discuss remaining risks: key management, replay attack, no liveness detection.
- [ ] Propose security improvements for production.

### 11. Testing and Evaluation

- [ ] List available unit/integration tests.
- [ ] Describe crypto E2E test.
- [ ] Describe device sync test.
- [ ] Describe edge integration test.
- [ ] Add MaixCAM latency/FPS benchmark.
- [ ] Add stress test 10/50/100/200 passengers.
- [ ] Discuss results, limitations, and reliability of measurements.

### 12. Challenges and Solutions

- [ ] Models behave differently on PC vs MaixCAM deployment.
- [ ] Crop/padding deviation reduces similarity.
- [ ] Cosine threshold too strict or too loose.
- [ ] RAM and performance limitations on edge.
- [ ] Cache sync, stale data, and orphan vectors.
- [ ] Camera freeze/restart.
- [ ] Web ONNX/browser pipeline and missing data.
- [ ] Encoding, documentation, logging/debugging across environments.
- [ ] Document solutions applied for each challenge.

### 13. Demo and Presentation Script

- [ ] Prepare sample bookings.
- [ ] Register sample faces.
- [ ] Trigger sync to MaixCAM.
- [ ] Recognize using MaixCAM camera.
- [ ] Display flight information.
- [ ] Open admin dashboard to observe status.
- [ ] Record demo video link if available.

### 14. Work Distribution

- [ ] Web frontend.
- [ ] Backend/API/database.
- [ ] Training/export model.
- [ ] Edge deployment.
- [ ] Testing/benchmark.
- [ ] Report/slide/demo.

### 15. Appendix

- [ ] API endpoints.
- [ ] Directory structure.
- [ ] Commands to run server/web/Qdrant.
- [ ] MaixCAM deployment commands.
- [ ] Configuration files `.env`, `config.json`.
- [ ] Repo/video demo links.
