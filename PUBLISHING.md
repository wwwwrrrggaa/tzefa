# Publishing Guide — Tzefa

> Where to publish this project and step-by-step instructions for each platform.

---

## Overview

Tzefa has several natural publishing targets depending on the audience:

| Platform | Audience | What You Publish |
|----------|----------|-----------------|
| **PyPI** | Python developers who `pip install` | The `tzefa` package (OCR pipeline + language + web server) |
| **GitHub Releases** | Users who want versioned downloads | Tagged releases with changelogs |
| **HuggingFace** | ML/AI community | Models, datasets, and a live demo Space |
| **Docker Hub** | Users who want one-command deployment | A container that runs the full web UI |
| **Academic** (arXiv, Papers with Code) | Researchers and educators | A paper describing the system |

The recommended order is: **GitHub → PyPI → HuggingFace → Docker Hub → Academic**.

---

## 1. GitHub (Repository Polish)

The repo is already on GitHub. These steps make it discoverable and professional.

### Add a License

Create a `LICENSE` file in the repo root. The HuggingFace repos already use **CC-BY-NC-3.0** (non-commercial). To keep things consistent:

- **CC-BY-NC-3.0** — if you want non-commercial restrictions
- **MIT** or **Apache-2.0** — if you want maximum adoption

Pick one and add the full license text as a `LICENSE` file.

### Add Repository Topics

Go to the repo page → gear icon next to "About" → add topics:

```
handwriting-recognition, ocr, computer-vision, pytorch, yolo,
trocr, custom-language, image-segmentation, flask, education
```

### Create a GitHub Release

```bash
# Tag the current state
git tag -a v0.1.0 -m "First public release — full 7-stage pipeline"
git push origin v0.1.0
```

Then go to **Releases → Draft a new release**:
- Tag: `v0.1.0`
- Title: `Tzefa v0.1.0 — Handwritten Code Recognition`
- Description: copy the "Working" section from the README
- Attach any demo images or screenshots

### Add a Shields Badge to README

```markdown
[![PyPI version](https://badge.fury.io/py/tzefa.svg)](https://pypi.org/project/tzefa/)
[![License: CC BY-NC 3.0](https://img.shields.io/badge/License-CC_BY--NC_3.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/3.0/)
```

---

## 2. PyPI (Python Package Index)

Publishing to PyPI lets anyone run `pip install tzefa`.

### Prerequisites

```bash
pip install build twine
```

### Prepare `pyproject.toml`

The current `pyproject.toml` needs a few additions before publishing:

```toml
[project]
name = "tzefa"
version = "0.1.0"
description = "End-to-end handwritten code recognition: OCR pipeline + custom language + web UI"
readme = "README.md"
license = {text = "CC-BY-NC-3.0"}      # or your chosen license
requires-python = ">=3.11"
authors = [
    {name = "Your Name", email = "you@example.com"},
]
keywords = ["ocr", "handwriting", "recognition", "computer-vision", "education"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Image Recognition",
    "Programming Language :: Python :: 3.11",
]

[project.urls]
Homepage = "https://github.com/wwwwrrrggaa/tzefa"
Repository = "https://github.com/wwwwrrrggaa/tzefa"
Issues = "https://github.com/wwwwrrrggaa/tzefa/issues"
```

Also ensure the full dependency list in `dependencies` matches what the pipeline actually needs at runtime (currently only `numpy`, `opencv-python`, `pillow` are listed — you may want to add `torch`, `torchvision`, `transformers`, `ultralytics`, `flask`, etc., or document them separately).

### Build and Upload

```bash
# 1. Build the package
python -m build

# This creates:
#   dist/tzefa-0.1.0.tar.gz
#   dist/tzefa-0.1.0-py3-none-any.whl

# 2. Upload to TestPyPI first (optional, recommended)
twine upload --repository testpypi dist/*
# Test: pip install --index-url https://test.pypi.org/simple/ tzefa

# 3. Upload to real PyPI
twine upload dist/*
```

### PyPI Account Setup

1. Go to https://pypi.org/account/register/ and create an account
2. Go to **Account settings → API tokens → Add API token**
3. Use the token with `twine upload` (it will prompt for credentials, use `__token__` as username and the token as password)

---

## 3. HuggingFace

> Detailed plan already exists in `Utils/HUGGINGFACE_PLAN.md`.

HuggingFace is the best platform for the ML components. The current setup:

| What | HF Repo | Status |
|------|---------|--------|
| Binarization model | `WARAJA/b5_model` | ✅ Uploaded |
| Line Segmentation model | `WARAJA/Tzefa-Line-Segmentation-YOLO` | ✅ Uploaded |
| Word OCR model | `WARAJA/Tzefa-Word-OCR-TrOCR` | ✅ Uploaded |
| Full pipeline demo | `WARAJA/Tzefa` (Space) | ✅ Uploaded |
| Datasets | 3 repos | Cards uploaded, bulk data pending |

### Remaining Steps

1. **Upload bulk dataset files** — run `python Tzefa_HF/scripts/upload_datasets.py` (this script is gitignored because it contains HF credentials — see `Utils/HUGGINGFACE_PLAN.md` for details)
2. **Set GPU hardware** on the Space — Settings → Hardware → T4 Small or ZeroGPU
3. **Add `HF_TOKEN` secret** to the Space — Settings → Secrets
4. **Test the Space end-to-end**
5. **Make repos public** when ready — Settings → Change visibility

### Making the Space Discoverable

- Add a good `README.md` card with a demo screenshot, description, and tags
- Link back to the GitHub repo
- Add the `handwriting-recognition`, `ocr`, `education` tags

---

## 4. Docker Hub

A Docker image lets anyone run the full web UI with one command.

### Create a `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r Tzefa_Ocr/requirements.txt

# Download model weights at build time (or mount them as volumes)
# RUN python -c "from huggingface_hub import hf_hub_download; ..."

EXPOSE 5000
CMD ["python", "Tzefa_Web/server.py"]
```

### Build and Push

```bash
# 1. Build the image
docker build -t tzefa:latest .

# 2. Tag for Docker Hub
docker tag tzefa:latest yourusername/tzefa:latest
docker tag tzefa:latest yourusername/tzefa:v0.1.0

# 3. Login and push
docker login
docker push yourusername/tzefa:latest
docker push yourusername/tzefa:v0.1.0
```

Users can then run:
```bash
docker run -p 5000:5000 yourusername/tzefa:latest
# Open http://localhost:5000
```

### Alternative: GitHub Container Registry (GHCR)

Instead of Docker Hub, you can use GitHub's own container registry:

```bash
docker tag tzefa:latest ghcr.io/wwwwrrrggaa/tzefa:latest
docker push ghcr.io/wwwwrrrggaa/tzefa:latest
```

This keeps everything within GitHub and links the image to your repo.

---

## 5. Academic Publishing

Tzefa combines custom DL models, a novel OCR pipeline, a custom programming language, and an education-oriented application — this is a publishable system.

### arXiv

1. Write a short paper (4–8 pages) describing:
   - The 7-stage pipeline architecture
   - The custom Tzefa language design (3-word instruction format, word-based numbers for OCR robustness)
   - Training details for the 3 DL models
   - Evaluation results (accuracy per stage, end-to-end success rate)
2. Submit to [arxiv.org](https://arxiv.org/) under **cs.CV** (Computer Vision) or **cs.HC** (Human-Computer Interaction)

### Papers with Code

1. Go to [paperswithcode.com](https://paperswithcode.com/)
2. Link your GitHub repo + any arXiv paper
3. This gives visibility in the ML community and links your code to benchmarks

### Conference Targets

| Venue | Fit | Deadline Cycle |
|-------|-----|---------------|
| **ICDAR** (Document Analysis and Recognition) | Core fit — handwriting recognition | Annual |
| **CHI** (Human-Computer Interaction) | Education angle — handwritten code for teaching | Annual |
| **EDM** (Educational Data Mining) | Teacher API / classroom integration angle | Annual |
| **AAAI / NeurIPS Demo Track** | System demo — end-to-end pipeline | Annual |

---

## Recommended Publishing Order

1. **Now**: Polish GitHub repo (license, topics, README badges)
2. **Now**: Complete HuggingFace setup (upload datasets, make public)
3. **Soon**: Publish to PyPI after fixing P0 bugs from `TODO.md`
4. **Soon**: Create a Docker image for easy deployment
5. **Later**: Write and submit an academic paper

---

## Quick Reference

> The `Tzefa_HF/scripts/` upload scripts are gitignored (they contain HF credentials). See `Utils/HUGGINGFACE_PLAN.md` for setup details.

| Action | Command / URL |
|--------|--------------|
| Build Python package | `python -m build` |
| Upload to PyPI | `twine upload dist/*` |
| Upload to TestPyPI | `twine upload --repository testpypi dist/*` |
| Create Git tag | `git tag -a v0.1.0 -m "msg" && git push origin v0.1.0` |
| Build Docker image | `docker build -t tzefa:latest .` |
| Push to Docker Hub | `docker push yourusername/tzefa:latest` |
| Push to GHCR | `docker push ghcr.io/wwwwrrrggaa/tzefa:latest` |
| Upload HF Space | `python Tzefa_HF/scripts/upload_space.py` |
| Upload HF models | `python Tzefa_HF/scripts/upload_models.py` |
| Upload HF datasets | `python Tzefa_HF/scripts/upload_datasets.py` |
