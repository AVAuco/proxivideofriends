<h1 align="center">ProxiVideoFriends: Revisiting Proxemics through Temporal and Social Reasoning</h1>

<p align="center">
Developed by <b>Isabel Jiménez-Velasco</b>, <b>Rafael Muñoz-Salinas</b>,
<b>Vicky Kalogeiton</b>, and <b>Manuel J. Marín-Jiménez</b>
</p>

<p align="center">
  <a href="https://www.researchgate.net/publication/401588701_ProxiVideoFriends_Revisiting_Proxemics_through_Temporal_and_Social_Reasoning"><img src="https://img.shields.io/badge/Paper-VISAPP%202026-blue"></a>
  <a href="#dataset"><img src="https://img.shields.io/badge/Dataset-ProxiVideoFriends-green"></a>
</p>

Official repository for **ProxiVideoFriends**, the first benchmark for **video-based proxemics classification**, and for our **temporal multitask model** that jointly predicts **proxemics** and **social relationships**.

<p align="center">
  <img src="assets/teaser.png" width="800">
</p>

---

## 🌍 Overview

Understanding **how close is too close** is essential for socially aware AI systems such as **social robots**, **virtual avatars**, and **embodied agents in VR/AR**. Proxemics, the study of how humans use space and physical contact in social interactions, is a fundamental component of non-verbal communication.

Yet, despite recent progress in video understanding and multimodal reasoning, **proxemics in video remains largely unexplored**.

In this project, we introduce:

- **ProxiVideoFriends**, the **first dataset explicitly designed for proxemics estimation in video**
- a **temporal multitask model** that jointly predicts **proxemics** and **social relationships**
- a systematic comparison against both **image-based baselines** and a modern **Video-Language Model (Qwen3-VL-30B)**

Our findings show that:

- **temporal modeling matters**
- **joint spatial-social learning gives the biggest boost**
- **audio helps relationship recognition, but not proxemics**
- **modern VLMs still struggle with fine-grained physical contact understanding**

---

## 🤖 Why this matters

<img src="assets/robot.png" align="right" width="35%">

A socially intelligent system should not only detect people and actions, but also understand **how people relate in space**.

Without reliable proxemics estimation:

- a robot may invade personal space
- a virtual avatar may behave inappropriately
- a simulation may fail to reproduce realistic human interaction

Proxemics is not just geometry. It also carries **social meaning**.


---
## ❗ What is missing in prior work

<img src="assets/gap.png" align="right" width="40%">

Despite its importance, proxemics research still suffers from four major limitations:

1. **No dedicated video benchmark** for proxemics  
2. Most methods are still **frame-based**  
3. Proxemics is often modeled **without social context**  
4. Modern VLMs have not been systematically evaluated on this fine-grained task  

This repository addresses that gap.

---

## ✨ Contributions

- We introduce **ProxiVideoFriends**, the **first video benchmark for proxemics**, jointly annotated for **physical contact** and **social relationships**.
- We propose a **temporal multitask architecture** that jointly learns proxemics and relationship recognition from video.
- We benchmark against:
  - **ProxemicsNet++** as a frame-based baseline
  - **Qwen3-VL-30B** in both **zero-shot** and **fine-tuned** settings
- We show that:
  - **temporal modeling is essential**
  - **social relationships improve proxemics recognition**
  - **audio is useful for relationships but not for proxemics**

---

## 🎬 Dataset: ProxiVideoFriends

Most proxemics datasets operate on **static images**, which cannot capture the **temporal evolution of human interactions**.  
To address this limitation, we introduce **ProxiVideoFriends**, built from **Season 3 of _Friends_** to study proxemics in **dynamic, realistic social scenes**.

<p align="center">
  <img src="assets/dataset_overview.png" width="900">
</p>

**Annotations per frame.** Bounding boxes, character identities, pair-level **proxemics labels**, and pair-level **social relationship labels**.

- **Proxemics.** Hand-Hand, Hand-Shoulder, Shoulder-Shoulder, Hand-Torso, Hand-Elbow, Elbow-Shoulder.  
- **Relationships.** Friends, Family, Couple, Professional, Commercial, No Relation.

| Frames | Pairs | Avg. clip length | FPS | Episode split | Overlap |
|---:|---:|---:|---:|---:|---:|
| 42,117 | 103,284 | 4.7 s | 24 | 13 train / 12 test | None |

---

## 🧠 Method

We propose a **temporal multitask model** for joint **proxemics** and **social relationship** recognition from video.

For each target pair, the model processes three visual inputs: the crop of **person 0**, the crop of **person 1**, and a **joint crop** containing both individuals. The two person branches share weights, while the pair branch remains independent to preserve interaction context.

Each stream is encoded with a pretrained temporal video backbone, such as **ResNet(2+1)D** or **mViTv2**. The resulting embeddings are then fused using either **Cross-Attention (CA)** or **CLS-token Transformer fusion**.

The model jointly predicts **proxemics** as a **multi-label** task and **social relationship** as a **multi-class** task, optimized with **binary cross-entropy** and **cross-entropy**, respectively.

We also explore an optional multimodal extension using **Whisper audio embeddings** to evaluate whether audio further improves performance.

<p align="center">
  <img src="assets/methodology.png" width="90%">
</p>

---

## 🧪 Baselines

We compare our method against two complementary baselines.

### 1. Frame-based baseline
We use **ProxemicsNet++**, a state-of-the-art method for proxemics classification in still images, applied **frame by frame**.

### 2. Video-Language Model baseline

<img src="assets/vlm_prompt.png" align="right" width="55%">

We evaluate **Qwen3-VL-30B** as a prompt-based video baseline.

For each video sequence:

- the target pair is highlighted with colored circles
- the model receives a structured proxemics prompt
- we test both:
  - **zero-shot**
  - **fine-tuned**



---

## 📊 Results

### 🏆 Proxemics classification on ProxiVideoFriends

| Method | mAP |
|---|---:|
| ProxemicsNet++ | 17.9 |
| Qwen3-VL-30B | 28.1 |
| Qwen3-VL-30B Fine-Tuned | 30.6 |
| mViTv2 (our temporal)| 30.2 |
| ResNet(2+1)D (our temporal) | 32.5 |
| 🏆 **Ours Temporal + Multitask Model** | **40.1** |

*Best result in bold.*

These results show that:
- video models clearly outperform frame-based approaches
- multitask learning provides the strongest improvement
- our method improves over the fine-tuned VLM by **9.5 mAP**

### 🤝 Effect of multitask learning

| Training Setup | Proxemics mAP | Relationship Acc | Relationship Macro-F1 |
|---|---:|---:|---:|
| Proxemics Only | 32.5 | - | - |
| Relationship Only | - | 39.5 | 17.0 |
| **Multitask (CLS Fusion)** | **40.1** | **45.9** | **20.1** |

*Best results in bold.*

Joint learning improves both tasks, confirming that **proxemics and social relationships are complementary**.

### 🔊 Effect of audio

| Model | Proxemics mAP | Relationship Acc | Relationship Macro-F1 |
|---|---:|---:|---:|
| **Multitask (Visual Only)** | **40.1** | 45.9 | 20.1 |
| Multitask + Audio | 37.6 | **46.2** | **25.0** |

*Best result for each metric in bold.*

Audio improves **relationship classification**, but not **proxemics**.

This suggests that:
- proxemics is primarily **visual**
- relationships benefit from **speech cues**, such as tone and prosody

---

## 🎭 Qualitative results

<img src="assets/failureCases.png" align="right" width="40%">

Our model performs well on many realistic, dynamic interactions, but proxemics remains challenging.

Typical failure modes include:

- **ambiguous physical contact**, such as confusing Hand-Hand with Hand-Elbow
- **occlusions and scene complexity**
- difficult camera angles
- overlapping people in crowded scenes


---

## 📁 Repository structure

```text
ProxiVideoFriends/
├── README.md
├── assets/

```
**COMING SOON**

------------------------------------------------------------------------

## ⚙️ Installation

**COMING SOON**

------------------------------------------------------------------------

## 🏋️ Training

**COMING SOON**

------------------------------------------------------------------------

## 📏 Inference

**COMING SOON**

------------------------------------------------------------------------

## 📌 Citation

If you use this repository, dataset, or method in your research, please cite:

``` bibtex
@inproceedings{JimenezVelasco2026ProxiVideoFriends,
  author    = {Jim{\'e}nez-Velasco, I. and Mu{\~n}oz-Salinas, R. and Kalogeiton, V. and Mar{\'i}n-Jim{\'e}nez, M. J.},
  title     = {ProxiVideoFriends: Revisiting Proxemics through Temporal and Social Reasoning},
  booktitle = {Proceedings of the 21st International Conference on Computer Vision Theory and Applications (VISAPP)},
  volume    = {1},
  pages     = {107--116},
  year      = {2026},
  isbn      = {978-989-758-804-4},
  issn      = {2184-4321}
}
```

------------------------------------------------------------------------

## 🙏 Acknowledgments

This research was supported by:

**Project PID2023-147296NB-I00**\
Spanish Ministry of Science, Innovation and Universities.

------------------------------------------------------------------------

