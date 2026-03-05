# ProxiVideoFriends: Video-Based Proxemics and Social Relationship Understanding

### 📚 Official Repository for **ProxiVideoFriends** -- Video Proxemics Dataset and Multitask Temporal Model

Developed by **Isabel Jiménez-Velasco**, **Rafael Muñoz-Salinas**,
**Vicky Kalogeiton**, and **Manuel J. Marín-Jiménez**.

<div align="center">

  📄 **Paper:** *ProxiVideoFriends: Revisiting Proxemics through Temporal
and Social Reasoning*\
🎥 **Task:** Video-based Proxemics & Social Relationship Recognition\
📊 **Dataset:** ProxiVideoFriends (Friends TV Series)

</div>

------------------------------------------------------------------------

## 🧠 Overview

Understanding **human spatial behavior** is essential for building
socially-aware artificial intelligence systems. In human interactions,
**proxemics** refers to how people use physical space and body contact
to convey social meaning.

Applications include:

-   Human-Robot Interaction
-   Socially aware virtual assistants
-   AR/VR avatars
-   Smart surveillance
-   Social interaction understanding

While many computer vision systems recognize **actions**, few models
understand **how people physically interact and relate socially over
time**.

This project introduces:

-   **ProxiVideoFriends**, the **first video dataset** specifically
    designed for proxemics analysis.
-   A **temporal multitask deep learning model** that jointly predicts:
    -   **Proxemic interactions**
    -   **Social relationships**

Unlike previous image-based approaches, this work models **temporal
dynamics in human interactions**, significantly improving performance.

The project extends previous work on **ProxemicsNet++** for still
images.

------------------------------------------------------------------------

# 🎬 ProxiVideoFriends Dataset

## Motivation

Most proxemics datasets operate on **static images**, which cannot
capture the **temporal evolution of human interactions**.

To address this limitation, we introduce **ProxiVideoFriends**, a
dataset built from the TV series *Friends* that contains rich real-world
interactions between characters.

This dataset enables the study of:

-   Physical contact dynamics
-   Social relationships
-   Multimodal cues (video + audio)

------------------------------------------------------------------------

## Dataset Characteristics

  Property                   Value
  -------------------------- -----------------------
  Source                     *Friends* -- Season 3
  Annotated Frames           **42,117**
  Annotated Pairs            **103,284**
  Proxemic Interactions      **43,629**
  Non-contact Interactions   **59,655**
  Average Clip Length        4.7 seconds
  Frame Rate                 24 fps

Each frame contains annotations for:

-   Person **bounding boxes**
-   **Character identity**
-   **Proxemic interaction type**
-   **Social relationship**

------------------------------------------------------------------------

## Proxemics Classes

Physical contact is categorized into **six proxemic classes**:

-   Hand--Hand\
-   Hand--Shoulder\
-   Shoulder--Shoulder\
-   Hand--Torso\
-   Hand--Elbow\
-   Elbow--Shoulder

------------------------------------------------------------------------

## Social Relationship Classes

Relationships follow the taxonomy of the **PISC dataset**:

-   Friends\
-   Family\
-   Couple\
-   Professional\
-   Commercial\
-   No Relation

------------------------------------------------------------------------

## Additional Annotations

The dataset also includes:

-   **2D keypoints**
-   **DensePose body maps**
-   **Bounding boxes**
-   **Character identities**
-   **Audio segments**

This allows research on **multimodal human interaction modeling**.

------------------------------------------------------------------------

# 🧠 Proposed Method

## Temporal Multitask Model

The proposed architecture jointly solves two tasks:

1️⃣ **Proxemics Classification (Multi-Label)**\
2️⃣ **Social Relationship Classification (Multi-Class)**

Learning both tasks simultaneously improves performance through **shared
social representations**.

------------------------------------------------------------------------

## Model Inputs

Each video clip is processed using **three visual inputs**:

  Input       Description
  ----------- ----------------------------------------
  Person P0   Crop of first individual
  Person P1   Crop of second individual
  Pair Crop   Joint crop containing both individuals

This design allows the model to capture:

-   Individual body posture
-   Interaction context

------------------------------------------------------------------------

## Temporal Backbone

Video features are extracted using pretrained backbones:

-   **ResNet(2+1)D**
-   **mViTv2**

Both pretrained on **Kinetics-400**.

------------------------------------------------------------------------

## Fusion Strategies

Two fusion mechanisms are explored:

### Cross-Attention Fusion

-   Exchanges information between branches
-   Produces **Contextual Attention Features**

### CLS Token Fusion

-   Inspired by Transformer architectures
-   Uses self-attention to merge representations

------------------------------------------------------------------------

## Optional Audio Branch

Audio embeddings extracted using **Whisper** can be integrated.

Observations:

-   Audio improves **relationship classification**
-   Audio has **minimal impact on proxemics**

This confirms that proxemics is mainly a **visual task**.

------------------------------------------------------------------------

# 📊 Results

## Comparison with Baselines

  Method                       mAP (Proxemics)
  ---------------------------- -----------------
  Frame-level ProxemicsNet++   17.9
  Qwen3-VL                     28.1
  Qwen3-VL Fine-Tuned          30.6
  Video Model (ResNet)         32.5
  **Ours (Multitask Model)**   **40.1**

Temporal modeling significantly improves performance.

------------------------------------------------------------------------

## Effect of Multitask Learning

  Model                       Proxemics mAP   Relationship Acc
  --------------------------- --------------- ------------------
  Proxemics Only              32.5            ---
  Relationship Only           ---             39.5
  **Joint Multitask Model**   **40.1**        **45.9**

Multitask learning enables the model to learn **shared social cues**,
improving both tasks.

------------------------------------------------------------------------

# 🧪 Repository Structure

  **COMING SOON**

------------------------------------------------------------------------

# ⚙️ Installation

**COMING SOON**

------------------------------------------------------------------------

# 🚀 Training

**COMING SOON**

------------------------------------------------------------------------

# 🔎 Inference

**COMING SOON**

------------------------------------------------------------------------

# 📚 Citation

If you use this dataset or model in your research, please cite:

``` bibtex
@inproceedings{JimenezVelasco2026,
  title={ProxiVideoFriends: Revisiting Proxemics through Temporal and Social Reasoning},
  author={Jimenez-Velasco, Isabel and Muñoz-Salinas, Rafael and Kalogeiton, Vicky and Marín-Jiménez, Manuel J.},
  booktitle={VISAPP},
  year={2026}
}
```

------------------------------------------------------------------------

# 🤝 Acknowledgments

This research was supported by:

**Project PID2023-147296NB-I00**\
Spanish Ministry of Science, Innovation and Universities.

------------------------------------------------------------------------

