
# MVL-Loc: Leveraging Vision-Language Model for Generalizable Multi-Scene Camera Relocalization

Official implementation of the paper:

> **MVL-Loc: Leveraging Vision-Language Model for Generalizable Multi-Scene Camera Relocalization**  
> Zhendong Xiao, Shan Yang, Shujie Ji, et al.  

---

## 📘 Overview
MVL-Loc is an end-to-end **multi-scene 6-DoF camera relocalization framework** that leverages pretrained **Vision-Language Models (VLMs)** such as CLIP to improve cross-scene generalization and robustness under dynamic environments.

By introducing **language-guided semantic priors** and a **cross-modal fusion module**, MVL-Loc learns geometry-aware representations that align visual and textual cues. The method achieves state-of-the-art performance on **7Scenes** and **Cambridge Landmarks** datasets.

---

## 🚀 Features
- **Cross-Modal Alignment:** Image features (queries) attend to language embeddings (keys/values) via scaled dot-product attention.
- **Scene-Aware Generalization:** Language priors improve representation consistency across scenes.
- **Lightweight Pose Regression Head:** Predicts camera translation and rotation with improved accuracy.
- **Robustness to Dynamic Changes:** Enhanced feature stability under illumination and motion variations.

---
