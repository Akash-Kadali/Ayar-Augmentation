<div align="center">
<h1>🚀 Customized Dual-Interrelated Diffusion Model for Semiconductor Defect Augmentation</h1>
<h3>Powered by Ayar Labs | Based on CVPR 2025 Paper</h3>
<br>

<p>
Adapted and optimized by <strong>Sri Akash Kadali</strong> during my internship at <strong>Ayar Labs</strong>, this project is a customized implementation of the CVPR 2025 work <em>"DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation"</em>. This version is restructured specifically for semiconductor die defect augmentation across Critical, Laser, and Body regions.
</p>

<a href="https://arxiv.org/abs/2408.13509"><img src="https://img.shields.io/badge/arXiv-2408.13509-A42C25.svg" alt="arXiv"></a>

</div>

---

## 🧠 Abstract

Anomaly inspection in semiconductor manufacturing suffers due to extremely scarce labeled defect images. We solve this using **DualAnoDiff**, a powerful diffusion-based model that generates highly realistic and diverse synthetic defects.

Unlike traditional GANs, DualAnoDiff leverages two interrelated diffusion paths: one generates the full image while the other generates only the defect region, improving spatial consistency and realism.

In this customized implementation:

* We use real semiconductor chip images from Ayar Labs.
* Perform region-specific augmentation (e.g., Critical area only).
* Integrate outputs into downstream defect classification pipelines.

---

## 🧰 My Customizations at Ayar Labs

* ✅ Region-specific training on <code>Critical</code>, <code>Laser</code>, and <code>Body</code>
* ✅ Patched all path variables and configs for `/notebooks/my_mvtec/LaserDefect`
* ✅ Integrated Hugging Face's <code>runwayml/stable-diffusion-v1-5</code>
* ✅ Enabled <code>accelerate</code> GPU launch with precision & memory flags
* ✅ Used mixed precision and torch GPU configs to reduce OOM
* ✅ Future scope: Integrate synthetic defects into YOLO defect detector for performance benchmarking

---

## 🔧 Installation & Setup

```bash
# Step 1: Clone & Get SD weights
git clone https://github.com/yinyjin/DualAnoDiff.git
cd DualAnoDiff/dual-interrelated_diff

# Download stable diffusion weights
git clone https://huggingface.co/stable-diffusion-v1-5 stable-diffusion-v1-5

# Step 2: Install dependencies
pip install -r requirements.txt
```

---

## 📁 Folder Structure (Custom)

```
/notebooks/my_mvtec/LaserDefect/
├── train/good/
├── test/good/
└── test/bad/           # 489 manually collected defect images
```

---

## 🚀 Training (GPU Accelerated)

```bash
cd dual-interrelated_diff
bash train.sh
```

✅ Model uses `accelerate launch` with CUDA\_VISIBLE\_DEVICES=0.
✅ Output images will be saved in `all_generate/LaserDefect/bad`.

---

## 🔍 Inference

```bash
python inference_mvtec_split.py LaserDefect bad
```

* Adjust <code>guidance\_scale</code> and <code>inference\_steps</code> for diversity
* Generated images will be saved in `all_generate` folders

---

## 🎯 Use Cases in Ayar Labs Pipeline

* ➕ Augment rare classes like Laser or Facet Damage
* ➕ Balance training datasets via synthetic image generation
* ➕ Future: Use DFMGAN-style mask editing or Poisson blending

---

## 📊 Environment Setup (Locked Versions)

```
accelerate==0.24.1
protobuf==3.20.3
clip, einops, timm, torch==2.0.1+cu118
transformers==4.30.2
torchvision==0.15.2+cu118
scikit-image, matplotlib, pandas, Pillow
```

---

## 🧪 Sample Results

<img width="700" src="https://github.com/user-attachments/assets/7128b95d-3a35-4838-ad88-c2150afdee2d" />

---

## 🥇 GAN vs Diffusion: Why Diffusion Wins

| Feature                       | GAN (e.g., DFMGAN) | DualAnoDiff (Diffusion) |
| ----------------------------- | ------------------ | ----------------------- |
| Realism of Defects            | ❌ Artifacts        | ✅ High Fidelity         |
| Diversity of Generated Images | ❌ Limited Variety  | ✅ Very Diverse          |
| Mask Alignment                | ❌ Often Misaligned | ✅ Mask is Co-generated  |
| Easy to Train                 | ✅ Fast             | ❌ Slower but Stable     |
| Few-shot Performance          | ❌ Poor             | ✅ Excellent             |

✅ In our case (semiconductor defect gen), **diffusion methods are clearly superior**, especially when defect variety and visual fidelity are crucial.

---

## 📌 Citation

```bibtex
@article{jin2024dualanodiff,
  title={DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation},
  author={Jin, Ying and Peng, Jinlong and He, Qingdong and Hu, Teng and Chen, Hao and Wu, Jiafu and Zhu, Wenbing and Chi, Mingmin and Liu, Jun and Wang, Yabiao and others},
  journal={arXiv preprint arXiv:2408.13509},
  year={2024}
}
```

---

<div align="center">
  <b>Made with ❤️ and A100s at Ayar Labs</b>
</div>
