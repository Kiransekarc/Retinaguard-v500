# RetinaGuard V500 
**An Advanced Clinical Decision Support System for Retinitis Pigmentosa**

![RetinaGuard V500](https://img.shields.io/badge/Status-Review%200%20Ready-success)
![Version](https://img.shields.io/badge/Version-5.0.0-blue)
![Domain](https://img.shields.io/badge/Domain-Medical%20AI%20%2F%20Ophthalmology-red)

## 📌 Problem Statement
Retinitis Pigmentosa (RP) is a rare genetic eye disease causing severe vision loss and blindness. While Deep Learning (AI) models have achieved high accuracy in detecting RP on clean, idealized datasets, they frequently fail in real-world clinical environments. Traditional AI acts as a "black box" that commonly misdiagnoses artifacts (camera flash, stitched image borders) or unrelated diseases (Age-Related Macular Degeneration, Diabetic Retinopathy) as RP due to over-sensitivity to dark pixels. Furthermore, standard AI models fail to detect edge-case RP variants (Sine Pigmento, Sectoral RP) and cannot adapt to poor lighting from affordable handheld cameras. 

## 📂 Domain Overview
* **Domain:** Medical Artificial Intelligence / Healthcare Informatics
* **Sub-Domain:** Ophthalmology, Retinal Image Processing
* **Focus Area:** Clinical Decision Support Systems (CDSS) for rare genetic retinal dystrophies.

---

## 📄 Base Paper & Limitations
**Reference Base Paper Concept:** *Deep learning models for the automated detection of Retinitis Pigmentosa from color fundus photographs.*

### Limitations of the Base Paper (Existing Models):
1. **Black Box Nature:** Existing AI models output a simple probability score without clinical rationale, which is legally and medically insufficient for doctors to trust.
2. **Massive False Positives:** Basic AI fails to mathematically differentiate between RP "Bone Spicules" (melanin pigment) and Diabetic "Hemorrhages" (dark red blood) or AMD pigment clumping.
3. **Inability to Handle Hardware Variance:** Models trained on $50,000 tabletop scanners fail completely when given underexposed, blurry images from affordable handheld or smartphone cameras.
4. **Variant Blindness:** Standard models only look for the "Classic Triad" and fail to diagnose rare variants like Sine Pigmento (RP without pigment) or Sectoral RP.

---

## 🚀 How RetinaGuard V500 Overcomes These Limitations
Our project completely abandons the vulnerable "Black Box" approach, replacing it with a **10-Expert Clinical Decision Support System** governed by a strict, rules-based Decision Engine.

1. **Multi-Expert Architecture:** RetinaGuard deploys 10 independent algorithmic "Clinical Experts" that mathematically extract and measure specific biological features (Vessel Density, Pigment Clusters, Optic Disc Pallor, Texture Degeneration).
2. **Graceful Degradation & Differential Diagnosis:** If the AI neural network panics due to a camera flash artifact, the Clinical Experts veto the AI. The Decision Engine safely downgrades the verdict to "Borderline/Monitor" and generates a true Differential Diagnosis (e.g., suggesting Diabetic Retinopathy instead of RP).
3. **Adaptive Camera Calibration:** We implemented dynamic color-space calibration profiles. If a handheld/smartphone camera is used, the system automatically applies Gamma/CLAHE correction and safely bypasses strict FDA quality thresholds.
4. **Color-Space Pathological Filtering:** The system analyzes images in both LAB and RGB color spaces to mathematically differentiate dark red blood (Diabetic Hemorrhage) from pure black melanin (RP Bone Spicule), completely eliminating false positives.
5. **Variant Detection Pathways:** Custom logical pathways dynamically identify Retinitis Punctata Albescens (RPA), Sine Pigmento, and Sectoral RP.

---

## 💡 Feasibility Analysis
**1. Technical Feasibility:**
* **Low Computational Overhead:** By shifting the bulk of the analysis from massive Deep Learning networks to highly optimized Mathematical/Morphological Computer Vision algorithms (OpenCV), the system runs efficiently on standard CPUs without requiring expensive cloud GPUs.
* **Modular Design:** The 10-Expert system is highly modular, meaning new experts or disease variants can be added mathematically without retraining the entire neural network from scratch.

**2. Economic Feasibility:**
* **Hardware Agnostic:** Traditional RP diagnostic tools require $50,000+ tabletop fundus scanners. Our dynamic camera calibration allows the software to accurately diagnose RP using $500 handheld or smartphone-based fundus cameras.
* **Low-Resource Clinics:** By reducing hardware costs and cloud computing requirements, this CDSS can be deployed in rural and low-resource medical clinics globally.

**3. Operational Feasibility:**
* **Clinical Trust & Legal Compliance:** The "White-Box" rules-based engine provides explicit, medically sound justifications (e.g., "Severe Vessel Attenuation: 4.4% Density") for every diagnosis, ensuring doctors can trust and legally verify the AI's decision.
* **Accessible UI:** The dashboard is built as a lightweight web application, meaning any doctor with a standard laptop and web browser can instantly use the system without complex installations.

---

## 🛠️ Technology Stack
* **Frontend:** HTML5, CSS3 (Custom Glassmorphism Medical UI), Vanilla JavaScript
* **Backend:** Python (Flask API)
* **Computer Vision:** OpenCV, NumPy (Spatial texture extraction, LAB color-space isolation)
* **Deep Learning:** TensorFlow/Keras (Initial Anomaly Detection)
* **Database:** MongoDB (Patient Progression Tracking)

## ⚙️ How to Run
1. Install dependencies: `pip install -r requirements.txt` and `npm install`
2. Start the Node.js Frontend Server: `node server.js`
3. Start the Python Flask Backend: `python app.py`
4. Access the Clinical Dashboard at `http://localhost:5000`

## 🧠 Core Diagnostic Rules (The Decision Engine)
The system evaluates the 10 Clinical Experts using a hardcoded medical framework:
* **Rule 1:** Classic RP (Triad Complete - 100% confidence)
* **Rule 2:** RP Variants (Sectoral, RPA, Sine Pigmento)
* **Rule 3:** Positive consensus (AI Confident + Multiple Clinical Votes)
* **Rule 4:** Suspicious (AI Uncertain + Peripheral Degeneration)
* **Rule 5:** Borderline (Minor artifacts, Vetoed AI)
* **Rule 6:** Negative / Healthy Retina
