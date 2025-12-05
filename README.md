# Dilution Refrigerator Line Noise Explorer  
*A Streamlit application for modeling thermal noise propagation through cryogenic microwave lines.*

<!-- <p align="center">
  <img src="logo.png" alt="App Logo" width="180"/>
</p> -->

---

## 📌 Overview

This application provides an interactive tool to model **thermal photon occupation**,  
**effective noise temperature**, and **noise suppression** along a microwave drive line inside a  
**dilution refrigerator**.  

It is designed for researchers working with superconducting qubits, TLS spectroscopy, and  
cryogenic microwave experiments that require quantitative understanding of:

- Thermal noise propagation  
- Noise temperature conversion  
- Effects of attenuation at different stages  
- Frequency-dependent photon statistics  
- Achieving the *single-photon limit* in experiments  

The app numerically simulates a multistage attenuation chain using a  
**thermal beam-splitter model**, allowing rapid exploration of wiring configurations and  
device-level thermal loading.

---

## ✨ Features

- Interactive adjustment of:
  - Stage temperatures (300 K → MXC)
  - Attenuation at each cryostat stage  
  - Frequency band and resolution  
- Automatic unit handling:
  - K for warm stages  
  - mK for Still / MXC  
- Computation of:
  - Mean thermal occupation \( \bar n(T, f) \)  
  - Effective temperature from occupation  
  - Noise suppression in dB relative to room temperature  
- Optional display of intermediate values (`n_eff`)  
- Beautiful, responsive plots (Plotly):
  - Photon number vs frequency  
  - Effective temperature vs frequency  
  - Noise suppression vs frequency  
- Stage-by-stage summary metrics  
- Expandable tables  
- Clean mathematical documentation inside the app  
- Optional visibility toggles for each plot  
- Fully Streamlit-Cloud compatible  
<!-- 
---

## 📷 Screenshots

> *(Replace these with actual screenshots after deployment)*

<p align="center">
  <img src="assets/screenshot1.png" width="600">
</p>

<p align="center">
  <img src="assets/screenshot2.png" width="600">
</p>

---

## 🚀 Running Locally

### **1. Clone the repository**

```bash
git clone https://github.com/USERNAME/dilution-fridge-noise-app.git
cd dilution-fridge-noise-app -->
