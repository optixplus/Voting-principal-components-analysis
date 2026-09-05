# Voting Principal Components Analysis (vPCA)

This repository provides a MATLAB implementation of **voting principal components analysis (vPCA)**, a demodulation algorithm for unstable phase-shifting interferometry (PSI).

---

## 📌 Overview

vPCA combines principal components analysis (PCA), Hough voting, and iterative least-squares fitting to estimate the phase and phase-shifting angles in the presence of vibration, turbulence, and noise. The included script uses simulated PSI data to demonstrate the method and compares its results with conventional PCA.

---

## 📂 Included Method

### 1. Voting Principal Components Analysis (`vPCA.m`)

The script:

1. Simulates noisy phase-shifting interferograms with vibration and turbulence disturbances.
2. Estimates phase-shifting angles using PCA.
3. Identifies disturbance-corrupted frames through Hough voting at multiple pixels.
4. Refines the phase and phase-shifting angles using iterative least-squares fitting on the retained frames.
5. Reports phase and angle RMSE values and visualizes the results.

---

## 👨‍💻 Authors

* **Shouyu Wang** — OptiX+ Laboratory, Wuxi University
* **Javier Vargas** — CSIC
* **Chen Li** — Wuxi University
* **Wei Yu** — OptiX+ Laboratory, Wuxi University
* **Aihui Sun** — Computational Optics Laboratory, Jiangnan University

---

## ⚙️ Requirements

* MATLAB
* Image Processing Toolbox (for `fspecial` and `imfilter`)

---

## 🚀 Usage

1. Clone the repository and open MATLAB in the repository folder.
2. Run `vPCA.m`.
3. Inspect the command-window output for the detected disturbance-corrupted frames and RMSE values. The script also displays reconstructed phases, estimated phase-shifting angles, and Hough-voting results.

The simulation parameters, including the number of interferograms and disturbance ratios, are defined in the `PSI` section near the top of `vPCA.m`.

---

## 📖 Applications

* Phase demodulation for unstable phase-shifting interferometry
* Computational optical phase imaging

---

## 📜 Citation

If you use this code, please cite:

> Shouyu Wang, Javier Vargas, Chen Li, Wei Yu, and Aihui Sun, “Voting principal components analysis (vPCA): A demodulation algorithm designed for unstable phase-shifting interferometry,” Submitted.

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## ✉️ Contact

For questions, collaboration, or issues, please contact the authors or open an issue in this repository.
