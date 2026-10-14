# HVQC: Hybrid Variational Quantum Classifier for Medical Images

A small hybrid quantum-classical model that classifies medical images as **disease vs. no disease**. A classical encoder squeezes each image down to 8 numbers, and a hand-built 8-qubit circuit makes the final decision. The model is built in Qiskit and PyTorch and tested on six [MedMNIST](https://medmnist.com/) tasks.

## How it works

```
image → flatten → PCA(8) → small CNN → angles (π·sigmoid) → RY encoding
      → Converging-Cascade ansatz (8 qubits) → read Z on qubit 4 → p(disease) → BCE loss → Adam
```

- **Encoding:** each of the 8 features becomes an `RY` rotation on its own qubit.
- **Converging-Cascade ansatz (1 block):** 16 trainable rotations and 7 CNOTs. Two CNOT chains (qubits 0→3 and 7→4) meet at the readout qubit 4, which has one final `RX` and `RY`. 7 CNOTs is the minimum needed to connect 8 qubits, and every parameter is a single-qubit rotation, so the parameter-shift gradient is exact.
- **Prediction:** the model predicts disease when the Pauli-Z expectation on qubit 4 is below 0.
- **Training:** the CNN and the circuit are trained together.

## Tasks (class 1 = disease)

| Dataset | Disease vs. control |
|---|---|
| PneumoniaMNIST | pneumonia vs. normal |
| BreastMNIST | malignant vs. normal/benign |
| DermaMNIST | melanoma vs. nevi |
| OCTMNIST | CNV/DME/drusen vs. normal |
| RetinaMNIST | DR grade 2–4 vs. grade 0 (grade 1 dropped) |
| PathMNIST | colorectal adenocarcinoma vs. normal mucosa |

## Notebooks

| File | What it does |
|---|---|
| `HVQC_manual_ansatz.ipynb` | Builds and explains the manual ansatz |
| `HVQC_full_project.ipynb` | Full pipeline on PneumoniaMNIST, with baselines, noise and IBM hardware |
| `HVQC_multi_dataset.ipynb` | The same pipeline across all six diseases (5 seeds) |

`HVQC_Study_Guide.pdf` explains every design choice and every result in plain language.

## Results (5 seeds, 200 train / ≤200 test images)

- Matches its classical twin (same CNN, one ordinary neuron in place of the circuit) and `efficient_su2` on accuracy, with 16 dials and 7 CNOTs against 48 dials and 14 CNOTs.
- Beats and is steadier than `real_amplitudes` (+0.052 mean accuracy, Wilcoxon p = 0.010).
- All 16 dials are alive. Gradients are healthy, and parameter-shift matches the fast simulator and Qiskit Aer to about 1e-7 or better.
- Ran on **ibm_marrakesh** (20 PneumoniaMNIST test images, 4000 shots): 18/20 correct on the ideal simulator and on the hardware.

**Honest limits:** there is no quantum advantage, since SVMs match or beat the model. Entanglement shows only a hint (breast ultrasound). The hardware sample is small, with a 95% interval of roughly 70–97%.

## Setup

```bash
pip install qiskit qiskit-aer qiskit-ibm-runtime torch scikit-learn medmnist matplotlib
```

Run the notebooks in Colab or Jupyter. IBM hardware is off by default (`RUN_HARDWARE = False`). To enable it, store your IBM Quantum API key as an environment variable or Colab secret. **Never paste it into a notebook or commit it.**

## Roadmap

More qubits and PCA components, more training images and regularisation (the model overfits: about 0.98 train vs. 0.80 test), repeated noise runs, and a larger hardware sample.

