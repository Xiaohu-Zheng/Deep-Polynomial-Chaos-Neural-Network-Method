# Deep-Polynomial-Chaos-Neural-Network-Method

[![CI/CD Pipeline](https://github.com/Xiaohu-Zheng/Deep-Polynomial-Chaos-Neural-Network-Method/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/Xiaohu-Zheng/Deep-Polynomial-Chaos-Neural-Network-Method/actions)
[![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Consistency Regularization-based Deep Polynomial Chaos Neural Network Method for Reliability Analysis**

## 📖 Paper

This repository implements the method presented in:

> **Consistency regularization-based deep polynomial chaos neural network method for reliability analysis**  
> Zheng, Xiaohu; Yao, Wen; Zhang, Yunyang; Zhang, Xiaoya  
> *Reliability Engineering & System Safety*, 2022, 227: 108732  
> [DOI: 10.1016/j.ress.2022.108732](https://doi.org/10.1016/j.ress.2022.108732)

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Xiaohu-Zheng/Deep-Polynomial-Chaos-Neural-Network-Method.git
cd Deep-Polynomial-Chaos-Neural-Network-Method
pip install -r requirements.txt
```

### Usage

```bash
python PCNN_Tube_DNNs.py
```

## 📁 Project Structure

```
Deep-Polynomial-Chaos-Neural-Network-Method/
├── PCNN.py                  # Main neural network implementation
├── Deep_PCE.py              # Polynomial chaos expansion
├── data_process.py          # Data processing utilities
├── pce_loss.py              # Loss functions
├── PCNN_Tube_DNNs.py        # Main example script
├── tube_fun.py              # Example function
├── tube_data/               # Data directory
├── tests/                   # Test suite
└── README.md               # This file
```

## 📊 Citation

```bibtex
@article{Zheng2022Consistency,
   author = {Zheng, Xiaohu and Yao, Wen and Zhang, Yunyang and Zhang, Xiaoya},
   title = {Consistency regularization-based deep polynomial chaos neural network method for reliability analysis},
   journal = {Reliability Engineering \& System Safety},
   volume = {227},
   pages = {108732},
   ISSN = {09518320},
   DOI = {10.1016/j.ress.2022.108732},
   year = {2022}
}
```

## 📧 Contact

- **Author**: Xiaohu Zheng
- **Email**: zhengxiaohu16@nudt.edu.cn
- **GitHub**: [@Xiaohu-Zheng](https://github.com/Xiaohu-Zheng)

---

**Star ⭐ this repository if you find it helpful!**
