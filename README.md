# Q-SSM: Quantum-Optimized State Space Model for Long-Horizon Multivariate Time Series Forecasting

This repository contains the official PyTorch + PennyLane implementation of **Q-SSM**, a quantum-optimized state space model designed for **long-horizon forecasting** of multivariate time series.  
Q-SSM combines a lightweight recurrent state-space backbone with a **learnable quantum gating mechanism**, ensuring both efficiency and improved accuracy over strong baselines such as **S-Mamba**, **Autoformer**, and **Informer**.

---

##  Proposed architecture
- **Quantum gating**: a learnable qubit expectation gate regulates recurrence updates.  
- **Linear-time recurrent backbone**: $\mathcal{O}(W(Fk+kd))$ complexity for input window length $W$.  
- **Residual forecasting**: decoder adds predictions to the last observed values for stability.  
- **Dataset-specific feature engineering**:  
  - **ETT**: sine/cosine encodings of hour and day-of-year.  
  - **Traffic**: sine/cosine encodings of hour-of-day, day-of-week, and weekend indicator.  
  - **Exchange**: raw daily currency series without seasonal augmentations.  

---

## Requirements
Install the required dependencies with:
```bash
pip install -r requirements.txt
