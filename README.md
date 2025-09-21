# Robust Physics-Informed Neural Network Approach for Estimating Heterogeneous Elastic Properties from Noisy Displacement Data

![Uploading 21_02_M_IEPINNFramework.png…]()

## 📄 Abstract
Accurately estimating spatially heterogeneous elasticity parameters, particularly Young's modulus and Poisson's ratio, from noisy displacement measurements remains a significant challenge in inverse elasticity problems. Existing inverse estimation techniques are often limited by instability, high noise sensitivity, and difficulties in recovering the absolute scale of Young's modulus. This work presents a novel Inverse Elasticity Physics-Informed Neural Network (IE-PINN) to robustly reconstruct heterogeneous elasticity distributions from noisy displacement data based on the principles of linear elasticity. The IE-PINN incorporates three distinct neural network architectures, each dedicated to modeling displacement fields, strain fields, and elasticity distributions. This approach significantly enhances stability and accuracy under measurement noise. Additionally, a two-phase estimation strategy is proposed: the first phase recovers relative spatial distributions of Young's modulus and Poisson's ratio, while the second phase calibrates the absolute scale of Young's modulus using boundary loading conditions. Methodological innovations, including positional encoding, sine activation functions, and a sequential pretraining strategy, further improve the model's performance and robustness. Extensive numerical experiments demonstrate that IE-PINN effectively overcomes critical limitations faced by existing methods, providing accurate absolute-scale elasticity estimations even under severe noise conditions. This advancement holds substantial potential for clinical imaging diagnostics and mechanical characterization, where measurements typically encounter substantial noise.


## 📄 Data



## 📄 Citation

