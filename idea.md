## 1. **Cost-Effective Sim-to-Real Transfer for Agile UAVs via Diffusion Models – Enhancing UAV Control through Generative Filtering**

Our research proposes an innovative framework leveraging diffusion models for cost-effective, robust sim-to-real transfer in agile unmanned aerial vehicles (UAVs). Traditional agile UAV control heavily relies on expensive external motion capture systems, limiting accessibility and scalability. This project aims to overcome these limitations by utilizing low-cost onboard sensors, including IMUs, low-resolution cameras, RGB-D arrays, and Lighthouse V2 positioning, paired with state-of-the-art conditional diffusion models for UAV state estimation.

### Three Innovative Points

1. **Generative Multi-Modal Filtering with Diffusion Models**
   - **Core Concept:** Reframe UAV state estimation as a conditional generative denoising process. Initial rough estimates from onboard sensors are progressively refined using diffusion models, which integrate multi-modal sensor data, inherently managing noisy, incomplete, or intermittent sensor readings.
   - **Innovation Value:** This generative approach surpasses conventional filtering by producing robust, probabilistic state distributions, significantly enhancing resilience to adverse sensor conditions, such as motion blur or sensor occlusion, thus enabling reliable, agile maneuvers.
2. **Weakly-Supervised Sim-to-Real Adaptation**
   - **Core Concept:** Implement extensive domain randomization within simulation environments (e.g., IsaacLab, Gazebo), followed by fine-tuning diffusion models on minimal real-world data using self-supervised consistency objectives.
   - **Innovation Value:** This significantly narrows the reality gap, facilitating reliable real-world UAV performance without extensive real-world data, thus streamlining and economizing sim-to-real transitions.
3. **Real-Time Inference Enhanced by Physical Priors**
   - **Core Concept:** Incorporate known UAV physical dynamics and kinematic constraints into the generative diffusion process, ensuring physically plausible, dynamically coherent real-time state estimations suitable for agile control.
   - **Innovation Value:** Real-time feasibility and dynamic realism are achieved, enabling practical deployment in high-speed, high-performance scenarios, greatly surpassing traditional computationally intensive state estimation methods.

### Potential Values

- **Cost Reduction:** Eliminating dependence on costly external motion capture systems drastically reduces setup expenses, enhancing accessibility for both commercial and academic research.
- **Enhanced Robustness:** Multi-modal diffusion models inherently improve UAV operation robustness in complex, unpredictable environments.
- **Improved Sim-to-Real Generalization:** Domain randomization and self-supervised fine-tuning significantly enhance real-world applicability of simulation-trained UAV models.
- **Real-Time Performance:** Designed with optimized sampling strategies, ensuring suitability for agile, real-time UAV control tasks.