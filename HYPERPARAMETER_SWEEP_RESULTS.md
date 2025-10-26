# 🔧 Hyperparameter Sweep Results for MNIST Instability

## 🎯 Objective

This document presents the results of hyperparameter sweeps conducted to address the instability issues observed in MNIST experiments, specifically for configurations that showed poor momentum performance:
- **MNIST K=5, n=4**: Showed only 6.08% improvement with M-DSA
- **MNIST K=10, n=6**: Showed -51.48% degradation with M-DSA

## 📊 Sweep Configurations

### Sweep Parameters
- **Alpha values**: [0.005, 0.01, 0.02, 0.05]
- **Beta values**: [0.8, 0.9, 0.95, 0.99]
- **Total combinations**: 16 M-DSA configurations per sweep
- **Baseline comparisons**: 3 DSA configurations (α=0.005, 0.01, 0.02)

---

## 🧪 MNIST K=5, n=4 Results

### DSA Baseline Performance
| Algorithm | Alpha | Final Error |
|-----------|-------|-------------|
| DSA | 0.005 | 0.804190 |
| DSA | 0.01  | 0.807802 |
| **DSA** | **0.02** | **0.778443** (Best) |

### M-DSA Hyperparameter Sweep Results

**Top 10 M-DSA Configurations (sorted by final error):**

| Rank | Alpha | Beta | Final Error | Improvement vs Best DSA |
|------|-------|------|-------------|------------------------|
| 1 | 0.010 | 0.95 | 0.575239 | **+26.10%** |
| 2 | 0.020 | 0.90 | 0.592422 | **+23.90%** |
| 3 | 0.010 | 0.90 | 0.595127 | **+23.55%** |
| 4 | 0.050 | 0.80 | 0.608162 | **+21.87%** |
| 5 | 0.020 | 0.80 | 0.632381 | **+18.76%** |
| 6 | 0.005 | 0.95 | 0.658093 | **+15.46%** |
| 7 | 0.020 | 0.95 | 0.678940 | **+12.78%** |
| 8 | 0.050 | 0.95 | 0.683476 | **+12.20%** |
| 9 | 0.005 | 0.90 | 0.725901 | **+6.75%** |
| 10 | 0.050 | 0.90 | 0.738229 | **+5.17%** |

**Worst M-DSA Configurations:**
| Rank | Alpha | Beta | Final Error | Improvement vs Best DSA |
|------|-------|------|-------------|------------------------|
| 14 | 0.005 | 0.80 | 0.854556 | -9.78% |
| 15 | 0.020 | 0.99 | 0.917738 | -17.89% |
| 16 | 0.050 | 0.99 | 0.954039 | -22.56% |

### Key Findings for MNIST K=5, n=4

✅ **Momentum CAN work well**: Best configuration shows **26.10% improvement**
✅ **Optimal settings**: α=0.01, β=0.95
✅ **Alpha sensitivity**: α=0.01 and α=0.02 work best
✅ **Beta sensitivity**: β=0.90-0.95 optimal range
❌ **High momentum (β=0.99)**: Consistently poor performance
❌ **Very low alpha (α=0.005)**: Mixed results

---

## 🧪 MNIST K=10, n=6 Results

### DSA Baseline Performance
| Algorithm | Alpha | Final Error |
|-----------|-------|-------------|
| DSA | 0.005 | 0.852570 |
| DSA | 0.01  | 0.645996 |
| **DSA** | **0.02** | **0.532074** (Best) |

### M-DSA Hyperparameter Sweep Results

**Top 10 M-DSA Configurations (sorted by final error):**

| Rank | Alpha | Beta | Final Error | Improvement vs Best DSA |
|------|-------|------|-------------|------------------------|
| 1 | 0.050 | 0.90 | 0.656544 | **-23.39%** |
| 2 | 0.010 | 0.95 | 0.727631 | **-36.75%** |
| 3 | 0.010 | 0.80 | 0.746795 | **-40.36%** |
| 4 | 0.020 | 0.95 | 0.790704 | **-48.61%** |
| 5 | 0.050 | 0.80 | 0.799832 | **-50.32%** |
| 6 | 0.005 | 0.80 | 0.813168 | **-52.83%** |
| 7 | 0.005 | 0.90 | 0.823578 | **-54.79%** |
| 8 | 0.005 | 0.95 | 0.827362 | **-55.50%** |
| 9 | 0.010 | 0.99 | 0.841109 | **-58.08%** |
| 10 | 0.050 | 0.95 | 0.856355 | **-60.95%** |

**Worst M-DSA Configurations:**
| Rank | Alpha | Beta | Final Error | Improvement vs Best DSA |
|------|-------|------|-------------|------------------------|
| 14 | 0.010 | 0.90 | 0.895336 | -68.27% |
| 15 | 0.005 | 0.99 | 0.911713 | -71.35% |
| 16 | 0.050 | 0.99 | 0.951701 | -78.87% |

### Key Findings for MNIST K=10, n=6

❌ **Momentum struggles**: Best configuration still **23.39% worse** than DSA
❌ **High-dimensional challenge**: K=10 is too high for current momentum settings
❌ **No positive improvements**: All M-DSA configurations perform worse than best DSA
⚠️ **Best attempt**: α=0.05, β=0.90 (still 23% worse than DSA)
❌ **High momentum (β=0.99)**: Consistently very poor performance
❌ **Low alpha (α=0.005)**: Poor performance across all beta values

---

## 📈 Hyperparameter Analysis

### Alpha Performance Patterns

**MNIST K=5, n=4:**
- **α=0.01**: Best average performance
- **α=0.02**: Second best
- **α=0.05**: Moderate performance
- **α=0.005**: Worst average performance

**MNIST K=10, n=6:**
- **α=0.05**: Best average performance (but still negative)
- **α=0.01**: Second best
- **α=0.02**: Third best
- **α=0.005**: Worst average performance

### Beta Performance Patterns

**MNIST K=5, n=4:**
- **β=0.95**: Best average performance
- **β=0.90**: Second best
- **β=0.80**: Third best
- **β=0.99**: Worst average performance

**MNIST K=10, n=6:**
- **β=0.90**: Best average performance (but still negative)
- **β=0.80**: Second best
- **β=0.95**: Third best
- **β=0.99**: Worst average performance

---

## 🎯 Recommendations

### For MNIST K=5, n=4 (SOLVED ✅)
**Optimal hyperparameters**: α=0.01, β=0.95
- **Improvement**: 26.10% over best DSA
- **Stability**: Consistent good performance
- **Alternative**: α=0.02, β=0.90 (23.90% improvement)

### For MNIST K=10, n=6 (CHALLENGING ❌)
**Current best attempt**: α=0.05, β=0.90
- **Still 23.39% worse** than best DSA
- **Recommendation**: Consider different approaches:
  1. **Adaptive step sizes**: Start with larger α, decay over time
  2. **Different momentum schedules**: β=0.9 → 0.95 → 0.99
  3. **Warmup period**: Start with DSA, switch to M-DSA after convergence
  4. **Different initialization**: Better starting points for high-dimensional spaces

### General Hyperparameter Guidelines

✅ **DO:**
- Use α=0.01-0.02 for moderate K (K≤5)
- Use β=0.90-0.95 for most cases
- Test multiple combinations systematically

❌ **AVOID:**
- β=0.99 (high momentum) - consistently poor
- α=0.005 (very low step size) - slow convergence
- High-dimensional spaces (K>8) without careful tuning

---

## 🔬 Technical Insights

### Why Momentum Works for K=5 but Not K=10

1. **Optimization Landscape**: Higher K creates more complex, non-convex landscapes
2. **Momentum Accumulation**: High momentum can overshoot in complex landscapes
3. **Consensus Dynamics**: More nodes (n=6) may amplify momentum instabilities
4. **Eigenvalue Structure**: MNIST's eigenvalue decay may be unfavorable for high K

### Hyperparameter Sensitivity

- **Alpha sensitivity**: Higher for high-dimensional problems
- **Beta sensitivity**: Critical for stability, especially in distributed settings
- **Interaction effects**: α and β interact non-linearly

---

## 📊 Generated Plots

The hyperparameter sweeps generated comprehensive visualization plots:

1. **Individual sweep plots**: 4-panel analysis for each configuration
2. **Summary plots**: Comparative analysis across all configurations
3. **Heatmaps**: α vs β performance visualization
4. **Improvement charts**: M-DSA vs DSA comparison

All plots are saved in the `results/` directory with descriptive filenames.

---

## 🚀 Next Steps

### Immediate Actions
1. **Use optimal settings** for MNIST K=5, n=4: α=0.01, β=0.95
2. **Avoid momentum** for MNIST K=10, n=6 until better hyperparameters found
3. **Document findings** in main experiment results

### Future Research
1. **Adaptive hyperparameters**: Time-varying α and β
2. **Advanced momentum**: Nesterov acceleration, Adam-style updates
3. **Warmup strategies**: Gradual momentum introduction
4. **Architecture modifications**: Different consensus mechanisms

---

## 📁 Files Generated

- `results/hyperparameter_sweep_mnist_K5_n4_*.npz`: Raw sweep data
- `results/hyperparameter_sweep_mnist_K5_n4_*_sweep_plot.png`: 4-panel sweep plots
- `results/hyperparameter_sweep_mnist_K10_n6_*.npz`: Raw sweep data
- `results/hyperparameter_sweep_mnist_K10_n6_*_sweep_plot.png`: 4-panel sweep plots

---

*Analysis completed on: October 26, 2025*  
*Total sweep experiments: 2 configurations × 16 M-DSA + 3 DSA = 35 experiments*  
*Key finding: Momentum can work for K=5 (26% improvement) but struggles with K=10*
