# Q-Learning TurtleBot Navigation System

> **From 0% to 100% success rate** - A complete Q-learning solution that transforms a failing robot into a navigation expert

## What This Does

Train a TurtleBot to navigate complex mazes using enhanced Q-learning with **real-time visualization** and **smart reward engineering**.

**Results**: 100% success rate, 18.8 average steps to goal, stable learning in ~200 episodes

## Quick Start

```bash
git clone https://github.com/solonso/Autonomous_System.git
cd Autonomous_System/Q_learning
pip install numpy matplotlib
python realtime_demo.py
```

**Files**: `realtime_demo.py` • `turtlebot.png` • `README.md`

## The Q-Learning Breakthrough

### **Sparse Rewards → Smart Guidance**
```python
# Before: Only +1 at goal, -1 everywhere else (robot gets lost)
# After: Distance-based rewards guide every step
reward = 0.1 if closer_to_goal else -0.05  # Learning signal everywhere!
```

### **Smart Exploration Strategy**
```python
# Adaptive epsilon: Start exploring (0.9) → Gradually exploit (0.05)
epsilon = max(0.05, 0.9 * (0.999 ** episode))
```

### **Goal-Biased Initialization**
```python
# Initialize Q-table to attract robot toward goal
Q[goal_position] = 10.0  # "Hey robot, go here!"
```

## Performance Transformation

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Success Rate** | 0% | **100%** | Complete success |
| **Navigation** | Stuck/Failed | **18.8 steps** | Efficient paths |
| **Learning** | Never converges | **~200 episodes** | Stable learning |

### Learning Journey
- **Early**: 10.04 avg reward (learning basics)
- **Middle**: 11.30 avg reward (getting good)  
- **Late**: 11.17 avg reward (expert level)

## Visual Features

### **Real-Time Training Dashboard**
- Episode rewards with moving averages
- Steps-to-goal efficiency tracking
- Exploration rate visualization
- Value function heatmaps

### **Policy Visualization**
- **Arrow maps** showing optimal actions
- **Value landscapes** across the environment
- **Success analytics** and learning phases

## Technical Specs

| Parameter | Value | Why It Works |
|-----------|-------|--------------|
| Learning Rate (α) | 0.15 | Fast but stable updates |
| Discount (γ) | 0.99 | Values long-term rewards |
| Episodes | 2000 | Sufficient for mastery |
| Environment | 20×14 maze | Complex but solvable |

## Educational Value

Perfect for learning:
- **Q-learning fundamentals** with visual feedback
- **Reward engineering** techniques  
- **Exploration vs exploitation** balance
- **Policy interpretation** through arrows and heatmaps

## Code Architecture

```python
RealtimeMapEnv()        # Environment + TurtleBot visualization
RealtimeQLearning()     # Enhanced Q-learning with analytics  
train()                 # Real-time learning with progress tracking
demonstrate_policy()    # Show off the learned strategy
```

## Research Impact

Demonstrates solutions to core RL challenges:
- **Sparse reward problem** → Distance-based shaping
- **Exploration challenges** → Adaptive epsilon decay  
- **Slow convergence** → Smart initialization
- **Black box learning** → Comprehensive visualization

## Future Possibilities

- Multi-agent scenarios
- Dynamic environments  
- Transfer learning
- Deep Q-Networks (DQN)
- Real robot deployment

## Contributors

**Solomon Chibuzo Nwafor** [@solonso](https://github.com/solonso)

*Enhanced from group lab work with Mazen Elgabalawy at Universitat de Girona, Spain*

---

**Bottom Line**: This isn't just code—it's a complete transformation of how Q-learning can work when properly engineered. Watch a robot go from clueless to expert in real-time! 