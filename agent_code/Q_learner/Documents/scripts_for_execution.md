#### Training

for training the model once and debugging the code
```bash
python main.py play --agents Q_learner coin_collector_agent peaceful_agent random_agent --train 1 --n-rounds 1 --scenario coin-heaven
```

For training the q-learning table in depth

```bash
python main.py play --agents Q_learner coin_collector_agent peaceful_agent random_agent --train 1 --n-rounds 10000 --scenario coin-heaven --no-gui
```

For testing the agent to other agents
```bash
python main.py play --agents Q_learner coin_collector_agent peaceful_agent random_agent --n-rounds 1 --scenario coin-heaven
```