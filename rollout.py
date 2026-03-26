import torch
import numpy as np
from train_model import WorldModel
import pickle

# 加载数据（拿初始状态）
with open("panda_data.pkl", "rb") as f:
    data = pickle.load(f)

state = data[0][0]   # 初始状态

# 加载模型
input_dim = len(state) + len(data[0][1])
state_dim = len(state)

model = WorldModel(input_dim, state_dim)
model.load_state_dict(torch.load("world_model.pt"))
model.eval()

# rollout（模型自己预测未来）
traj = []

for t in range(100):
    action = np.random.uniform(-5, 5, size=7)

    x = np.concatenate([state, action])
    x = torch.tensor(x, dtype=torch.float32)

    delta = model(x).detach().numpy()
    state = state + delta

    traj.append(state.copy())

print("✅ rollout 完成，长度:", len(traj))
