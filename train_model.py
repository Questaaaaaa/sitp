import pickle
import numpy as np
import torch
import torch.nn as nn

# =========================
# 📦 1. 读取数据
# =========================
with open("panda_data.pkl", "rb") as f:
    data = pickle.load(f)

X, Y = [], []

for state, action, next_state in data:
    X.append(np.concatenate([state, action]))
    Y.append(next_state - state)   # 🔥 delta

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

print("X:", X.shape)
print("Y:", Y.shape)


# =========================
# 🧠 2. 定义模型
# =========================
class WorldModel(nn.Module):
    def __init__(self, input_dim, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)
        )

    def forward(self, x):
        return self.net(x)

model = WorldModel(X.shape[1], Y.shape[1])


# =========================
# ⚙️ 3. 训练
# =========================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(30):
    pred = model(X)
    loss = loss_fn(pred, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.6f}")


# =========================
# 💾 4. 保存模型
# =========================
torch.save(model.state_dict(), "world_model.pt")
print("✅ 模型已保存")
