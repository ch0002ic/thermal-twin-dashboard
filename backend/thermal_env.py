import numpy as np
import gym
from gym import spaces
from thermal_model import ThermalModel
import torch
from pinn_surrogate import PINNSurrogate

class ThermalControlEnv(gym.Env):
    """
    Gym environment for 1D heat conduction system
    The agent controls airflow_rate to keep max temperature below a threshold
    State: [power_load, airflow_rate, ambient_temp, tim_conductivity]
    Action: change in airflow_rate (continuous)
    Reward: -max_temperature (or penalize if above threshold)
    """
    def __init__(self, nx=50, temp_threshold=60.0, backend='physics', pinn_model_path=None, device=None):
        super().__init__()
        self.nx = nx
        self.temp_threshold = temp_threshold
        self.backend = backend
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if backend == 'pinn':
            self.pinn = PINNSurrogate().to(self.device)
            assert pinn_model_path is not None, "pinn_model_path must be provided for backend='pinn'"
            self.pinn.load_state_dict(torch.load(pinn_model_path, map_location=self.device))
            self.pinn.eval()
        else:
            self.pinn = None
        # Observation: [power_load, airflow_rate, ambient_temp, tim_conductivity]
        self.observation_space = spaces.Box(
            low=np.array([0, 0.01, 0, 0.1], dtype=np.float32),
            high=np.array([1000, 10, 50, 10], dtype=np.float32),
            dtype=np.float32
        )
        # Action: delta airflow_rate (continuous, -1 to 1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.reset()

    def reset(self):
        # Randomize initial state
        self.state = np.array([
            np.random.uniform(0, 1000),
            np.random.uniform(0.01, 10),
            np.random.uniform(0, 50),
            np.random.uniform(0.1, 10)
        ], dtype=np.float32)
        self.steps = 0
        return self.state.copy()

    def step(self, action):
        # Clip and apply action to airflow_rate
        delta = float(np.clip(action[0], -1, 1))
        self.state[1] = np.clip(self.state[1] + delta, 0.01, 10)
        # Simulate physics or PINN
        if self.backend == 'physics':
            model = ThermalModel(nx=self.nx)
            T = model.solve_heat_equation(*self.state)
        elif self.backend == 'pinn':
            # Use PINN surrogate
            params = torch.tensor(self.state, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, 4)
            x = torch.linspace(0, 0.1, self.nx, device=self.device).unsqueeze(0)  # (1, nx)
            params_expanded = params.unsqueeze(1).expand(-1, self.nx, -1)
            x_ = x.unsqueeze(-1)
            inp = torch.cat([params_expanded, x_], dim=-1)  # (1, nx, 5)
            with torch.no_grad():
                T = self.pinn(inp).cpu().numpy().squeeze(0).squeeze(-1)  # (nx,)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        max_temp = np.max(T)
        # Reward: negative max temp, penalize if above threshold
        reward = -max_temp
        done = max_temp > self.temp_threshold or self.steps >= 49
        info = {'max_temp': max_temp, 'profile': T}
        self.steps += 1
        return self.state.copy(), reward, done, info

    def render(self, mode='human'):
        print(f"State: {self.state}, Steps: {self.steps}")

# Example agent class (replace with a real DRL agent, e.g., from Stable Baselines3)
class DummyAgent:
    def __init__(self, action_space):
        self.action_space = action_space
    def predict(self, obs):
        # Replace this with your agent's policy
        return self.action_space.sample(), None

# Example usage
if __name__ == "__main__":
    # To use PINN, set backend='pinn' and provide the path to the trained model
    # env = ThermalControlEnv(backend='pinn', pinn_model_path='pinn_model.pt')
    env = ThermalControlEnv()
    agent = DummyAgent(env.action_space)  # Replace with your trained agent
    n_episodes = 5
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        max_temps = []
        steps = 0
        while not done:
            action, _ = agent.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            max_temps.append(info['max_temp'])
            steps += 1
        print(f"Episode {ep+1} finished. Steps: {steps}, Total reward: {total_reward:.2f}, Max temp in episode: {np.max(max_temps):.2f}")
    print("\nYou can now use this environment with a DRL agent (e.g., Stable Baselines3). Replace DummyAgent with your agent and use agent.predict(obs).\n")
