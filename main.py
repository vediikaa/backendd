from fastapi import FastAPI
from pydantic import BaseModel
from stable_baselines3 import PPO
import os

# RL training imports
from stable_baselines3.common.env_util import make_vec_env
from section_env import SectionEnv

app = FastAPI()

MODEL_PATH = "models/ppo_section.zip"
model = None

# Try loading model at startup
if os.path.exists(MODEL_PATH):
    try:
        model = PPO.load(MODEL_PATH)
        print(f"✅ Loaded PPO model from {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Failed to load model: {e}")


class RequestData(BaseModel):
    trains: list


@app.post("/predict")
def predict(data: RequestData):
    global model

    if model is None:
        return {"error": "No trained model found. Please train first via /train"}

    # Convert input to dummy observation (toy env expects 6 values)
    obs = [0, 0, 1, 0, 0.5, 0.5]
    action, _ = model.predict(obs, deterministic=True)

    return {"recommended_action": int(action)}


@app.post("/train")
def train_model():
    global model
    print("🚂 Training started...")

    env = make_vec_env(lambda: SectionEnv(), n_envs=4)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=20000)

    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

    return {"status": "training complete", "model_path": MODEL_PATH}
