"""
Model loading and the single-episode runner shared by policies and the harness.
"""

import io
import os
import tempfile
import zipfile

from sb3_contrib import RecurrentPPO
import numpy as np
import torch


def load_model(path: str, device: str = 'cpu', env=None):
    """
    Load a model. Supports both RecurrentPPO (with _CPULSTMWrapper patch)
    and standard PPO.

    device: defaults to 'cpu' (the original behavior, for eval call sites).
    train.py's --resume-from path passes the real training device (e.g.
    'mps') so a resumed run continues on the same device the interrupted
    run used, rather than being silently forced onto CPU.

    env: if given, attached at load time via RecurrentPPO.load(path, env=env)
    rather than a separate model.set_env(env) call afterward. This matters
    when resuming with a DIFFERENT n_envs than the checkpoint was trained
    with (e.g. reducing parallelism after a memory incident) -- SB3's
    set_env() hard-asserts the new env's num_envs matches the checkpoint's
    exactly and refuses to resize the rollout buffers; passing env directly
    to .load() is the path SB3 itself points to for changing n_envs on
    resume.
    """
    zip_path = path if path.endswith('.zip') else path + '.zip'

    # Check if this is a recurrent model (contains lstm layers in policy state dict)
    is_recurrent = False
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            if 'policy.pth' in zf.namelist():
                sd = torch.load(io.BytesIO(zf.read('policy.pth')), map_location='cpu')
                if any('lstm' in k for k in sd.keys()):
                    is_recurrent = True
    except Exception:
        pass

    if is_recurrent:
        all_files = {}
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                all_files[name] = zf.read(name)

        # Target policy.pth explicitly — not pytorch_variables.pth
        sd = torch.load(
            io.BytesIO(all_files['policy.pth']), map_location='cpu')

        new_sd = {}
        for k, v in sd.items():
            nk = (k
                  .replace('lstm_actor.lstm.', 'lstm_actor.')
                  .replace('lstm_critic.lstm.', 'lstm_critic.'))
            new_sd[nk] = v

        buf = io.BytesIO()
        torch.save(new_sd, buf)
        all_files['policy.pth'] = buf.getvalue()

        with tempfile.TemporaryDirectory() as tmpdir:
            patched = os.path.join(tmpdir, 'model.zip')
            with zipfile.ZipFile(patched, 'w', zipfile.ZIP_DEFLATED) as zf_out:
                for name, data in all_files.items():
                    zf_out.writestr(name, data)
            return RecurrentPPO.load(patched, env=env, device=device)
    else:
        from stable_baselines3 import PPO
        return PPO.load(zip_path, env=env, device=device)


def _run_episode(model, env, seed: int):
    """
    Run one episode with correct RecurrentPPO LSTM state tracking.
    Returns (is_pct, episode_reward, forced_liquidation, actions_taken).
    """
    obs, info = env.reset(seed=seed)
    arrival_price = info.get("arrival_price", 100.0)

    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)

    done = False
    ep_reward = 0.0
    fill_qtys = []
    fill_prices = []
    actions_taken = []

    while not done:
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_start,
            deterministic=True,
        )
        episode_start = np.zeros((1,), dtype=bool)

        actions_taken.append(int(action))
        obs, reward, done, _, info = env.step(int(action))
        ep_reward += reward

        if info.get("fill_qty", 0) > 0:
            fill_qtys.append(info["fill_qty"])
            fill_prices.append(info["fill_price"])

    forced = info.get("forced_liquidation_qty", 0) > 0

    is_pct = info.get("total_IS_pct")
    if is_pct is not None:
        is_pct = -is_pct
    elif fill_qtys:
        total_qty = sum(fill_qtys)
        vwap = sum(q * p for q, p in zip(fill_qtys, fill_prices)) / total_qty
        is_pct = (vwap - arrival_price) / arrival_price * 100

    return is_pct, ep_reward, forced, actions_taken
