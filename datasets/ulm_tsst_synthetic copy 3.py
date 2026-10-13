# datasets/ulm_tsst_synthetic.py
"""
Synthetic ULM‑TSST‑Style RAW Multimodal Time‑Series Dataset
-----------------------------------------------------------
"""

import os
import torch
import numpy as np

# IMPORTANT: absolute import (no relative import)
from datasets.base_dataset import BaseMultimodalDataset


class SyntheticULMTSSTDataset(BaseMultimodalDataset):
    """
    Synthetic RAW multimodal dataset aligned with ULM‑TSST.
    """

    def __init__(
        self,
        num_samples: int,
        target_seq_len: int,
        modalities: dict,
        labels: dict,
        device: str = "cpu",
        num_subjects: int = 10,
        num_sessions: int = 3,
        save_dir: str | None = None,
    ):
        super().__init__(T=target_seq_len, S=0)

        self.num_samples = num_samples
        self.T = target_seq_len
        self.modalities = modalities
        self.labels = labels
        self.device = torch.device(device)

        self.num_subjects = num_subjects
        self.num_sessions = num_sessions

        self.save_dir = save_dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        self.rng = np.random.default_rng(seed=42)

        # subject-specific parameters
        self.subject_baseline_valence = torch.randn(self.num_subjects, device=self.device) * 0.2
        self.subject_stress_gain = 0.8 + torch.rand(self.num_subjects, device=self.device) * 0.4
        self.subject_noise_scale = 0.5 + torch.rand(self.num_subjects, device=self.device) * 0.5

    # -------------------------------------------------------------
    # Ornstein–Uhlenbeck VA trajectories
    # -------------------------------------------------------------
    def _ou_process(self, theta=0.4, mu=0.0, sigma=0.25):
        dt = 1.0 / self.T
        x = np.zeros(self.T)
        for t in range(1, self.T):
            dx = theta * (mu - x[t - 1]) * dt + sigma * np.sqrt(dt) * self.rng.normal()
            x[t] = x[t - 1] + dx
        return torch.tensor(x, device=self.device).float()

    def _va_trajectory(self, subject_id: int):
        v = self._ou_process(theta=0.4, mu=0.0, sigma=0.25)
        a = self._ou_process(theta=0.5, mu=0.3, sigma=0.3)

        # subject baseline valence
        v = v + self.subject_baseline_valence[subject_id]

        # emotion events
        num_events = self.rng.integers(2, 5)
        t_axis = torch.linspace(0, 1, self.T, device=self.device)

        for _ in range(num_events):
            center = float(self.rng.uniform(0.2, 0.8))
            width = float(self.rng.uniform(0.05, 0.15))
            amp_v = float(self.rng.uniform(-0.7, 0.7))
            amp_a = float(self.rng.uniform(0.3, 1.0))

            gauss = torch.exp(-0.5 * ((t_axis - center) / width) ** 2)
            v = v + amp_v * gauss
            a = a + amp_a * gauss

        v = torch.clamp(v, -1.0, 1.0)
        a = torch.clamp(a, -1.0, 1.0)

        return v, a

    # -------------------------------------------------------------
    # Piecewise TSST stress curve
    # -------------------------------------------------------------
    def _stress_curve(self, arousal: torch.Tensor, subject_id: int):
        t = torch.linspace(0, 1, self.T, device=self.device)

        stress = torch.zeros_like(t)

        stress = torch.where(t < 0.2, 0.2, stress)
        stress = torch.where((t >= 0.2) & (t < 0.4), 0.4, stress)
        stress = torch.where((t >= 0.4) & (t < 0.6), 0.7, stress)
        stress = torch.where((t >= 0.6) & (t < 0.8), 0.9, stress)
        stress = torch.where(t >= 0.8, 0.3, stress)

        ar_norm = (arousal - arousal.min()) / (arousal.max() - arousal.min() + 1e-6)
        stress = 0.5 * stress + 0.5 * ar_norm

        stress = stress * self.subject_stress_gain[subject_id]

        noise = torch.tensor(self.rng.normal(0, 0.1, self.T), device=self.device).float()
        stress = stress + noise

        return torch.clamp(stress, 0.0, 2.0)

    # -------------------------------------------------------------
    # RAW modality generators
    # -------------------------------------------------------------
    def _ecg(self, stress: torch.Tensor):
        t_axis = torch.linspace(0, 1, self.T, device=self.device)
        hr = 60 + 40 * stress
        ecg = torch.sin(2 * np.pi * hr * t_axis)
        ecg += torch.randn(self.T, device=self.device) * 0.05
        return ecg

    def _eda(self, stress: torch.Tensor):
        tonic = 0.2 + 0.5 * stress
        phasic = torch.randn(self.T, device=self.device) * 0.05
        return tonic + phasic

    def _resp(self, stress: torch.Tensor):
        freq = 0.2 + 0.3 * stress
        t_axis = torch.linspace(0, 1, self.T, device=self.device)
        resp = torch.sin(2 * np.pi * freq * t_axis)
        resp += torch.randn(self.T, device=self.device) * 0.02
        return resp

    def _bpm(self, stress: torch.Tensor):
        return 60 + 40 * stress + torch.randn(self.T, device=self.device) * 2

    def _eeg(self, stress: torch.Tensor):
        t_axis = torch.linspace(0, 1, self.T, device=self.device)
        alpha = torch.sin(2 * np.pi * 10 * t_axis)
        beta = torch.sin(2 * np.pi * 20 * t_axis)
        return alpha * (1 - stress) + beta * stress + torch.randn(self.T, device=self.device) * 0.1

    def _emg(self, stress: torch.Tensor):
        return stress + torch.randn(self.T, device=self.device) * 0.1

    def _eye(self, stress: torch.Tensor):
        return torch.randn(self.T, device=self.device) * (0.1 + 0.3 * stress)

    def _audio(self, stress: torch.Tensor, arousal: torch.Tensor):
        return torch.randn(self.T, device=self.device) * (0.2 + 0.5 * arousal + 0.3 * stress)

    def _video(self, stress: torch.Tensor, valence: torch.Tensor):
        return torch.randn(self.T, device=self.device) * (0.1 + 0.4 * stress + 0.3 * (1 - valence))

    def _text(self, stress: torch.Tensor, valence: torch.Tensor):
        return torch.randn(self.T, device=self.device) * (0.1 + 0.3 * stress + 0.2 * (1 - valence))

    # -------------------------------------------------------------
    # Labels
    # -------------------------------------------------------------
    def _labels(self, v: torch.Tensor, a: torch.Tensor, s: torch.Tensor):
        labels = {}

        if self.labels.get("valence", False):
            labels["valence"] = v.unsqueeze(1)
        if self.labels.get("arousal", False):
            labels["arousal"] = a.unsqueeze(1)
        if self.labels.get("dominance", False):
            labels["dominance"] = (0.3 * s + 0.2 * a).unsqueeze(1)
        if self.labels.get("stress", False):
            labels["stress"] = (1.2 * s).unsqueeze(1)
        if self.labels.get("workload", False):
            labels["workload"] = (0.9 * s + 0.3 * a).unsqueeze(1)

        for name, enabled in self.labels.items():
            if not enabled:
                labels[name] = torch.zeros(self.T, 1, device=self.device)

        return labels

    # -------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------
    def _metadata(self, idx: int):
        clip_id = idx
        subject_id = idx % self.num_subjects
        session_id = (idx // self.num_subjects) % self.num_sessions
        return {
            "clip_id": clip_id,
            "subject_id": subject_id,
            "session_id": session_id,
        }

    # -------------------------------------------------------------
    # Save to .npz
    # -------------------------------------------------------------
    def _save_sample(self, sample: dict, meta: dict):
        if self.save_dir is None:
            return

        clip_id = meta["clip_id"]
        path = os.path.join(self.save_dir, f"clip_{clip_id:04d}.npz")

        np_dict = {}

        for k, v in sample.items():
            if k in ["labels", "meta", "masks"]:
                continue
            np_dict[f"mod_{k}"] = v.detach().cpu().numpy()

        for lbl, v in sample["labels"].items():
            np_dict[f"lbl_{lbl}"] = v.detach().cpu().numpy()

        for mk, mv in sample["masks"].items():
            np_dict[f"mask_{mk}"] = mv.detach().cpu().numpy()

        np_dict["clip_id"] = np.array(meta["clip_id"])
        np_dict["subject_id"] = np.array(meta["subject_id"])
        np_dict["session_id"] = np.array(meta["session_id"])

        np.savez(path, **np_dict)

    # -------------------------------------------------------------
    # Dataset length
    # -------------------------------------------------------------
    def __len__(self):
        return self.num_samples

    # -------------------------------------------------------------
    # Get item
    # -------------------------------------------------------------
    def __getitem__(self, idx):
        sample = {}

        meta = self._metadata(idx)
        subject_id = meta["subject_id"]

        v, a = self._va_trajectory(subject_id=subject_id)
        s = self._stress_curve(arousal=a, subject_id=subject_id)

        masks = {}
        for m, enabled in self.modalities.items():
            if enabled:
                if m == "ecg": sample[m] = self._ecg(s)
                elif m == "eda": sample[m] = self._eda(s)
                elif m == "resp": sample[m] = self._resp(s)
                elif m == "bpm": sample[m] = self._bpm(s)
                elif m == "eeg": sample[m] = self._eeg(s)
                elif m == "emg": sample[m] = self._emg(s)
                elif m == "eye": sample[m] = self._eye(s)
                elif m == "audio": sample[m] = self._audio(s, a)
                elif m == "video": sample[m] = self._video(s, v)
                elif m == "text": sample[m] = self._text(s, v)

                masks[m] = torch.ones(self.T, dtype=torch.bool, device=self.device)
            else:
                sample[m] = torch.zeros(self.T, device=self.device)
                masks[m] = torch.zeros(self.T, dtype=torch.bool, device=self.device)

        sample["labels"] = self._labels(v, a, s)
        sample["masks"] = masks
        sample["meta"] = meta

        self._save_sample(sample, meta)

        return sample
