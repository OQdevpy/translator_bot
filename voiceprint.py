"""Speaker embeddings (voiceprints) with a WeSpeaker ResNet34 ONNX model, no torch needed.

The model turns a few seconds of speech into a 256-dim vector; cosine similarity between
two vectors tells how likely two recordings are the same person. Features follow Kaldi's
80-bin log-mel filterbank, which is what the model was trained on.
"""
import io
import logging
import os
import wave

import numpy as np

log = logging.getLogger("stt_web")

MODEL_URL = (
    "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/main/voxceleb_resnet34_LM.onnx"
)
MODEL_SHA256 = "7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068"
SAMPLE_RATE = 16000
MIN_SECONDS = 1.0  # shorter clips give unreliable voiceprints


def read_wav(data: bytes) -> np.ndarray:
    """16-bit mono WAV -> float samples in int16 scale (what the model's features expect)."""
    with wave.open(io.BytesIO(data)) as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError("expected 16-bit mono WAV")
        rate = w.getframerate()
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)
    if rate != SAMPLE_RATE:
        positions = np.arange(0, len(samples) - 1, rate / SAMPLE_RATE)
        samples = np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)
    return samples


def _mel_banks(num_bins=80, n_fft=512, low=20.0, high=SAMPLE_RATE / 2) -> np.ndarray:
    mel = lambda f: 1127.0 * np.log(1.0 + f / 700.0)  # noqa: E731
    lo, hi = mel(low), mel(high)
    delta = (hi - lo) / (num_bins + 1)
    fft_mel = mel(np.arange(n_fft // 2) * SAMPLE_RATE / n_fft)
    banks = np.zeros((num_bins, n_fft // 2 + 1), dtype=np.float32)
    for m in range(num_bins):
        left, center, right = lo + m * delta, lo + (m + 1) * delta, lo + (m + 2) * delta
        up = (fft_mel - left) / (center - left)
        down = (right - fft_mel) / (right - center)
        banks[m, : n_fft // 2] = np.maximum(0.0, np.minimum(up, down))
    return banks


_BANKS = _mel_banks()
_WINDOW = ((0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 399)) ** 0.85).astype(np.float32)  # povey


def fbank(samples: np.ndarray) -> np.ndarray:
    """Kaldi-style 80-dim log mel filterbank (25 ms window, 10 ms shift), mean-normalised."""
    n = 1 + (len(samples) - 400) // 160
    if n < 1:
        raise ValueError("audio too short")
    idx = np.arange(400)[None, :] + 160 * np.arange(n)[:, None]
    frames = samples[idx].astype(np.float32)
    frames -= frames.mean(axis=1, keepdims=True)
    frames = np.concatenate([frames[:, :1] * (1 - 0.97), frames[:, 1:] - 0.97 * frames[:, :-1]], axis=1)
    frames *= _WINDOW
    power = np.abs(np.fft.rfft(frames, n=512)) ** 2
    feats = np.log(np.maximum(power @ _BANKS.T, np.finfo(np.float32).eps))
    return (feats - feats.mean(axis=0)).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class Voiceprints:
    def __init__(self, model_path: str):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])

    @classmethod
    def load(cls, data_dir: str) -> "Voiceprints | None":
        """Downloads the model into data_dir on first use; None (Gemini-only speaker matching) if unavailable."""
        path = os.getenv("VOICEPRINT_MODEL") or os.path.join(data_dir, "voxceleb_resnet34_LM.onnx")
        try:
            if not os.path.exists(path):
                download_model(path)
            return cls(path)
        except Exception as e:
            log.warning("Voiceprint model unavailable, speakers are matched by Gemini only: %s", e)
            return None

    def embed(self, wav: bytes) -> np.ndarray | None:
        samples = read_wav(wav)
        if len(samples) < MIN_SECONDS * SAMPLE_RATE:
            return None
        (emb,) = self.session.run(["embs"], {"feats": fbank(samples)[None]})
        return emb[0]


def download_model(path: str):
    import hashlib

    import requests

    log.info("Downloading voiceprint model to %s", path)
    resp = requests.get(MODEL_URL, timeout=300)
    resp.raise_for_status()
    if hashlib.sha256(resp.content).hexdigest() != MODEL_SHA256:
        raise ValueError("voiceprint model checksum mismatch")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(resp.content)
    os.replace(tmp, path)
