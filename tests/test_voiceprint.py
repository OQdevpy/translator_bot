import hashlib
import io
import logging
import os
import wave

import numpy as np
import pytest
import requests

import voiceprint
from voiceprint import MIN_SECONDS, SAMPLE_RATE, Voiceprints, cosine, fbank, read_wav

REAL_MODEL = os.getenv("VOICEPRINT_TEST_MODEL") or (
    "/tmp/claude-0/-home-user-translator-bot/faaf8ec8-48cb-5420-8ded-c14a1679bb22/scratchpad/models/voxceleb_resnet34_LM.onnx"
)
FFT_BIN_HZ = SAMPLE_RATE / 512


def wav_bytes(samples, rate=SAMPLE_RATE, channels=1, width=2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        if width == 2:
            w.writeframes(np.asarray(samples, dtype="<i2").tobytes())
        else:
            w.writeframes(bytes(np.asarray(samples, dtype=np.uint8)))
    return buf.getvalue()


def noise(seconds, level=300.0, seed=0, rate=SAMPLE_RATE):
    return np.random.default_rng(seed).standard_normal(int(seconds * rate)) * level


def tone(freq, seconds, level=8000.0, rate=SAMPLE_RATE):
    return np.sin(2 * np.pi * freq * np.arange(int(seconds * rate)) / rate) * level


class FakeSession:
    """Stands in for onnxruntime.InferenceSession: records the features, returns a fixed embedding."""

    def __init__(self, dim=256):
        self.calls = []
        self.dim = dim

    def run(self, outputs, feeds):
        self.calls.append((outputs, feeds))
        batch = feeds["feats"].shape[0]
        return [np.tile(np.arange(self.dim, dtype=np.float32), (batch, 1))]


def fake_voiceprints(session=None) -> Voiceprints:
    vp = object.__new__(Voiceprints)  # skip loading a real ONNX model
    vp.session = session or FakeSession()
    return vp


# ---------------------------------------------------------------- read_wav


def test_read_wav_keeps_int16_scale():
    samples = [0, 1, -1, 1234, 32767, -32768]
    out = read_wav(wav_bytes(samples))
    assert out.dtype == np.float32
    assert out.tolist() == [float(s) for s in samples]


def test_read_wav_empty_file_gives_no_samples():
    assert len(read_wav(wav_bytes([]))) == 0
    assert len(read_wav(wav_bytes([], rate=8000))) == 0


def test_read_wav_upsamples_8k_by_linear_interpolation():
    samples = np.array([0, 100, -100, 300, 1000], dtype=np.int16)
    out = read_wav(wav_bytes(samples, rate=8000))
    assert out.dtype == np.float32
    # every original sample is kept and a midpoint is inserted after it (the very last one has no successor)
    assert out[::2].tolist() == samples[:-1].astype(float).tolist()
    assert out[1::2].tolist() == [50.0, 0.0, 100.0, 650.0]
    assert len(out) == 2 * (len(samples) - 1)


def test_read_wav_resamples_one_second_to_16k_rate():
    assert len(read_wav(wav_bytes(np.zeros(48000), rate=48000))) == SAMPLE_RATE
    assert len(read_wav(wav_bytes(np.zeros(44100), rate=44100))) == SAMPLE_RATE
    # 8 kHz upsampling loses only the final interpolation step
    assert len(read_wav(wav_bytes(np.zeros(8000), rate=8000))) == SAMPLE_RATE - 2


def test_read_wav_downsamples_48k_by_taking_every_third_sample():
    samples = (np.arange(4800) % 2000 - 1000).astype(np.int16)
    out = read_wav(wav_bytes(samples, rate=48000))
    assert np.array_equal(out, samples[::3].astype(np.float32))


def test_resampled_tone_keeps_its_frequency():
    out = read_wav(wav_bytes(tone(440, 1.0, rate=48000), rate=48000))
    spectrum = np.abs(np.fft.rfft(out))
    peak_hz = spectrum.argmax() * SAMPLE_RATE / len(out)
    assert peak_hz == pytest.approx(440, abs=2)


@pytest.mark.parametrize(
    "kwargs",
    [dict(channels=2), dict(width=1)],
    ids=["stereo", "8-bit"],
)
def test_read_wav_rejects_other_formats(kwargs):
    data = wav_bytes(np.zeros(1600, dtype=np.uint8 if kwargs.get("width") == 1 else np.int16), **kwargs)
    with pytest.raises(ValueError, match="16-bit mono"):
        read_wav(data)


def test_read_wav_rejects_garbage():
    with pytest.raises(wave.Error):
        read_wav(b"definitely not a wav file at all")


# ---------------------------------------------------------------- fbank


@pytest.mark.parametrize("length,frames", [(400, 1), (559, 1), (560, 2), (16000, 98), (32000, 198)])
def test_fbank_frame_count_follows_kaldi_snip_edges(length, frames):
    feats = fbank(noise(length / SAMPLE_RATE))
    assert feats.shape == (frames, 80)
    assert feats.dtype == np.float32
    assert frames == 1 + (length - 400) // 160


@pytest.mark.parametrize("length", [0, 1, 160, 399])
def test_fbank_rejects_audio_shorter_than_one_window(length):
    with pytest.raises(ValueError, match="too short"):
        fbank(np.zeros(length, dtype=np.float32))


def test_fbank_is_mean_normalised_per_dimension():
    signal = np.concatenate([noise(0.7), noise(0.7, seed=1) + tone(1000, 0.7)])
    feats = fbank(signal)
    assert np.all(np.isfinite(feats))
    assert np.abs(feats.mean(axis=0)).max() < 1e-4
    assert feats.std() > 0.1  # normalisation removes the mean, not the content


def test_fbank_of_silence_is_finite():
    feats = fbank(np.zeros(SAMPLE_RATE, dtype=np.float32))
    assert np.all(np.isfinite(feats))
    assert np.allclose(feats, 0.0, atol=1e-5)


def test_mel_banks_cover_the_spectrum_in_increasing_order():
    banks = voiceprint._BANKS
    assert banks.shape == (80, 257)
    assert np.all(banks >= 0) and banks.max() <= 1.0 + 1e-6
    assert np.all(banks[:, -1] == 0)  # Kaldi leaves the Nyquist bin out
    peaks = banks.argmax(axis=1) * FFT_BIN_HZ
    assert np.all(np.diff(peaks) >= 0)
    assert 20 < peaks[0] < 100 and 7000 < peaks[-1] < 8000


def tone_rise(freq):
    """Features are mean-normalised over time, so compare noise alone with noise plus the tone."""
    signal = np.concatenate([noise(1.0, level=1000), noise(1.0, level=1000, seed=1) + tone(freq, 1.0)])
    feats = fbank(signal)
    half = feats.shape[0] // 2
    return feats[half + 3 :].mean(axis=0) - feats[: half - 3].mean(axis=0)


@pytest.mark.parametrize("freq", [1000.0, 2000.0, 3000.0, 6000.0])
def test_a_pure_tone_raises_energy_in_the_matching_mel_bin(freq):
    expected_bank = int(voiceprint._BANKS[:, int(round(freq / FFT_BIN_HZ))].argmax())
    rise = tone_rise(freq)
    assert abs(int(rise.argmax()) - expected_bank) <= 1
    assert rise[expected_bank] > 0.9 * rise.max()
    # bins far from the tone barely move
    far = [b for b in range(80) if abs(b - expected_bank) > 10]
    assert np.abs(rise[far]).max() < 0.1 * rise[expected_bank]


def test_higher_tones_land_in_higher_mel_bins():
    peaks = [int(tone_rise(f).argmax()) for f in (300.0, 1000.0, 2500.0, 5000.0, 7000.0)]
    assert peaks == sorted(peaks) and len(set(peaks)) == len(peaks)


def kaldi_fbank_reference(samples):
    """A plain per-frame transcription of Kaldi's compute-fbank-feats defaults used by WeSpeaker
    (25 ms povey window, 10 ms shift, snip_edges, no dither, preemphasis 0.97, 512-point FFT,
    80 mel bins from 20 Hz to Nyquist, natural log) followed by per-utterance mean subtraction."""
    mel = lambda f: 1127.0 * np.log(1.0 + f / 700.0)  # noqa: E731
    lo, hi = mel(20.0), mel(8000.0)
    delta = (hi - lo) / 81
    banks = np.zeros((80, 257))
    for m in range(80):
        left, center, right = lo + m * delta, lo + (m + 1) * delta, lo + (m + 2) * delta
        for k in range(256):
            f = mel(k * 16000 / 512)
            if left < f < right:
                banks[m, k] = (f - left) / (center - left) if f <= center else (right - f) / (right - center)
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 399)) ** 0.85
    rows = []
    for i in range(1 + (len(samples) - 400) // 160):
        frame = np.array(samples[i * 160 : i * 160 + 400], dtype=np.float64)
        frame = frame - frame.mean()
        frame = np.append(frame[0] - 0.97 * frame[0], frame[1:] - 0.97 * frame[:-1])
        spectrum = np.abs(np.fft.rfft(frame * window, 512)) ** 2
        rows.append(np.log(np.maximum(banks @ spectrum, np.finfo(np.float32).eps)))
    feats = np.array(rows)
    return feats - feats.mean(axis=0)


def test_fbank_matches_a_straightforward_kaldi_implementation():
    signal = noise(0.6, level=2000, seed=3) + tone(700, 0.6, level=3000) + 500  # with a DC offset
    np.testing.assert_allclose(fbank(signal), kaldi_fbank_reference(signal), atol=2e-3)


def test_fbank_does_not_modify_its_input():
    signal = noise(0.5).astype(np.float32)
    copy = signal.copy()
    fbank(signal)
    assert np.array_equal(signal, copy)


# ---------------------------------------------------------------- cosine


def test_cosine():
    a = np.array([1.0, 2.0, 3.0])
    assert cosine(a, a) == pytest.approx(1.0)
    assert cosine(a, 10 * a) == pytest.approx(1.0)  # scale does not matter
    assert cosine(a, -a) == pytest.approx(-1.0)
    assert cosine(np.array([1.0, 0.0]), np.array([0.0, 5.0])) == pytest.approx(0.0)
    assert cosine(np.array([1.0, 1.0]), np.array([1.0, 0.0])) == pytest.approx(1 / np.sqrt(2))
    assert isinstance(cosine(a, a), float)


def test_cosine_with_a_zero_vector_is_zero_not_nan():
    assert cosine(np.zeros(4), np.ones(4)) == 0.0
    assert cosine(np.zeros(4), np.zeros(4)) == 0.0


# ---------------------------------------------------------------- Voiceprints.embed


def test_embed_returns_none_for_clips_shorter_than_min_seconds():
    vp = fake_voiceprints()
    short = noise(MIN_SECONDS - 0.01)
    assert vp.embed(wav_bytes(short)) is None
    assert vp.embed(wav_bytes([])) is None
    assert vp.session.calls == []  # the model is not even run


def test_embed_runs_the_model_on_fbank_features():
    vp = fake_voiceprints()
    samples = noise(MIN_SECONDS)  # exactly the minimum is enough
    emb = vp.embed(wav_bytes(samples))
    assert emb.shape == (256,)
    assert emb.tolist() == list(range(256))
    ((outputs, feeds),) = vp.session.calls
    assert outputs == ["embs"]
    feats = feeds["feats"]
    assert feats.dtype == np.float32
    assert feats.shape == (1, 1 + (len(samples) - 400) // 160, 80)
    expected = fbank(read_wav(wav_bytes(samples)))
    assert np.array_equal(feats[0], expected)


def test_embed_resamples_before_measuring_length():
    vp = fake_voiceprints()
    assert vp.embed(wav_bytes(noise(1.5, rate=8000), rate=8000)) is not None
    feats = vp.session.calls[0][1]["feats"]
    assert feats.shape == (1, 1 + (2 * (12000 - 1) - 400) // 160, 80)


def test_embed_rejects_stereo():
    vp = fake_voiceprints()
    with pytest.raises(ValueError):
        vp.embed(wav_bytes(np.zeros(2 * SAMPLE_RATE), channels=2))
    assert vp.session.calls == []


# ---------------------------------------------------------------- Voiceprints.load / download_model


def test_load_returns_none_when_download_fails(monkeypatch, tmp_path, caplog):
    monkeypatch.delenv("VOICEPRINT_MODEL", raising=False)
    calls = []

    def failing_download(path):
        calls.append(path)
        raise requests.ConnectionError("no network")

    monkeypatch.setattr(voiceprint, "download_model", failing_download)
    with caplog.at_level(logging.WARNING, logger="stt_web"):
        assert Voiceprints.load(str(tmp_path)) is None
    assert calls == [str(tmp_path / "voxceleb_resnet34_LM.onnx")]
    assert "no network" in caplog.text


def test_load_uses_an_existing_model_without_downloading(monkeypatch, tmp_path):
    monkeypatch.delenv("VOICEPRINT_MODEL", raising=False)
    model = tmp_path / "voxceleb_resnet34_LM.onnx"
    model.write_bytes(b"model")
    monkeypatch.setattr(voiceprint, "download_model", lambda path: pytest.fail("must not download"))
    opened = []
    monkeypatch.setattr(Voiceprints, "__init__", lambda self, path: opened.append(path))
    assert isinstance(Voiceprints.load(str(tmp_path)), Voiceprints)
    assert opened == [str(model)]


def test_load_downloads_the_model_once_when_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("VOICEPRINT_MODEL", raising=False)
    downloads, opened = [], []

    def download(path):
        downloads.append(path)
        with open(path, "wb") as f:
            f.write(b"model")

    monkeypatch.setattr(voiceprint, "download_model", download)
    monkeypatch.setattr(Voiceprints, "__init__", lambda self, path: opened.append(path))
    assert Voiceprints.load(str(tmp_path)) is not None
    assert Voiceprints.load(str(tmp_path)) is not None
    path = str(tmp_path / "voxceleb_resnet34_LM.onnx")
    assert downloads == [path]
    assert opened == [path, path]


def test_load_honours_voiceprint_model_env(monkeypatch, tmp_path):
    custom = tmp_path / "elsewhere" / "custom.onnx"
    monkeypatch.setenv("VOICEPRINT_MODEL", str(custom))
    downloads = []
    monkeypatch.setattr(voiceprint, "download_model", lambda path: downloads.append(path))
    monkeypatch.setattr(Voiceprints, "__init__", lambda self, path: None)
    Voiceprints.load(str(tmp_path / "data"))
    assert downloads == [str(custom)]


def test_load_returns_none_for_a_corrupt_model_file(monkeypatch, tmp_path):
    monkeypatch.delenv("VOICEPRINT_MODEL", raising=False)
    (tmp_path / "voxceleb_resnet34_LM.onnx").write_bytes(b"this is not an onnx model")
    monkeypatch.setattr(voiceprint, "download_model", lambda path: pytest.fail("must not download"))
    assert Voiceprints.load(str(tmp_path)) is None  # the real onnxruntime refuses it


class FakeResponse:
    def __init__(self, content=b"", status=200):
        self.content = content
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


def fake_get(monkeypatch, response):
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return response

    monkeypatch.setattr(requests, "get", get)
    return calls


def test_download_model_checksum_mismatch_leaves_nothing_behind(monkeypatch, tmp_path):
    calls = fake_get(monkeypatch, FakeResponse(b"tampered or truncated model"))
    path = tmp_path / "models" / "voxceleb_resnet34_LM.onnx"
    with pytest.raises(ValueError, match="checksum"):
        voiceprint.download_model(str(path))
    assert calls == [(voiceprint.MODEL_URL, {"timeout": 300})]
    assert not path.exists()
    assert not (tmp_path / "models" / "voxceleb_resnet34_LM.onnx.part").exists()


def test_download_model_http_error_leaves_nothing_behind(monkeypatch, tmp_path):
    fake_get(monkeypatch, FakeResponse(b"<html>404</html>", status=404))
    path = tmp_path / "voxceleb_resnet34_LM.onnx"
    with pytest.raises(requests.HTTPError):
        voiceprint.download_model(str(path))
    assert list(tmp_path.iterdir()) == []


def test_download_model_writes_verified_file_atomically(monkeypatch, tmp_path):
    content = b"pretend onnx bytes" * 100
    monkeypatch.setattr(voiceprint, "MODEL_SHA256", hashlib.sha256(content).hexdigest())
    fake_get(monkeypatch, FakeResponse(content))
    path = tmp_path / "a" / "b" / "model.onnx"
    voiceprint.download_model(str(path))
    assert path.read_bytes() == content
    assert sorted(p.name for p in path.parent.iterdir()) == ["model.onnx"]  # no .part left


def test_load_after_checksum_mismatch_returns_none(monkeypatch, tmp_path):
    monkeypatch.delenv("VOICEPRINT_MODEL", raising=False)
    fake_get(monkeypatch, FakeResponse(b"wrong bytes"))
    assert Voiceprints.load(str(tmp_path)) is None
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------- the real model (optional)


@pytest.fixture(scope="module")
def real_voiceprints():
    if not os.path.exists(REAL_MODEL):
        pytest.skip(f"voiceprint model not found at {REAL_MODEL}")
    return Voiceprints(REAL_MODEL)


def test_real_model_returns_a_256_dim_finite_embedding(real_voiceprints):
    signal = noise(2.0, level=2000) + tone(220, 2.0, level=4000) + tone(660, 2.0, level=1500)
    emb = real_voiceprints.embed(wav_bytes(np.clip(signal, -32768, 32767)))
    assert emb.shape == (256,)
    assert np.all(np.isfinite(emb))
    assert np.linalg.norm(emb) > 0
    again = real_voiceprints.embed(wav_bytes(np.clip(signal, -32768, 32767)))
    assert cosine(emb, again) == pytest.approx(1.0, abs=1e-5)  # deterministic
    assert real_voiceprints.embed(wav_bytes(noise(0.5))) is None
