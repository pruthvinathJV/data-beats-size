"""VLM-as-judge scoring served by vLLM.

The judge emits one token. Instead of trusting that argmax digit, we read the
top-k logprobs and take the probability-weighted score over "1".."5" (G-Eval
style). This gives a continuous score that ranks far better than a 5-level label.
"""

import base64
import hashlib
import io
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

DIGITS = ("1", "2", "3", "4", "5")
MIN_DIGIT_MASS = 0.5  # below this, the judge did not really answer with a score


def load_rubric(path: str | Path) -> tuple[str, str]:
    """Rubric text and the SHA-256 of the file, matching results/handrate_manifest.json."""
    raw = Path(path).read_bytes()
    return raw.decode().strip(), hashlib.sha256(raw).hexdigest()


def resize_max_side(img: Image.Image, max_side: int | None) -> Image.Image:
    img = img.convert("RGB")
    if max_side and max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side), Image.Resampling.BICUBIC)
    return img


def to_data_url(img: Image.Image, max_side: int | None) -> str:
    buf = io.BytesIO()
    resize_max_side(img, max_side).save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build_conversation(rubric: str, image_url: str, caption: str) -> list[dict]:
    # Rubric first so vLLM's prefix cache shares it across every request.
    return [
        {"role": "system", "content": rubric},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": f"Caption: {caption}\nScore:"},
            ],
        },
    ]


def digit_probs(top: list[tuple[str | None, float]]) -> np.ndarray:
    """Probability mass on each digit from (decoded_token, logprob) pairs."""
    p = np.zeros(len(DIGITS))
    for tok, lp in top:
        t = (tok or "").strip()
        if t in DIGITS:
            p[int(t) - 1] += math.exp(lp)
    return p


@dataclass(frozen=True)
class JudgeScore:
    expected: float  # probability-weighted score in [1, 5]
    argmax: int  # most likely digit, 0 if none
    digit_mass: float  # total probability on "1".."5"
    probs: tuple[float, ...]  # renormalized over the digits

    @property
    def parse_ok(self) -> bool:
        return self.digit_mass >= MIN_DIGIT_MASS


def summarize(p: np.ndarray) -> JudgeScore:
    mass = float(p.sum())
    if mass == 0:
        return JudgeScore(float("nan"), 0, 0.0, (0.0,) * len(DIGITS))
    q = p / mass
    return JudgeScore(
        expected=float(q @ np.arange(1, len(DIGITS) + 1)),
        argmax=int(q.argmax()) + 1,
        digit_mass=mass,
        probs=tuple(float(x) for x in q),
    )


class VLMJudge:
    def __init__(
        self,
        model: str,
        rubric: str | Path,
        max_image_side: int | None = 512,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.90,
        top_logprobs: int = 20,
        seed: int = 0,
    ):
        from vllm import LLM, SamplingParams

        self.model = model
        self.rubric, self.rubric_sha = load_rubric(rubric)
        self.max_image_side = max_image_side
        self.llm = LLM(
            model=model,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"image": 1},
            enable_prefix_caching=True,
            seed=seed,
        )
        self._score_params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=top_logprobs)
        self._SamplingParams = SamplingParams

    def digit_token_ids(self) -> dict[str, list[int]]:
        tok = self.llm.get_tokenizer()
        return {d: tok.encode(d, add_special_tokens=False) for d in DIGITS}

    def _conversations(self, images, captions, max_side):
        return [
            build_conversation(self.rubric, to_data_url(img, max_side), cap)
            for img, cap in zip(images, captions, strict=True)
        ]

    def score(self, images, captions, max_side: int | None = -1) -> list[JudgeScore]:
        """max_side=-1 uses the configured resize; None sends full resolution."""
        side = self.max_image_side if max_side == -1 else max_side
        outs = self.llm.chat(
            self._conversations(images, captions, side), self._score_params, use_tqdm=False
        )
        return [
            summarize(
                digit_probs([(lp.decoded_token, lp.logprob) for lp in o.outputs[0].logprobs[0].values()])
            )
            for o in outs
        ]

    def generate_text(self, images, captions, max_side: int | None, max_tokens: int = 8) -> list[str]:
        """Plain decoding, the unoptimized baseline for the throughput table."""
        params = self._SamplingParams(max_tokens=max_tokens, temperature=0.0)
        outs = self.llm.chat(self._conversations(images, captions, max_side), params, use_tqdm=False)
        return [o.outputs[0].text for o in outs]
