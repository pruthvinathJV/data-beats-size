"""CLIP embeddings and CLIPScore (Hessel et al., 2021): 2.5 * max(cos(img, txt), 0)."""

import numpy as np

CLIPSCORE_W = 2.5


class CLIPScorer:
    def __init__(self, model_name: str, batch_size: int = 256, device: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = CLIPModel.from_pretrained(model_name, torch_dtype=self.dtype).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.batch_size = batch_size

    def _normed(self, feats) -> np.ndarray:
        if not isinstance(feats, self.torch.Tensor):  # newer transformers return a model output
            feats = feats.pooler_output
        feats = feats.float()
        return (feats / feats.norm(dim=-1, keepdim=True)).cpu().numpy()

    def embed_images(self, images) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(images), self.batch_size):
                px = self.processor(
                    images=[im.convert("RGB") for im in images[i : i + self.batch_size]],
                    return_tensors="pt",
                ).pixel_values.to(self.device, self.dtype)
                out.append(self._normed(self.model.get_image_features(pixel_values=px)))
        return np.concatenate(out)

    def embed_texts(self, texts) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                # CLIP's text encoder sees at most 77 tokens; long web captions are truncated.
                enc = self.processor(
                    text=list(texts[i : i + self.batch_size]),
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                ).to(self.device)
                out.append(self._normed(self.model.get_text_features(**enc)))
        return np.concatenate(out)

    def score(self, images, captions) -> np.ndarray:
        cos = (self.embed_images(images) * self.embed_texts(captions)).sum(axis=1)
        return CLIPSCORE_W * np.clip(cos, 0, None)
