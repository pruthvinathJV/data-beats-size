"""Caption metrics: CIDEr and CHAIR.

CIDEr uses pycocoevalcap's scorer with a plain-Python tokenizer (lowercase,
punctuation stripped) because the reference PTB tokenizer needs Java.

CHAIR (Rohrbach et al., 2018) is a simplified reimplementation: an object word
counts as hallucinated if its COCO category is neither annotated in the image
nor mentioned in any reference caption. Two-word synonyms are matched first;
plurals are singularized with a small irregular table plus -ies/-es/-s rules
(the original uses a dictionary lemmatizer).
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

_PUNCT = re.compile(r"[^a-z0-9 ]+")
_IRREGULAR = {"men": "man", "women": "woman", "children": "child", "mice": "mouse", "geese": "goose",
              "knives": "knife", "leaves": "leaf", "wolves": "wolf", "shelves": "shelf", "feet": "foot",
              "teeth": "tooth", "oxen": "ox", "people": "people", "sheep": "sheep", "fish": "fish"}


def singular_forms(phrase: str) -> list[str]:
    """The phrase plus candidate singulars of its last word, most specific first."""
    *head, last = phrase.split(" ")
    forms = [last]
    if last in _IRREGULAR:
        forms.append(_IRREGULAR[last])
    if last.endswith("ies"):
        forms.append(last[:-3] + "y")
    if last.endswith("es"):
        forms.append(last[:-2])
    if last.endswith("s"):
        forms.append(last[:-1])
    return [" ".join([*head, f]) for f in forms]


def tokenize(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", text.lower()).split())


def cider(refs: dict, cands: dict) -> tuple[float, np.ndarray]:
    """Corpus CIDEr-D and per-image scores. refs: id -> [str], cands: id -> str."""
    from pycocoevalcap.cider.cider import Cider

    gts = {i: [tokenize(r) for r in rs] for i, rs in refs.items()}
    res = {i: [tokenize(cands[i])] for i in refs}
    score, per_image = Cider().compute_score(gts, res)
    return float(score), np.asarray(per_image)


class Chair:
    def __init__(self, synonyms_path, instances_path):
        self.word_to_cat: dict[str, str] = {}
        for line in Path(synonyms_path).read_text().splitlines():
            words = [w.strip() for w in line.strip().split(",") if w.strip()]
            for w in words:
                self.word_to_cat[w.lower()] = words[0].lower()
        self.max_len = max(len(w.split()) for w in self.word_to_cat)
        inst = json.loads(Path(instances_path).read_text())
        cat_name = {c["id"]: c["name"] for c in inst["categories"]}
        self.annotated: dict[int, set] = defaultdict(set)
        for a in inst["annotations"]:
            name = cat_name[a["category_id"]].lower()
            self.annotated[a["image_id"]].add(self.word_to_cat.get(name, name))

    def objects(self, caption: str) -> set[str]:
        toks = tokenize(caption).split()
        found, i = set(), 0
        while i < len(toks):
            for n in range(self.max_len, 0, -1):
                if i + n > len(toks):
                    continue
                phrase = " ".join(toks[i : i + n])
                cat = next((self.word_to_cat[c] for c in singular_forms(phrase) if c in self.word_to_cat), None)
                if cat:
                    found.add(cat)
                    i += n
                    break
            else:
                i += 1
        return found

    def score(self, image_ids, captions, refs) -> dict:
        n_obj = n_hall = n_caps_hall = 0
        per_image = []
        for img_id, cap, rs in zip(image_ids, captions, refs):
            truth = self.annotated.get(int(img_id), set()) | set().union(*(self.objects(r) for r in rs))
            mentioned = self.objects(cap)
            hall = mentioned - truth
            n_obj += len(mentioned)
            n_hall += len(hall)
            n_caps_hall += bool(hall)
            per_image.append(bool(hall))
        return {"CHAIRi": n_hall / max(n_obj, 1), "CHAIRs": n_caps_hall / max(len(captions), 1),
                "objects_per_caption": n_obj / max(len(captions), 1), "per_image": np.asarray(per_image)}
