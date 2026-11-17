"""Jina reranker v2 base (multilingual), int8 ONNX export, scored on CPU; the score is the first output logit."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from .retrieval import with_breadcrumb

REPO = "jinaai/jina-reranker-v2-base-multilingual"
FILES = ["onnx/model_int8.onnx", "tokenizer.json"]


def download(folder="models/jina_v2_base"):
    """Download the model files from Hugging Face and check them against the published SHA-256."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tree = {}
    try:
        for sub in sorted({str(Path(f).parent).replace("\\", "/") for f in FILES}):
            url = f"https://huggingface.co/api/models/{REPO}/tree/main" + ("" if sub == "." else "/" + sub)
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(url, timeout=60) as r:
                        tree.update({f["path"]: f for f in json.load(r)})
                    break
                except OSError:
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
    except OSError:
        if all((folder / Path(f).name).exists() for f in FILES):
            print("warning: Hugging Face unreachable; using the existing model files without re-checking their SHA-256")
            return folder
        raise
    for f in FILES:
        target = folder / Path(f).name
        if not target.exists():
            urllib.request.urlretrieve(f"https://huggingface.co/{REPO}/resolve/main/{f}", target)
        want = (tree[f].get("lfs") or {}).get("oid")
        if want and hashlib.sha256(target.read_bytes()).hexdigest() != want:
            target.unlink()
            raise ValueError(f"SHA-256 mismatch for {f}; file deleted")
    return folder


class JinaReranker:
    def __init__(self, folder="models/jina_v2_base", max_length=512, threads=8):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        folder = Path(folder)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(folder / "model_int8.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
        self.inputs = [i.name for i in self.session.get_inputs()]
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_length, strategy="only_second")
        pad = self.tokenizer.token_to_id("<pad>")
        self.tokenizer.enable_padding(pad_id=pad if pad is not None else 0, pad_token="<pad>")

    def scores(self, query, texts):
        out = []
        for a in range(0, len(texts), 16):
            enc = self.tokenizer.encode_batch([(query, t) for t in texts[a:a + 16]])
            feed = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                    "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
            logits = self.session.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0]
            out += [float(x[0]) if hasattr(x, "__len__") else float(x) for x in logits]
        return out

    def rerank(self, query, chunks):
        s = self.scores(query, [with_breadcrumb(c) for c in chunks])
        order = sorted(range(len(chunks)), key=lambda i: (-s[i], i))
        return [chunks[i] for i in order]
