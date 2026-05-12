"""NaVILA inference server — runs in the *navila* conda environment.

Listens on a TCP socket. Each request is a newline-delimited JSON object
containing 8 base64-encoded JPEG frames and an instruction string.
Returns a newline-delimited JSON command.

Protocol
--------
Client → Server (one JSON line):
    {"frames": ["<base64 JPEG>", ...×8], "instruction": "<text>"}

Server → Client (one JSON line), one of:
    {"action": "MOVE_FORWARD", "distance_cm": 25,  "raw": "<model output>"}
    {"action": "TURN_LEFT",    "degree": 15,        "raw": "<model output>"}
    {"action": "TURN_RIGHT",   "degree": 15,        "raw": "<model output>"}
    {"action": "STOP",                              "raw": "<model output>"}

Usage (from repo root, in navila conda env)
-------------------------------------------
    PYTHONPATH=$PYTHONPATH:NaVILA \
    /srv/local/ganeshr3/conda/envs/navila/bin/python \
        legged-loco/scripts/navila_server.py \
        --model_path NaVILA/checkpoints/navila-llama3-8b-8f \
        --port 15432

The server prints "READY" to stdout once the model is loaded and it is
accepting connections. The run script waits for this signal.
"""

import argparse
import base64
import copy
import io
import json
import re
import socket
import sys

import numpy as np
import torch
from PIL import Image

# NaVILA / LLaVA imports — only available in the navila conda env
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NUM_FRAMES = 8
_PAD_SIZE   = (512, 512)   # black pad frames match navila_trainer.py


def sample_and_pad(images: list, num_frames: int = _NUM_FRAMES) -> list:
    """Replicate sample_and_pad_images from navila_trainer.py exactly."""
    frames = copy.deepcopy(images)
    while len(frames) < num_frames:
        frames.insert(0, Image.new("RGB", _PAD_SIZE, color=(0, 0, 0)))
    latest = frames[-1]
    idxs   = np.linspace(0, len(frames) - 1, num=num_frames - 1,
                         endpoint=False, dtype=int)
    return [frames[i] for i in idxs] + [latest]


def build_prompt(instruction: str, num_frames: int) -> str:
    """Replicate the prompt from navila_trainer.py exactly."""
    interleaved = "<image>\n" * (num_frames - 1)
    return (
        f"Imagine you are a robot programmed for navigation tasks. "
        f"You have been given a video of historical observations {interleaved}, "
        f"and current observation <image>\n. "
        f'Your assigned task is: "{instruction}" '
        f"Analyze this series of images to decide your next action, which could "
        f"be turning left or right by a specific degree, moving forward a certain "
        f"distance, or stop if the task is completed."
    )


_PATTERNS = {
    "stop":          re.compile(r"\bstop\b",           re.IGNORECASE),
    "move_forward":  re.compile(r"\bis move forward\b", re.IGNORECASE),
    "turn_left":     re.compile(r"\bis turn left\b",    re.IGNORECASE),
    "turn_right":    re.compile(r"\bis turn right\b",   re.IGNORECASE),
}
_DIST_RE   = re.compile(r"move forward (\d+) cm",  re.IGNORECASE)
_LEFT_RE   = re.compile(r"turn left (\d+) degree",  re.IGNORECASE)
_RIGHT_RE  = re.compile(r"turn right (\d+) degree", re.IGNORECASE)


def _snap(value: int, grid: list[int]) -> int:
    return min(grid, key=lambda x: abs(x - value))


def parse_output(text: str) -> dict:
    """Parse NaVILA text output → command dict (matches navila_trainer.py logic)."""
    if _PATTERNS["move_forward"].search(text):
        m = _DIST_RE.search(text)
        dist = int(m.group(1)) if m else 25
        dist = _snap(dist, [25, 50, 75])
        return {"action": "MOVE_FORWARD", "distance_cm": dist, "raw": text}

    if _PATTERNS["turn_left"].search(text):
        m = _LEFT_RE.search(text)
        deg = int(m.group(1)) if m else 15
        deg = _snap(deg, [15, 30, 45])
        return {"action": "TURN_LEFT", "degree": deg, "raw": text}

    if _PATTERNS["turn_right"].search(text):
        m = _RIGHT_RE.search(text)
        deg = int(m.group(1)) if m else 15
        deg = _snap(deg, [15, 30, 45])
        return {"action": "TURN_RIGHT", "degree": deg, "raw": text}

    # Default: stop (covers explicit "stop" and unrecognised outputs)
    return {"action": "STOP", "raw": text}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, tokenizer, image_processor, frames: list, instruction: str) -> str:
    """Run NaVILA on a list of PIL Images and return raw text output."""
    frames = sample_and_pad(frames, _NUM_FRAMES)
    question = build_prompt(instruction, len(frames))

    conv = conv_templates["llama_3"].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    images_tensor = process_images(frames, image_processor, model.config).to(
        model.device, dtype=torch.float16
    )
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor.half(),
            do_sample=False,
            temperature=0.0,
            max_new_tokens=32,
            use_cache=True,
            stopping_criteria=[stopping],
            pad_token_id=tokenizer.eos_token_id,
        )

    out = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if out.endswith(stop_str):
        out = out[: -len(stop_str)].strip()
    return out


# ---------------------------------------------------------------------------
# Socket server
# ---------------------------------------------------------------------------

def serve(model, tokenizer, image_processor, host: str, port: int):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)

    print(f"READY", flush=True)   # signal to run script
    print(f"[navila_server] Listening on {host}:{port}", flush=True)

    while True:
        conn, addr = server.accept()
        print(f"[navila_server] Client connected: {addr}", flush=True)
        buf = b""
        with conn:
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        req = json.loads(line.decode())
                        frames = [
                            Image.open(io.BytesIO(base64.b64decode(f))).convert("RGB")
                            for f in req["frames"]
                        ]
                        instruction = req.get("instruction", "Navigate to the goal.")
                        raw = run_inference(model, tokenizer, image_processor,
                                            frames, instruction)
                        cmd = parse_output(raw)
                        print(f"[navila_server] {raw!r}  →  {cmd}", flush=True)
                        conn.sendall((json.dumps(cmd) + "\n").encode())
                    except Exception as e:
                        err = {"action": "STOP", "raw": "", "error": str(e)}
                        print(f"[navila_server] ERROR: {e}", flush=True)
                        conn.sendall((json.dumps(err) + "\n").encode())
        print(f"[navila_server] Client disconnected.", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="Path to NaVILA checkpoint directory")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=15432)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"[navila_server] Loading model from {args.model_path} ...", flush=True)
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, model_name
    )
    model = model.eval()
    print(f"[navila_server] Model loaded: {model_name}", flush=True)

    serve(model, tokenizer, image_processor, args.host, args.port)


if __name__ == "__main__":
    main()
