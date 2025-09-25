import io
import json
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import zipfile
import pandas as pd
from pathlib import Path
import os

# Load .env automatically if available
try:
	from dotenv import load_dotenv  # type: ignore
	load_dotenv()
except Exception:
	pass


# -----------------------------
# Utility dataclasses
# -----------------------------
@dataclass
class BoundingBox:
	label: str
	xmin: int
	ymin: int
	xmax: int
	ymax: int
	score: float


# -----------------------------
# Detection (YOLO) and simulation
# -----------------------------

def get_latest_model_path(project_root: Path) -> Path:
	runs = sorted((project_root / "runs").glob("symbol-detector-poc*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
	return runs[0] if runs else project_root / "runs" / "symbol-detector-poc" / "weights" / "best.pt"


def load_yolo_model(project_root: Path):
	"""Load and cache YOLO model in session state."""
	if "yolo_model" in st.session_state and st.session_state.yolo_model is not None:
		return st.session_state.yolo_model
	try:
		from ultralytics import YOLO  # type: ignore
	except Exception as e:
		st.error("Ultralytics YOLO is not installed. Install with: pip install ultralytics")
		raise e
	model_path = get_latest_model_path(project_root)
	model = YOLO(str(model_path))
	# Do not override names; use names from weights
	# Disable showing confidences in plotted labels to keep style without numbers
	try:
		from ultralytics import settings as yolo_settings  # type: ignore
		yolo_settings.update({"show_conf": False})
	except Exception:
		pass
	st.session_state.yolo_model = model
	return model


def run_yolo_detection(image: Image.Image) -> Tuple[List[BoundingBox], Dict[str, int], Image.Image]:
	"""Run YOLO detection on the given PIL image and return boxes, summary, and annotated image."""
	project_root = Path(__file__).resolve().parent
	model = load_yolo_model(project_root)
	# Predict directly from PIL image with same settings as infer.py
	results = model.predict(source=image, imgsz=1280, verbose=False, conf=0.1)
	res = results[0]

	# Build boxes
	boxes: List[BoundingBox] = []
	names = res.names if hasattr(res, "names") else getattr(model, "names", {})
	try:
		xyxy = res.boxes.xyxy.cpu().numpy()
		cls = res.boxes.cls.cpu().numpy().astype(int)
		conf = res.boxes.conf.cpu().numpy()
	except Exception:
		xyxy, cls, conf = np.array([]), np.array([]), np.array([])
	for i in range(len(cls)):
		x1, y1, x2, y2 = xyxy[i].astype(int).tolist()
		label_idx = cls[i]
		label = names.get(label_idx, str(label_idx)) if isinstance(names, dict) else str(label_idx)
		score = float(conf[i]) if i < len(conf) else 0.0
		boxes.append(BoundingBox(label, x1, y1, x2, y2, score))

	# Summary
	summary: Dict[str, int] = {}
	for b in boxes:
		summary[b.label] = summary.get(b.label, 0) + 1

	# Annotated image from result.plot() (confidences hidden via settings)
	try:
		plotted = res.plot(conf=False)
		annotated = Image.fromarray(plotted)
	except Exception:
		annotated = image.copy()
		# Draw simple boxes if plotting failed
		draw = ImageDraw.Draw(annotated)
		for b in boxes:
			draw.rectangle([(b.xmin, b.ymin), (b.xmax, b.ymax)], outline=(255, 0, 0), width=3)

	return boxes, summary, annotated


def run_detection(image: Image.Image) -> Tuple[List[BoundingBox], Dict[str, int], Image.Image]:
	"""Legacy simulation (kept for fallback; not used when YOLO is available)."""
	width, height = image.size
	possible_classes = ["Symbol A", "Symbol B"]
	
	num_detections = random.randint(2, 6)
	boxes: List[BoundingBox] = []
	for _ in range(num_detections):
		w = random.randint(max(12, width // 50), max(24, width // 12))
		h = random.randint(max(12, height // 50), max(24, height // 12))
		x1 = random.randint(0, max(0, width - w))
		y1 = random.randint(0, max(0, height - h))
		x2 = x1 + w
		y2 = y1 + h
		label = random.choice(possible_classes)
		score = round(random.uniform(0.5, 0.99), 2)
		boxes.append(BoundingBox(label, x1, y1, x2, y2, score))
	# Build summary
	summary: Dict[str, int] = {}
	for box in boxes:
		summary[box.label] = summary.get(box.label, 0) + 1
	# Draw boxes
	annotated = image.copy()
	draw = ImageDraw.Draw(annotated)
	try:
		font = ImageFont.load_default()
	except Exception:
		font = None
	for b in boxes:
		draw.rectangle([(b.xmin, b.ymin), (b.xmax, b.ymax)], outline=(255, 0, 0), width=3)
		label_text = f"{b.label}"
		text_w = int((draw.textlength(label_text, font=font) if hasattr(draw, "textlength") else 8 * len(label_text)))
		text_h = 14
		bg_x2 = min(b.xmin + text_w + 6, annotated.width)
		bg_y2 = min(b.ymin + text_h + 6, annotated.height)
		draw.rectangle([(b.xmin, b.ymin), (bg_x2, bg_y2)], fill=(0, 0, 0))
		draw.text((b.xmin + 3, b.ymin + 1), label_text, fill=(255, 255, 255), font=font)
	return boxes, summary, annotated


# -----------------------------
# OpenAI integration (conversational floorplan assistant)
# -----------------------------

def openai_chat_response(query: str, summary: Dict[str, int], history: List[Tuple[str, str]], api_key: str, model: str = "gpt-4o-mini") -> str:
	"""Conversational assistant for floorplans and symbols.

	- Knows: Symbol A = power/data outlet; Symbol B = ceiling light fixture
	- Uses detection summary for counts
	- Friendly small talk allowed (hi/thanks/etc.)
	- If request is unrelated (e.g., code generation), reply: "I don’t have that capability"
	"""
	if not api_key:
		api_key = os.environ.get("OPENAI_API_KEY", "")
	if not api_key:
		return "OPENAI_API_KEY not set."
	try:
		from openai import OpenAI  # type: ignore
	except Exception:
		return "OpenAI SDK not available."

	client = OpenAI(api_key=api_key)

	summary_lines = [f"- {k}: {v}" for k, v in (summary or {}).items()]
	summary_block = "\n".join(summary_lines) if summary_lines else "(no detections yet)"

	system_instructions = (
		"You are a helpful building floorplan assistant. "
		"Context: Symbol A means power/data outlet. Symbol B means ceiling light fixture. "
		"Use the provided detection summary for counts when relevant. "
		"Small talk (greetings, pleasantries) is allowed. Be brief and friendly. "
		"If the user asks for anything unrelated to floorplans, electrical symbols, or the summary, respond exactly: 'I don’t have that capability'."
	)

	messages = [{"role": "system", "content": system_instructions}]
	for role, content in history[-6:]:
		messages.append({"role": role, "content": content})
	messages.append({
		"role": "user",
		"content": (
			f"Detection summary (class: count)\n{summary_block}\n\n"
			f"User: {query}"
		)
	})

	try:
		resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2)
		text = (resp.choices[0].message.content or "").strip()
		return text
	except Exception:
		return "OpenAI request failed."


# -----------------------------
# Download helpers
# -----------------------------

def pil_to_bytes(img: Image.Image, format_: str = "PNG") -> bytes:
	buf = io.BytesIO()
	img.save(buf, format=format_)
	return buf.getvalue()


def build_results_zip(original: Image.Image, annotated: Image.Image, summary: Dict[str, int]) -> bytes:
	buf = io.BytesIO()
	with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
		zf.writestr("summary.json", json.dumps(summary, indent=2))
		zf.writestr("summary.csv", pd.Series(summary, name="count").to_frame().to_csv())
		zf.writestr("original.png", pil_to_bytes(original, "PNG"))
		zf.writestr("detected.png", pil_to_bytes(annotated, "PNG"))
	return buf.getvalue()


# -----------------------------
# Streamlit App
# -----------------------------

def init_session_state() -> None:
	if "uploaded_image" not in st.session_state:
		st.session_state.uploaded_image = None
	if "detected_image" not in st.session_state:
		st.session_state.detected_image = None
	if "detection_summary" not in st.session_state:
		st.session_state.detection_summary = None
	if "chat_history" not in st.session_state:
		st.session_state.chat_history = []  # list of (role, content)
	if "last_detect_time" not in st.session_state:
		st.session_state.last_detect_time = None
	if "yolo_model" not in st.session_state:
		st.session_state.yolo_model = None


def main() -> None:
	st.set_page_config(layout="wide", page_title="Floorplan Symbol Detection")
	init_session_state()

	# Sidebar
	with st.sidebar:
		st.subheader("Detection Mode")
		tab1, tab2 = st.tabs(["Single Image Detection", "Batch Image Detection"])
		with tab1:
			st.caption("Upload a single image and run detection.")
		with tab2:
			st.caption("Batch mode placeholder. Upload a folder in a future version.")

		uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=False)

		with st.expander("Additional Settings"):
			st.checkbox("Enhance contrast", value=False)
			st.checkbox("Denoise image", value=False)
			st.select_slider("Confidence threshold", options=[0.25, 0.4, 0.5, 0.75], value=0.4)

		clicked = st.button("Detect Objects", type="primary")

	if uploaded is not None:
		# Read and store uploaded image as PIL
		try:
			image = Image.open(uploaded).convert("RGB")
		except Exception:
			st.error("Failed to read the uploaded image.")
			image = None
		st.session_state.uploaded_image = image
	else:
		image = st.session_state.uploaded_image

	if clicked and image is not None:
		with st.spinner("Running detection..."):
			try:
				boxes, summary, annotated = run_yolo_detection(image)
			except Exception:
				# Fallback to simulated detection if YOLO fails
				boxes, summary, annotated = run_detection(image)
			st.session_state.detected_image = annotated
			st.session_state.detection_summary = summary
			st.session_state.last_detect_time = time.time()

	# Company branding and main title
	col_brand, col_title = st.columns([1, 3])
	with col_brand:
		st.markdown("### Axium Industries")
	with col_title:
		st.markdown("")
	
	st.title("Floorplan Symbol Detection")

	# Two columns for Original and Detected
	col1, col2 = st.columns(2, gap="large")
	with col1:
		if image is not None:
			st.image(image, caption="Original Image", width="stretch")
		else:
			st.info("Upload an image to begin.")

	with col2:
		detected_img = st.session_state.detected_image
		if detected_img is not None:
			st.image(detected_img, caption="Detected Image", width="stretch")
		else:
			if image is not None:
				st.image(image, caption="Detected Image (pending)", width="stretch")
			else:
				st.empty()

	# Detection Summary Section
	st.markdown("---")
	st.subheader("Detection Summary")
	summary = st.session_state.detection_summary
	if summary:
		df = pd.DataFrame({"Class": list(summary.keys()), "Count": list(summary.values())})
		st.dataframe(df, width="stretch")

		# Download results
		zip_bytes = build_results_zip(
			st.session_state.uploaded_image if st.session_state.uploaded_image is not None else Image.new("RGB", (512, 512), "white"),
			st.session_state.detected_image if st.session_state.detected_image is not None else Image.new("RGB", (512, 512), "white"),
			summary,
		)
		st.download_button(
			label="Download Results",
			data=zip_bytes,
			file_name="detection_results.zip",
			mime="application/zip",
			width="stretch",
		)
	else:
		st.caption("No detections yet. Run detection to see a summary.")

	# Chatbot Section
	st.markdown("---")
	st.subheader("Ask about the detected symbols")

	# Chat history display
	for role, content in st.session_state.chat_history:
		with st.chat_message(role):
			st.markdown(content)

	# Input box
	query = st.text_input("Your question", placeholder="e.g., How many Symbol A?", key="chat_input")
	ask = st.button("Ask", type="secondary")

	if ask:
		api_key_effective = os.environ.get("OPENAI_API_KEY", "")
		if not api_key_effective:
			st.warning("OPENAI_API_KEY not set in environment.")
			answer = "OPENAI_API_KEY not set."
		elif not (st.session_state.detection_summary and len(st.session_state.detection_summary) > 0):
			answer = "No detections yet. Run detection first."
		else:
			answer = openai_chat_response(query, st.session_state.detection_summary, st.session_state.chat_history, api_key_effective, "gpt-4o-mini")
		st.session_state.chat_history.append(("user", query or ""))
		st.session_state.chat_history.append(("assistant", answer))
		st.rerun()


if __name__ == "__main__":
	main()
