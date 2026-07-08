"""
PII Layer — Phase 3.

Two responsibilities:
1. TEXT: detect sensitive entities, replace with consistent tokens,
   maintain a registry so real values can be substituted back after pipeline.
2. IMAGE: blur faces and license plates before any image leaves the system.

Uses spaCy for NER (runs locally, no API call).
Uses OpenCV Haar cascades for face detection (built into opencv, no download).

Token format: [PERSON_1], [PERSON_2], [ID_1], [DATE_1] etc.
Consistency rule: same real value always gets same token within one session.
"""

import re
import io
import cv2
import numpy as np
from PIL import Image

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    _nlp = None
    SPACY_AVAILABLE = False


# Maps spaCy entity labels to our token category names
LABEL_TO_CATEGORY = {
    "PERSON":   "PERSON",
    "ORG":      "ORG",
    "GPE":      "LOCATION",
    "LOC":      "LOCATION",
    "DATE":     "DATE",
    "TIME":     "TIME",
    "MONEY":    "AMOUNT",
    "CARDINAL": "NUMBER",
    "FAC":      "LOCATION",
    "NORP":     "GROUP",
}

# Regex patterns for PII not caught by NER
REGEX_PATTERNS = {
    "ID": [
        r"\b\d{5}-\d{7}-\d{1}\b",          # Pakistan CNIC format
        r"\b[A-Z]{1,2}\d{6,9}\b",           # Passport numbers
        r"\b\d{9,12}\b",                     # Generic ID numbers
    ],
    "CONTACT": [
        r"\b\+?\d[\d\s\-]{9,14}\d\b",       # Phone numbers
        r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",  # Emails
    ],
    "FINANCIAL": [
        r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",  # Card numbers
        r"\bIBAN[:\s]?[A-Z]{2}\d{2}[A-Z0-9]{4,}\b",         # IBAN
    ],
    "PLATE": [
        r"\b[A-Z]{2,3}[\s\-]?\d{3,4}\b",    # Vehicle plates
    ],
}


class PIIRegistry:
    """
    Maintains a two-way mapping between real values and tokens.
    One registry per pipeline run — never persisted or shared.
    """
    def __init__(self):
        self._real_to_token: dict[str, str] = {}
        self._token_to_real: dict[str, str] = {}
        self._counters: dict[str, int] = {}

    def get_or_create_token(self, real_value: str, category: str) -> str:
        key = real_value.strip()
        if key in self._real_to_token:
            return self._real_to_token[key]
        count = self._counters.get(category, 0) + 1
        self._counters[category] = count
        token = f"[{category}_{count}]"
        self._real_to_token[key] = token
        self._token_to_real[token] = key
        return token

    def restore(self, text: str) -> str:
        """Replace all tokens in text with their original real values."""
        result = text
        for token, real in self._token_to_real.items():
            result = result.replace(token, real)
        return result

    def summary(self) -> dict:
        return {
            "entities_masked": len(self._real_to_token),
            "categories": dict(self._counters),
        }


def mask_text(text: str, registry: PIIRegistry) -> str:
    """
    Replace PII in text with consistent tokens using spaCy NER + regex.
    Returns masked text. Real values stored in registry for later restoration.
    """
    if not text.strip():
        return text

    masked = text

    # --- spaCy NER pass ---
    if SPACY_AVAILABLE and _nlp:
        doc = _nlp(text)
        # Process longest entities first to avoid partial replacements
        entities = sorted(
            [(ent.text, ent.label_) for ent in doc.ents],
            key=lambda x: len(x[0]),
            reverse=True,
        )
        for ent_text, label in entities:
            category = LABEL_TO_CATEGORY.get(label)
            if category:
                token = registry.get_or_create_token(ent_text, category)
                masked = masked.replace(ent_text, token)

    # --- Regex pass (catches IDs, contacts, financial that NER misses) ---
    for category, patterns in REGEX_PATTERNS.items():
        for pattern in patterns:
            matches = re.findall(pattern, masked)
            for match in sorted(set(matches), key=len, reverse=True):
                # Don't re-tokenize something already tokenized
                if match.startswith("[") and match.endswith("]"):
                    continue
                token = registry.get_or_create_token(match, category)
                masked = masked.replace(match, token)

    return masked


def mask_image(image: Image.Image) -> tuple[Image.Image, list[str]]:
    """
    Blur faces and license plates in image before it leaves the system.
    Returns (masked_image, list_of_what_was_blurred).
    """
    arr = np.array(image.convert("RGB"))
    blurred_regions = []

    # --- Face detection using built-in Haar cascade ---
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    for (x, y, w, h) in faces:
        roi = arr[y:y+h, x:x+w]
        blurred = cv2.GaussianBlur(roi, (51, 51), 30)
        arr[y:y+h, x:x+w] = blurred
        blurred_regions.append("face")

    # --- License plate region heuristic ---
    # Plates are typically wide rectangles in lower 40% of image
    h_img, w_img = arr.shape[:2]
    lower_region = arr[int(h_img * 0.6):, :]
    gray_lower = cv2.cvtColor(lower_region, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray_lower, 120, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y_c, w_c, h_c = cv2.boundingRect(cnt)
        aspect = w_c / max(h_c, 1)
        area = w_c * h_c
        if 2.5 < aspect < 6.0 and 800 < area < 15000:
            y_abs = int(h_img * 0.6) + y_c
            roi = arr[y_abs:y_abs+h_c, x:x+w_c]
            blurred = cv2.GaussianBlur(roi, (31, 31), 20)
            arr[y_abs:y_abs+h_c, x:x+w_c] = blurred
            blurred_regions.append("license_plate")

    masked_img = Image.fromarray(arr)
    return masked_img, list(set(blurred_regions))