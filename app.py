import os

# ---------------------------------------------------------------------------
# Multi-Threading Safeguards (Prevents Segmentation Faults / Memory Crashes)
# ---------------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import io
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
import pydicom  # DICOM (.dcm) support
from flask import Flask, render_template, request, jsonify

# Pydicom LUT helpers (location differs between pydicom 2.x and 3.x)
try:
    from pydicom.pixels import apply_modality_lut, apply_voi_lut
except ImportError:  # pydicom < 3
    from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut

# Force PyTorch single-thread CPU execution
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 60 * 1024 * 1024   # 60 MB upload ceiling (large DICOMs)

# ---------------------------------------------------------------------------
# System Framework Parameters & Configurations
# ---------------------------------------------------------------------------
app.config['threshold'] = 0.40                    # Calibrated Operational Decision Boundary (40%)
app.config['demographic_base'] = 'AHFCR Cohort'   # Abeokuta Heart Failure Clinical Registry (N=452)
app.config['reference'] = 'Ogah et al. (2014)'    # Regional Etiology & Prevalence Baseline
app.config['calibration_target'] = 'MLE Calibrated' # Maximum Likelihood Estimation Parameter Tuning

# ---------------------------------------------------------------------------
# Chest X-Ray Validation Gate Settings
# Tune these with evaluate_gate.py on your own images — do not guess them.
# ---------------------------------------------------------------------------
GATE = {
    # Structural (hard rejections)
    'min_side_px': 128,            # thumbnails / icons are rejected
    'aspect_min': 0.55,            # width / height
    'aspect_max': 1.80,
    'max_colorfulness': 12.0,      # mean channel spread; X-rays are grayscale
    'max_colored_pixel_frac': 0.05,# fraction of pixels with strong colour
    'max_dark_frac': 0.48,         # limb / extremity scans have a lot of black space
    'max_bright_frac': 0.55,       # documents, screenshots, blank pages
    'min_contrast_std': 20.0,      # flat / blank images
    'max_edge_frac': 0.14,         # text, charts, screenshots have dense sharp edges

    # Anatomical layout (soft checks → manual review)
    'min_mediastinum_contrast': 8.0,   # centre must be brighter than both lung fields
    'min_symmetry_corr': 0.35,         # left/right mirror correlation

    # Deep learning gate
    'reject_non_chest_prob': 0.50,     # class-2 probability that forces rejection
    'review_non_chest_prob': 0.20,     # class-2 probability that triggers review
    'min_chest_prob': 0.55,            # best chest-class probability required
    'max_norm_entropy': 0.85,          # 3-class entropy (0–1); higher = model unsure
    'max_tta_spread': 0.15,            # disagreement between augmented views
    'anatomy_override_prob': 0.95,     # model confidence that can override a soft anatomy fail

    # Optional CLIP zero-shot second opinion (off by default; ~600 MB download)
    'enable_clip': os.environ.get('ENABLE_CLIP_GATE', '0') == '1',
    'clip_reject_below': 0.50,
    'clip_review_below': 0.70,
}

ALLOWED_DICOM_MODALITIES = {'CR', 'DX', 'DR', 'RG'}           # plain radiography
NEUTRAL_DICOM_MODALITIES = {'', 'OT'}                         # unknown / "other" → rely on image checks
CHEST_KEYWORDS = ('CHEST', 'THORAX', 'THORACIC', 'LUNG', 'CXR', 'TORAX', 'PECHO', 'POITRINE', 'THX')

# PyTorch Execution Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'cardiomegaly_3class_mobilenetv2.pt')


class GateRejection(Exception):
    """Raised when an upload is definitively not a usable chest X-ray."""
    def __init__(self, reason, stage):
        super().__init__(reason)
        self.reason = reason
        self.stage = stage


# ---------------------------------------------------------------------------
# 0. Robust Image Loading (DICOM metadata gate + correct pixel handling)
# ---------------------------------------------------------------------------
def _to_uint8(arr):
    """Percentile-normalise any numeric array to 8-bit."""
    arr = arr.astype(np.float64)
    lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    if hi - lo < 1e-6:
        lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-6:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = np.clip((arr - lo) / (hi - lo), 0, 1) * 255.0
    return arr.astype(np.uint8)


def load_dicom(file_bytes, checks):
    """Reads a DICOM, validates its metadata, and returns a correctly windowed PIL image."""
    try:
        ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
    except Exception as e:
        raise GateRejection(f"Unreadable DICOM file ({e}).", 'dicom')

    modality = str(getattr(ds, 'Modality', '') or '').upper().strip()
    body_part = str(getattr(ds, 'BodyPartExamined', '') or '').upper()
    descriptions = ' '.join(str(getattr(ds, tag, '') or '') for tag in
                            ('StudyDescription', 'SeriesDescription', 'ProtocolName')).upper()

    if modality and modality not in ALLOWED_DICOM_MODALITIES and modality not in NEUTRAL_DICOM_MODALITIES:
        raise GateRejection(f"DICOM modality '{modality}' is not a plain radiograph (expected CR/DX).", 'dicom')

    if body_part and not any(k in body_part for k in CHEST_KEYWORDS):
        raise GateRejection(f"DICOM body part '{body_part}' is not the chest.", 'dicom')

    frames = int(getattr(ds, 'NumberOfFrames', 1) or 1)
    if frames > 1:
        raise GateRejection(f"Multi-frame DICOM ({frames} frames) — not a single chest radiograph.", 'dicom')

    metadata_confirms_chest = (modality in ALLOWED_DICOM_MODALITIES and
                               (any(k in body_part for k in CHEST_KEYWORDS) or
                                any(k in descriptions for k in CHEST_KEYWORDS)))
    checks.append({'check': 'DICOM metadata',
                   'result': 'chest radiograph confirmed' if metadata_confirms_chest else 'inconclusive',
                   'modality': modality or 'unknown', 'body_part': body_part or 'unknown'})

    try:
        arr = ds.pixel_array
    except Exception as e:
        raise GateRejection(
            f"Cannot decode DICOM pixels ({e}). Install pylibjpeg / pylibjpeg-libjpeg / gdcm for compressed files.",
            'dicom')

    photometric = str(getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')).upper()
    if arr.ndim == 3 and arr.shape[-1] in (3, 4) and photometric not in ('MONOCHROME1', 'MONOCHROME2'):
        return Image.fromarray(_to_uint8(arr[..., :3])), metadata_confirms_chest

    arr = apply_modality_lut(arr, ds)
    try:
        arr = apply_voi_lut(arr, ds)          # use the scanner's own window/level when present
    except Exception:
        pass
    img8 = _to_uint8(arr)
    if photometric == 'MONOCHROME1':          # MONOCHROME1 stores bone as dark — invert it
        img8 = 255 - img8
    return Image.fromarray(img8, mode='L'), metadata_confirms_chest


def load_standard_image(file_bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.load()
    except Exception:
        raise GateRejection("File is not a readable image (use JPG, PNG or DICOM).", 'decode')

    img = ImageOps.exif_transpose(img)
    if img.mode in ('I;16', 'I;16B', 'I;16L', 'I', 'F'):      # 16-bit PNG / TIFF radiographs
        img = Image.fromarray(_to_uint8(np.array(img)), mode='L')
    elif img.mode in ('RGBA', 'LA', 'P'):
        bg = Image.new('RGB', img.size, (0, 0, 0))
        rgba = img.convert('RGBA')
        bg.paste(rgba, mask=rgba.split()[-1])
        img = bg
    return img


# ---------------------------------------------------------------------------
# 1. Structural Gatekeeper (Pre-Inference Scan Validation)
# ---------------------------------------------------------------------------
def verify_chest_xray_structure(pil_img, checks):
    """
    Hard checks reject obvious non-radiographs (photos, documents, limbs, blanks).
    Soft anatomical checks return False when the layout doesn't look like a frontal chest film,
    which sends the scan to manual review instead of forcing a decision.
    Returns: anatomy_ok (bool)
    """
    w, h = pil_img.size

    # Check 1: Resolution
    if min(w, h) < GATE['min_side_px']:
        raise GateRejection(f"Image too small ({w}×{h}px) for radiographic assessment.", 'structure')

    # Check 2: Aspect Ratio Bounds
    aspect_ratio = w / float(h)
    if not (GATE['aspect_min'] <= aspect_ratio <= GATE['aspect_max']):
        raise GateRejection(f"Invalid scan aspect ratio ({round(aspect_ratio, 2)}:1).", 'structure')

    small = pil_img.copy()
    small.thumbnail((256, 256))

    # Check 3: Colour content (X-rays are grayscale; photos are not)
    rgb = np.asarray(small.convert('RGB'), dtype=np.float64)
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    colorfulness = float(spread.mean())
    colored_frac = float((spread > 40).mean())
    checks.append({'check': 'colour', 'colorfulness': round(colorfulness, 2),
                   'colored_pixel_frac': round(colored_frac, 3)})
    if colorfulness > GATE['max_colorfulness'] or colored_frac > GATE['max_colored_pixel_frac']:
        raise GateRejection("Colour photograph or graphic detected — radiographs are grayscale.", 'structure')

    gray = np.asarray(small.convert('L'), dtype=np.float64)
    gh, gw = gray.shape

    # Check 4: Pitch-black background ratio (limb scans feature large empty black space)
    dark_ratio = float((gray < 30).mean())
    # Check 5: Near-white ratio (documents, screenshots, blank pages)
    bright_ratio = float((gray > 235).mean())
    # Check 6: Global contrast
    contrast = float(gray.std())
    # Check 7: Sharp-edge density (text / UI / charts)
    gx = np.abs(np.diff(gray, axis=1))[:-1, :]
    gy = np.abs(np.diff(gray, axis=0))[:, :-1]
    edge_frac = float(((gx + gy) > 80).mean())

    checks.append({'check': 'intensity', 'dark_frac': round(dark_ratio, 3), 'bright_frac': round(bright_ratio, 3),
                   'contrast_std': round(contrast, 2), 'edge_frac': round(edge_frac, 3)})

    if dark_ratio > GATE['max_dark_frac']:
        raise GateRejection(f"Extremity/Limb scan detected ({round(dark_ratio * 100, 1)}% empty background).", 'structure')
    if bright_ratio > GATE['max_bright_frac']:
        raise GateRejection("Mostly white image — looks like a document, screenshot or blank page.", 'structure')
    if contrast < GATE['min_contrast_std']:
        raise GateRejection("Image has almost no contrast — blank or corrupted scan.", 'structure')
    if edge_frac > GATE['max_edge_frac']:
        raise GateRejection("Dense sharp edges detected — looks like text, a chart or a screenshot.", 'structure')

    # Soft Check A: Mediastinum brighter than both lung fields (frontal chest layout)
    band = gray[int(gh * 0.35):int(gh * 0.75), :]
    profile = np.convolve(band.mean(axis=0), np.ones(5) / 5.0, mode='same')
    centre = profile[int(gw * 0.42):int(gw * 0.58)].mean()
    left_lung = profile[int(gw * 0.10):int(gw * 0.42)].min()
    right_lung = profile[int(gw * 0.58):int(gw * 0.90)].min()
    mediastinum_contrast = float(min(centre - left_lung, centre - right_lung))

    # Soft Check B: Bilateral symmetry
    sym = np.asarray(pil_img.convert('L').resize((64, 64)), dtype=np.float64)
    left_half, right_half = sym[:, :32].ravel(), np.fliplr(sym[:, 32:]).ravel()
    if left_half.std() < 1e-6 or right_half.std() < 1e-6:
        symmetry = 0.0
    else:
        symmetry = float(np.corrcoef(left_half, right_half)[0, 1])

    layout_ok = mediastinum_contrast >= GATE['min_mediastinum_contrast']
    symmetry_ok = symmetry >= GATE['min_symmetry_corr']
    checks.append({'check': 'anatomy', 'mediastinum_contrast': round(mediastinum_contrast, 2),
                   'symmetry_corr': round(symmetry, 3), 'layout_ok': layout_ok, 'symmetry_ok': symmetry_ok})

    return layout_ok and symmetry_ok


# ---------------------------------------------------------------------------
# 2. Load PyTorch Model
# ---------------------------------------------------------------------------
def load_xray_model():
    """
    Loads unified 3-Class MobileNetV2 model:
    - Class 0: Normal Chest Radiograph
    - Class 1: Cardiomegaly Indicated
    - Class 2: Non-Chest / Invalid Input
    """
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"\n[WARNING] Model weights not found at: {MODEL_WEIGHTS_PATH}\n")
        return None

    try:
        try:
            model = models.mobilenet_v2(weights=None)
        except TypeError:  # older torchvision
            model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.last_channel, 3)
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"\n[SUCCESS] Unified 3-Class MobileNetV2 loaded on {device}!\n")
        return model
    except Exception as e:
        print(f"\n[ERROR] Failed loading weights: {e}\n")
        return None

model = load_xray_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def tta_views(pil_gray_rgb):
    """Test-time augmentation views: original, 92% centre crop, auto-contrast."""
    w, h = pil_gray_rgb.size
    cw, ch = int(w * 0.92), int(h * 0.92)
    left, top = (w - cw) // 2, (h - ch) // 2
    return [
        pil_gray_rgb,
        pil_gray_rgb.crop((left, top, left + cw, top + ch)),
        ImageOps.autocontrast(pil_gray_rgb, cutoff=1),
    ]


def run_model(pil_img):
    """Averaged 3-class probabilities plus disagreement between augmented views."""
    base = pil_img.convert('L').convert('RGB')
    batch = torch.stack([transform(v) for v in tta_views(base)]).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1).cpu().numpy()
    mean_probs = probs.mean(axis=0)
    chest_best_per_view = probs[:, :2].max(axis=1)
    spread = float(chest_best_per_view.max() - chest_best_per_view.min())
    entropy = float(-(mean_probs * np.log(mean_probs + 1e-9)).sum() / math.log(3))
    return mean_probs, spread, entropy


# ---------------------------------------------------------------------------
# 2b. Optional CLIP zero-shot second opinion
# ---------------------------------------------------------------------------
_clip = None
CLIP_PROMPTS = [
    "a frontal chest X-ray radiograph",               # index 0 = target
    "an X-ray of a hand, arm, leg, knee or foot",
    "an X-ray of a skull or teeth",
    "an abdominal or pelvic X-ray",
    "a CT scan slice",
    "an MRI scan",
    "an ultrasound image",
    "a photograph of a person, animal or object",
    "a document, screenshot, chart or text",
]

def get_clip():
    global _clip
    if _clip is None:
        from transformers import CLIPModel, CLIPProcessor
        name = "openai/clip-vit-base-patch32"
        _clip = (CLIPModel.from_pretrained(name).to(device).eval(), CLIPProcessor.from_pretrained(name))
    return _clip

def clip_chest_probability(pil_img):
    clip_model, processor = get_clip()
    inputs = processor(text=CLIP_PROMPTS, images=pil_img.convert('RGB'), return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        probs = clip_model(**inputs).logits_per_image.softmax(dim=1)[0]
    return float(probs[0].item())


# ---------------------------------------------------------------------------
# 3. Dynamic Clinical Insights Evaluator
# ---------------------------------------------------------------------------
def generate_clinical_evaluation(prob_cardiomegaly, prob_normal):
    p_cardio = prob_cardiomegaly / 100.0
    p_norm = prob_normal / 100.0

    epsilon = 1e-7
    entropy = -(p_cardio * math.log2(p_cardio + epsilon) + p_norm * math.log2(p_norm + epsilon))
    uncertainty_score = round(entropy * 100, 2)
    risk_index = round(prob_cardiomegaly, 1)

    if prob_cardiomegaly >= 75.0:
        status_level = "High Risk"
        risk_tier = "high"
        primary_condition = "Severe Heart Enlargement (Cardiomegaly)"
        damaged_region = "Lower Left Heart Apex & Outer Outline"
        ctr_assessment = "Significantly Elevated Heart-to-Chest Width (Over 55%)"
        recommended_action = "Immediate Cardiology referral. Order Echocardiogram and ECG for CHF staging."
        header_banner = "🔴 High Risk / Cardiomegaly Detected"
        is_safe = False

    elif 50.0 <= prob_cardiomegaly < 75.0:
        status_level = "Moderate Risk"
        risk_tier = "moderate"
        primary_condition = "Moderate Heart Enlargement"
        damaged_region = "Heart Shadow Showing Mild Strain"
        ctr_assessment = "Slightly Elevated Heart-to-Chest Width (51%–54%)"
        recommended_action = "Correlate with clinical risk factors. Recommend confirmatory Echocardiography."
        header_banner = "🟡 Moderate Risk / Cardiomegaly Indicated"
        is_safe = False

    elif 30.0 <= prob_cardiomegaly < 50.0:
        status_level = "Low Risk / Borderline"
        risk_tier = "moderate"
        primary_condition = "Borderline Heart Size"
        damaged_region = "Slight Variation in Heart Outline"
        ctr_assessment = "Near-Normal Heart-to-Chest Width (48%–50%)"
        recommended_action = "Sub-threshold scan. Cross-evaluate with clinical biomarker engine."
        header_banner = "🟡 Equivocal / Borderline Scan"
        is_safe = True

    else:
        status_level = "Low Risk"
        risk_tier = "normal"
        primary_condition = "Normal Heart Size & Shape"
        damaged_region = "No Damaged or Stressed Regions Detected"
        ctr_assessment = "Normal Heart-to-Chest Width (Under 48%)"
        recommended_action = "No radiographic evidence of heart enlargement. Continue standard clinical routine."
        header_banner = "🟢 Safe / Normal Scan"
        is_safe = True

    return {
        'status_level': status_level,
        'risk_tier': risk_tier,
        'primary_condition': primary_condition,
        'damaged_region': damaged_region,
        'ctr_assessment': ctr_assessment,
        'recommended_action': recommended_action,
        'header_banner': header_banner,
        'uncertainty_score': uncertainty_score,
        'risk_index': risk_index,
        'is_safe': is_safe
    }


# ---------------------------------------------------------------------------
# 4. Response Builders
# ---------------------------------------------------------------------------
def gate_response(reason, stage, checks, review=False):
    """Rejected (definitely not a usable chest X-ray) or Review (unsure → human decides)."""
    if review:
        banner = "⚠️ Scan Withheld – Manual Review Required"
        diagnosis = 'Uncertain Input – Manual Review'
        condition = f'Uncertain Input: {reason}'
        region = 'N/A - Withheld for Manual Review'
        action = f'Automated analysis withheld ({reason}). A clinician should confirm this is a frontal (PA/AP) chest X-ray, or upload a clearer image.'
    else:
        banner = "⚠️ Invalid Scan Uploaded"
        diagnosis = 'Non-Chest X-Ray Input'
        condition = f'Invalid Input: {reason}'
        region = f'N/A - Rejected by {stage.title()} Gate'
        action = f'Upload rejected ({reason}). Please upload a valid frontal chest X-ray.'

    return {
        'diagnosis': diagnosis, 'Diagnosis': diagnosis,
        'status': 'Rejected', 'Status': 'Rejected',       # frontend treats both outcomes as "no diagnosis"
        'gate_decision': 'review' if review else 'rejected',
        'gate_stage': stage,
        'gate_checks': checks,
        'primary_condition': condition, 'primaryCondition': condition,
        'damaged_region': region, 'damagedRegion': region,
        'ctr_assessment': 'Unable to calculate CTR on this image.',
        'recommended_action': action,
        'confidence': 0.0, 'estimated_confidence': 0.0,
        'cardiomegaly_probability': 0.0, 'normal_probability': 0.0,
        'risk_index': 0.0, 'model_uncertainty': 100.0,
        'scan_title': banner, 'header_status': banner, 'scan_status': banner, 'overall_status': banner,
        'is_safe': False,
        'disclaimer': 'Research prototype only. Not a validated diagnostic device. Not for clinical use.'
    }


# ---------------------------------------------------------------------------
# 5. Full Scan Pipeline (shared by the Flask route and evaluate_gate.py)
# ---------------------------------------------------------------------------
def analyse_upload(file_bytes, filename='', content_type=''):
    checks = []
    try:
        # --- LEVEL 0: DECODE (+ DICOM METADATA GATE) ---
        is_dicom = filename.lower().endswith('.dcm') or 'dicom' in (content_type or '').lower() \
            or file_bytes[128:132] == b'DICM'
        metadata_confirms_chest = False
        if is_dicom:
            raw_image, metadata_confirms_chest = load_dicom(file_bytes, checks)
        else:
            raw_image = load_standard_image(file_bytes)

        # --- LEVEL 1: STRUCTURAL GATEKEEPER ---
        anatomy_ok = verify_chest_xray_structure(raw_image, checks)

        # --- LEVEL 2: DEEP LEARNING INFERENCE (with test-time augmentation) ---
        probs, tta_spread, norm_entropy = run_model(raw_image)
        raw_normal, raw_cardiomegaly, raw_non_chest = (float(p) for p in probs)
        max_chest_raw_prob = max(raw_normal, raw_cardiomegaly)
        predicted_class = int(np.argmax(probs))
        checks.append({'check': 'model', 'p_normal': round(raw_normal, 4), 'p_cardiomegaly': round(raw_cardiomegaly, 4),
                       'p_non_chest': round(raw_non_chest, 4), 'tta_spread': round(tta_spread, 3),
                       'norm_entropy': round(norm_entropy, 3)})

        # --- LEVEL 3: REJECTION GUARD ---
        if predicted_class == 2 or raw_non_chest >= GATE['reject_non_chest_prob']:
            raise GateRejection(f"Non-chest radiograph detected (non-chest score {round(raw_non_chest * 100, 1)}%)", 'model')
        if max_chest_raw_prob < GATE['min_chest_prob']:
            raise GateRejection(f"Low thoracic match ({round(max_chest_raw_prob * 100, 1)}%)", 'model')

        # --- LEVEL 4: OPTIONAL CLIP SECOND OPINION ---
        if GATE['enable_clip']:
            try:
                clip_p = clip_chest_probability(raw_image)
                checks.append({'check': 'clip', 'p_chest_xray': round(clip_p, 4)})
                if clip_p < GATE['clip_reject_below'] and not metadata_confirms_chest:
                    raise GateRejection(f"Independent vision check disagrees (chest match {round(clip_p * 100, 1)}%)", 'clip')
                if clip_p < GATE['clip_review_below'] and not metadata_confirms_chest:
                    return gate_response(f"independent vision check uncertain ({round(clip_p * 100, 1)}%)", 'clip', checks, review=True)
            except GateRejection:
                raise
            except Exception as e:
                checks.append({'check': 'clip', 'error': str(e)})

        # --- LEVEL 5: UNCERTAINTY GUARD (→ manual review, never a forced diagnosis) ---
        review_reasons = []
        if raw_non_chest >= GATE['review_non_chest_prob']:
            review_reasons.append(f"non-chest score {round(raw_non_chest * 100, 1)}%")
        if norm_entropy > GATE['max_norm_entropy']:
            review_reasons.append("model output highly uncertain")
        if tta_spread > GATE['max_tta_spread']:
            review_reasons.append("unstable prediction across image views")
        if not anatomy_ok and max_chest_raw_prob < GATE['anatomy_override_prob'] and not metadata_confirms_chest:
            review_reasons.append("anatomy does not match a frontal chest film (lateral view, rotation or cropping?)")
        if review_reasons:
            return gate_response('; '.join(review_reasons), 'uncertainty', checks, review=True)

    except GateRejection as g:
        return gate_response(g.reason, g.stage, checks, review=False)

    # --- ACCEPTED: CLINICAL EVALUATION ---
    binary_total = raw_normal + raw_cardiomegaly + 1e-5
    norm_normal = round((raw_normal / binary_total) * 100, 2)
    norm_cardiomegaly = round(100.0 - norm_normal, 2)     # guarantees the pair sums to exactly 100

    eval_data = generate_clinical_evaluation(norm_cardiomegaly, norm_normal)
    diagnosis_label = "Cardiomegaly" if raw_cardiomegaly > raw_normal else "Normal"
    confidence = max(norm_normal, norm_cardiomegaly)

    return {
        'diagnosis': diagnosis_label, 'Diagnosis': diagnosis_label,
        'predicted_class': 1 if raw_cardiomegaly > raw_normal else 0,
        'gate_decision': 'accepted',
        'gate_checks': checks,
        'confidence': confidence, 'estimated_confidence': confidence,
        'cardiomegaly_probability': norm_cardiomegaly,
        'normal_probability': norm_normal,
        'status': eval_data['status_level'], 'Status': eval_data['status_level'],
        'risk_tier': eval_data['risk_tier'],
        'primary_condition': eval_data['primary_condition'], 'primaryCondition': eval_data['primary_condition'],
        'damaged_region': eval_data['damaged_region'], 'damagedRegion': eval_data['damaged_region'],
        'ctr_assessment': eval_data['ctr_assessment'],
        'recommended_action': eval_data['recommended_action'],
        'risk_index': eval_data['risk_index'],
        'model_uncertainty': eval_data['uncertainty_score'],
        'scan_title': eval_data['header_banner'], 'header_status': eval_data['header_banner'],
        'scan_status': eval_data['header_banner'], 'overall_status': eval_data['header_banner'],
        'is_safe': eval_data['is_safe'],
        'disclaimer': 'Research prototype only. Not a validated diagnostic device. Not for clinical use.'
    }


# ---------------------------------------------------------------------------
# 6. Flask Application Routes
# ---------------------------------------------------------------------------

@app.route('/')
def home():
    return render_template('index.html', config=app.config)


@app.route('/predict', methods=['POST'])
def predict_biomarker():
    """Tabular Logistic Regression Engine for Clinical Biomarkers"""
    try:
        age = float(request.form.get('age', 50))
        hbp = float(request.form.get('hbp', 120))
        ef = float(request.form.get('ef', 50))
        sc = float(request.form.get('sc', 1.0))

        z = -0.8 + (0.04 * age) + (0.015 * (hbp - 120)) - (0.08 * (ef - 50)) + (1.2 * (sc - 1.0))
        prob = 1.0 / (1.0 + math.exp(-z))
        percentage = round(prob * 100, 2)

        if prob >= app.config['threshold']:
            prediction_text = "🔴 High Risk of Congestive Heart Failure"
        else:
            prediction_text = "🟢 Low Risk of Congestive Heart Failure"
        probability_text = f"Algorithmic Mortality Risk Probability: {percentage}%"

        return render_template('index.html',
                               prediction_text=prediction_text,
                               probability_text=probability_text,
                               calc_percentage=percentage,
                               config=app.config)

    except Exception as e:
        return render_template('index.html', prediction_text="Error processing patient data.", probability_text=str(e), config=app.config)


@app.route('/scan_image', methods=['POST'])
def scan_image():
    """Deep Learning Pipeline Endpoint for Chest Radiographs & DICOM Files"""
    if model is None:
        return jsonify({'error': 'PyTorch model weights not loaded. Place cardiomegaly_3class_mobilenetv2.pt in project root.'}), 500

    file_key = next((k for k in ['xray', 'file', 'image', 'xray_image'] if k in request.files), None)
    if not file_key:
        return jsonify({'error': 'No file payload detected in request.'}), 400

    file = request.files[file_key]
    if file.filename == '':
        return jsonify({'error': 'No file selected for evaluation.'}), 400

    try:
        result = analyse_upload(file.read(), file.filename, file.content_type or '')
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Inference engine execution error: {str(e)}'}), 500


@app.route('/generate_synthetic', methods=['GET'])
def generate_synthetic():
    """Generates 50 synthetic evaluation profiles strictly within AHFCR demographic (39-59 yrs)."""
    synthetic_records = []
    random.seed(None)

    for i in range(1, 51):
        patient_id = f"AHFCR-SYN-{i:02d}"
        age = random.randint(39, 59)

        if random.random() < 0.45:
            sbp = random.randint(110, 130)
            ef = random.randint(52, 68)
            sc = round(random.uniform(0.7, 1.1), 1)
        else:
            sbp = random.randint(138, 185)
            ef = random.randint(22, 45)
            sc = round(random.uniform(1.3, 2.4), 1)

        z = -0.8 + (0.04 * age) + (0.015 * (sbp - 120)) - (0.08 * (ef - 50)) + (1.2 * (sc - 1.0))
        prob = 1.0 / (1.0 + math.exp(-z))
        percentage = round(prob * 100, 2)

        is_high = prob >= app.config['threshold']

        synthetic_records.append({
            'identity': patient_id,
            'age': age,
            'sbp': sbp,
            'ef': ef,
            'sc': sc,
            'risk_class': 'High Risk' if is_high else 'Low Risk',
            'is_high_risk': is_high,
            'percentage': percentage
        })

    return jsonify(synthetic_records)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)