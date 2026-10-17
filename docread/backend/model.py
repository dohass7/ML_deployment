import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Optimisations CPU
os.environ["OMP_NUM_THREADS"] = "6"  # Aligne avec ton nombre de cœurs
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["TORCH_NUM_THREADS"] = "6"
os.environ["TRANSFORMERS_VERBOSITY"] = "error" 
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import io
import json
import re

import pymupdf
from PIL import Image
import pillow_heif

import torch
from transformers import (
    Qwen3VLForConditionalGeneration,   # ← CHANGÉ
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info

# Active HEIC
pillow_heif.register_heif_opener()

# ============================================================
# Chargement du modèle Qwen3-VL (CPU)
# ============================================================

MODEL_PATH = "/app/models/Qwen3-VL-2B-Instruct"

print("⏳ Chargement du modèle Qwen3-VL...")

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

llm = Qwen3VLForConditionalGeneration.from_pretrained(   # ← CHANGÉ
    MODEL_PATH,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
)

llm.eval()
torch.set_num_threads(4)

print("✅ Modèle chargé.")


# ============================================================
# Status
# ============================================================

def check_status():
    return f"Qwen2-VL ready (params: {sum(p.numel() for p in llm.parameters()):,})"


# ============================================================
# Chargement des fichiers → liste d'images PIL
# ============================================================

def load_cv_as_images(filename, file_bytes, dpi=80, max_width=1024):
    """
    Retourne une liste d'images PIL.
    Supporte : PDF, JPG, PNG, HEIC, HEIF, WebP, BMP, TIFF.
    """
    name = filename.lower()

    def _resize_if_needed(img, max_w=max_width):
        if img.width > max_w:
            ratio = max_w / img.width
            new_size = (max_w, int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        return img.convert("RGB")

    # PDF
    if name.endswith(".pdf"):
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        images = []
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            images.append(_resize_if_needed(img))
        doc.close()
        return images

    # HEIC / HEIF
    if name.endswith((".heic", ".heif")):
        img = Image.open(io.BytesIO(file_bytes))
        return [_resize_if_needed(img)]

    # Images classiques
    if name.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp",
                      ".tiff", ".tif")):
        img = Image.open(io.BytesIO(file_bytes))
        return [_resize_if_needed(img)]

    # Fallback
    try:
        img = Image.open(io.BytesIO(file_bytes))
        return [_resize_if_needed(img)]
    except Exception as e:
        raise ValueError(f"Format non supporté : {filename} — {e}")


# ============================================================
# Prompt d'extraction
# ============================================================

PROMPT = """Tu es un expert comptable. Analyse cette image de facture et extrait uniquement ces 5 informations.

Réponds UNIQUEMENT avec un objet JSON valide, sans texte avant ni après, sans raisonnement.

JSON à remplir :
{
  "date_facture": "",
  "fournisseur": "",
  "client": "",
  "total_ttc": "",
  "tva": ""
}

Règles :
- "date_facture" : la date d'émission de la facture, au format JJ/MM/AAAA.
- "fournisseur" : le nom de l'entreprise qui émet la facture (pas le client). Il est souvent en HAUT de la facture, avec un logo.
- "client" : le nom de la personne ou de l'entreprise à qui la facture est destinée. Il est souvent dans un encadré "Adressé à", "Facturé à" ou "Client".
- "total_ttc" : le montant total TTC, avec le symbole de la devise si présent.
- "tva" : le montant total de TVA, ou le taux si le montant n'est pas indiqué.
- Si une information est absente de la facture, mets null.
"""


# ============================================================
# Parsing JSON robuste
# ============================================================

def extract_json(text: str) -> dict:
    """Extrait un objet JSON d'une réponse LLM, même avec du bruit."""
    text = text.strip()

    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("Aucun JSON valide trouvé dans la réponse LLM")


# ============================================================
# Prédiction principale
# ============================================================

def predict_invoice(filename: str, file_bytes: bytes) -> dict:
    """
    Analyse une facture (PDF ou image) et retourne :
    {
        "pages_count": int,
        "pages": [
            {"page": 1, "data": {...}, "raw_response": "..."},
            ...
        ]
    }
    """
    images = load_cv_as_images(filename, file_bytes)
    print(f"📄 {len(images)} page(s) détectée(s)")

    pages_results = []

    for i, img in enumerate(images, start=1):
        print(f"🔍 Analyse page {i}/{len(images)}...")

        try:
            # Construction du message multimodal
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": PROMPT},
                ],
            }]

            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, _ = process_vision_info(messages)

            inputs = processor(
                text=[prompt],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )

            with torch.no_grad():
                generated_ids = llm.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,      # déterministe
                )

            # Trim le prompt d'entrée
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            raw = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            print(f"📥 Réponse brute page {i} : {raw[:200]}")

            data = extract_json(raw)

            # Nettoyage : forcer les clés attendues
            cleaned = {
                "date_facture": data.get("date_facture"),
                "fournisseur": data.get("fournisseur"),
                "client": data.get("client"),
                "total_ttc": data.get("total_ttc"),
                "tva": data.get("tva"),
            }

            pages_results.append({
                "page": i,
                "data": cleaned,
                "raw_response": raw,
                "error": None,
            })

        except Exception as e:
            print(f"⚠️ Erreur page {i} : {e}")
            pages_results.append({
                "page": i,
                "data": {
                    "date_facture": None,
                    "fournisseur": None,
                    "client": None,
                    "total_ttc": None,
                    "tva": None,
                },
                "raw_response": None,
                "error": str(e),
            })

    return {
        "pages_count": len(images),
        "pages": pages_results,
    }