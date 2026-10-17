import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from datetime import datetime

from model import predict_invoice, check_status
from database import collection
from schemas import (
    PredictResponse,
    PageResult,
    InvoiceData,
    AnalysisDocument,
)

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="InvoiceRead API",
    version="1.0.0",
    description="Extraction d'informations sur factures / tickets (LLM vision)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health_check():
    try:
        status = check_status()
    except Exception as e:
        status = f"unreachable ({str(e)})"

    try:
        collection.count_documents({})
        mongo_status = "connected"
    except Exception as e:
        mongo_status = f"unreachable ({str(e)})"

    return {
        "status": "healthy",
        "message": "Backend service is running",
        "model_status": status,
        "mongodb": mongo_status,
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):

    # -------- Validation type fichier --------
    if not file.content_type:
        raise HTTPException(400, "Type de fichier inconnu.")

    allowed_types = (
        "image/",
        "application/pdf",
    )
    if not any(file.content_type.startswith(t) for t in allowed_types):
        raise HTTPException(
            400,
            "Le fichier doit être une image ou un PDF.",
        )

    # -------- Lecture --------
    try:
        print("FICHIER :", file.filename)
        print("CONTENT TYPE :", file.content_type)

        contents = await file.read()
        if not contents:
            raise HTTPException(400, "Fichier vide.")

        # -------- Prédiction --------
        result = predict_invoice(file.filename, contents)

        # -------- Persistance Mongo --------
        try:
            doc = AnalysisDocument(
                filename=file.filename,
                upload_date=datetime.utcnow(),
                pages_count=result["pages_count"],
                pages=[
                    PageResult(**p) for p in result["pages"]
                ],
            )
            collection.insert_one(
                doc.model_dump()
            )
        except Exception as e:
            print("MONGO INSERT ERROR :", str(e))

        # -------- Réponse --------
        return PredictResponse(
            filename=file.filename,
            pages_count=result["pages_count"],
            pages=[
                PageResult(**p) for p in result["pages"]
            ],
        )

    except HTTPException:
        raise
    except Exception as e:
        print("ERREUR /predict :", str(e))
        raise HTTPException(
            500,
            f"Erreur pendant l'analyse du document : {str(e)}",
        )