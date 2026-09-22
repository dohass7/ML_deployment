import os

# Désactive le GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import io

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware

#from fastapi.staticfiles import StaticFiles

#from fastapi.responses import FileResponse

from PIL import Image

from model import (
    predict_emotion,
    check_status
)

from pillow_heif import register_heif_opener

from datetime import datetime

from database import analyses

register_heif_opener()

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="FaceRead Emotion API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

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

        status = (
            f"unreachable ({str(e)})"
        )

    try:

        analyses.count_documents({})

        mongo_status = "connected"

    except Exception as e:

        mongo_status = (
            f"unreachable ({str(e)})"
        )

    return {
        "status": "healthy",

        "message":
            "Backend service is running",

        "model_status":
            status,

        "mongodb":
            mongo_status
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Vérification du fichier
    # --------------------------------------------------------

    if not file.content_type:

        raise HTTPException(
            status_code=400,
            detail="Type de fichier inconnu."
        )

    if not file.content_type.startswith(
        "image/"
    ):

        raise HTTPException(
            status_code=400,
            detail="Le fichier doit être une image."
        )

    # --------------------------------------------------------
    # Lecture
    # --------------------------------------------------------

    try:
        print("FICHIER :", file.filename)
        print("CONTENT TYPE :", file.content_type)

        contents = await file.read()

        if not contents:

            raise HTTPException(
                status_code=400,
                detail="Image vide."
            )

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        print("FORMAT IMAGE :", image.format)
        print("TAILLE IMAGE :", image.size)
        print("MODE IMAGE :", image.mode)

        # Force le chargement de l'image
        image.load()

        # ----------------------------------------------------
        # Prédiction
        # ----------------------------------------------------

        result = predict_emotion(
            image
        )
        try:
            analyses.insert_one({
                "filename": file.filename,
                "upload_date": datetime.utcnow(),
                "faces_count": len(result["faces"]),
                "faces": result["faces"],
            })
        except Exception as e:
            print("MONGO INSERT ERROR :", str(e))
        # Debug backend
        print(
            "RESULTAT ENVOYÉ AU FRONTEND :",
            {
                "faces": result["faces"],
                "image": (
                    f"<base64 {len(result['image'])} caractères>"
                    if result["image"] is not None
                    else None
                )
            }
        )

        return result

    except HTTPException:
        raise

    except Exception as e:

        print(
            "ERREUR /predict :",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Erreur pendant "
                "l'analyse de l'image : "
                f"{str(e)}"
            )
        )