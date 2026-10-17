from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# ============================================================
# SCHEMA D'EXTRACTION (ce que le LLM doit renvoyer)
# ============================================================

class InvoiceData(BaseModel):
    """Données extraites d'une facture/ticket."""

    date_facture: Optional[str] = Field(
        default=None,
        description="Date d'émission au format JJ/MM/AAAA"
    )
    fournisseur: Optional[str] = Field(
        default=None,
        description="Nom de l'entreprise émettrice"
    )
    client: Optional[str] = Field(
        default=None,
        description="Nom du destinataire de la facture"
    )
    total_ttc: Optional[str] = Field(
        default=None,
        description="Montant total TTC avec devise"
    )
    tva: Optional[str] = Field(
        default=None,
        description="Montant ou taux de TVA"
    )


class PageResult(BaseModel):
    """Résultat par page (une facture peut avoir plusieurs pages)."""

    page: int
    data: InvoiceData
    raw_response: Optional[str] = None   # debug
    error: Optional[str] = None


# ============================================================
# SCHEMA DE RÉPONSE API
# ============================================================

class PredictResponse(BaseModel):
    """Réponse globale de /predict."""

    filename: str
    pages_count: int
    pages: List[PageResult]


# ============================================================
# SCHEMA MONGO (document stocké)
# ============================================================

class AnalysisDocument(BaseModel):
    """Document inséré dans MongoDB."""

    filename: str
    upload_date: datetime
    pages_count: int
    pages: List[PageResult]
    status: str = "success"