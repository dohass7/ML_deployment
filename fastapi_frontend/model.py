import json
import numpy as np
import boto3

# Configuration SageMaker
APP_NAME = 'face-emotion-deploy'
REGION = 'eu-north-1'

# Labels des émotions
LABELS = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise']


def check_status():
    """Vérifie le statut de l'endpoint SageMaker."""
    sage_client = boto3.client('sagemaker', region_name=REGION)
    endpoint_description = sage_client.describe_endpoint(EndpointName=APP_NAME)
    return endpoint_description["EndpointStatus"]


def query_endpoint(image_array):
    """Envoie une image à l'endpoint SageMaker et retourne les prédictions."""
    client = boto3.session.Session().client("sagemaker-runtime", REGION)

    image_array_np = np.array(image_array)
    input_json = json.dumps({"instances": image_array_np.tolist()})

    response = client.invoke_endpoint(
        EndpointName=APP_NAME,
        Body=input_json,
        ContentType='application/json',
    )
    preds = response['Body'].read().decode("ascii")
    preds = json.loads(preds)
    return preds


def predict_emotion(face_array):
    """
    Prédit l'émotion via l'endpoint SageMaker.
    face_array : tableau numpy prétraité (batch, 32, 32, 1)
    """
    preds = query_endpoint(face_array)

    # Gestion des formats de réponse MLflow / SageMaker
    if isinstance(preds, dict):
        # Format {"predictions": [[...]]}
        predictions = preds.get("predictions", preds.get("outputs", preds))
    else:
        predictions = preds

    # Convertir en numpy pour argmax
    predictions_np = np.array(predictions)
    if predictions_np.ndim > 1:
        emotion_class = predictions_np[0].argmax()
    else:
        emotion_class = predictions_np.argmax()

    return LABELS[int(emotion_class)]
