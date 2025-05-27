import argparse
import base64
import io
from typing import Dict, Union
import torch
from transformers import pipeline

# checkpoint = "/opt/app-root/src/label-studio-ml-backend/label_studio_ml/examples/huggingface_ner/rhoai-poc/distilbert-finetuned-ner-1/checkpoint-204"
# token_classifier = pipeline("token-classification", model=checkpoint, aggregation_strategy="simple")

from kserve import InferRequest, InferResponse, Model, ModelServer, model_server
from kserve.errors import InvalidInput
from kserve import logging as kserve_logging

logger = kserve_logging.logger

class NERModel(Model):
    def __init__(self, name: str):
        super().__init__(name)
        # print('In __init__')

        self.model_checkpoint = args.model_checkpoint or "/mnt/models"
        self.pipeline = None
        self.ready = False
        self.load()

    def load(self):

        # Load the model
        print(f' -> Loading model ({self.model_checkpoint}) using from_pretrained')
        device = 0 if torch.cuda.is_available() else -1
        self.pipeline = pipeline(
            "token-classification",
            model=self.model_checkpoint,
            device=device,
            aggregation_strategy="simple"
        )

        # The ready flag is used by model ready endpoint for readiness probes,
        # set to True when model is loaded successfully without exceptions.
        self.ready = True

    def preprocess(
        self, payload: Union[Dict, InferRequest], headers: Dict[str, str] = None
    ) -> Dict:
        if isinstance(payload, Dict) and "instances" in payload:
            logger.info('setting request-type to "v1"...')
            headers["request-type"] = "v1"
        elif isinstance(payload, InferRequest):
            raise InvalidInput("v2 protocol not implemented")
        else:
            raise InvalidInput("invalid payload")

        return payload["instances"][0]

    def predict(
        self, payload: Union[Dict, InferRequest], headers: Dict[str, str] = None
    ) -> Union[Dict, InferResponse]:

        logger.info(f'Payload for generating predictions-> {payload}')

        # Generate repdictions
        return self.pipeline(**payload)

parser = argparse.ArgumentParser(parents=[model_server.parser])

parser.add_argument(
    "--model_checkpoint",
    type=str,
    help="Model checkpoiny to load (default: /mnt/models, adapt if you use the refiner model)",
)

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    print(f' -> Creating an instance of NERModel - modelName=[{args.model_name}]...')
    model = NERModel(args.model_name)
    # model.load()      # model is loaded from init

    print(' -> Calling start() on ModelServer...')
    ModelServer().start([model])
