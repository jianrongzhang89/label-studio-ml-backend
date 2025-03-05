import argparse
import base64
import io
from typing import Dict, Union

import torch
from transformers import pipeline
from kserve import InferRequest, InferResponse, Model, ModelServer, model_server
from kserve.errors import InvalidInput
from kserve import logging as kserve_logging

logger = kserve_logging.logger

class TransformersModel(Model):
    def __init__(self, name: str):
        super().__init__(name)
        # print('In __init__')

        self.model_id = args.model_id or "/mnt/models"
        self.lora_dir = args.lora_dir or None
        self.pipeline = None
        self.ready = False

        self.load()

    def load(self):

        # Load the model
        print(f' -> Loading model ({self.model_id}) using from_pretrained')
        self.pipeline = pipeline('text-generation', model=self.model_id)

        if args.device:
            print(f" -> Using {args.device} device")
            if args.device == "cuda":
                self.pipeline.to(torch.device("cuda"))
            elif args.device == "cpu":
                self.pipeline.to(torch.device("cpu"))
            elif args.device == "enable_model_cpu_offload":
                self.pipeline.enable_model_cpu_offload()
            elif args.device == "enable_sequential_cpu_offload":
                self.pipeline.enable_sequential_cpu_offload()
            else:
                raise ValueError(f"Invalid device: {args.device}")
        else:
            print(' -> Using "CUDA" device')
            self.pipeline.to(torch.device("cuda"))

        if self.lora_dir:
            print(f" -> Trying to load LoRA weights for this model from {self.lora_dir}")
            self.pipeline.load_lora_weights(self.lora_dir)
            print(f" -> Loaded LoRA weights for this model from {self.lora_dir}")

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

    def convert_lists_to_tuples(self, data):
        if isinstance(data, dict):
            return {k: self.convert_lists_to_tuples(v) for k, v in data.items()}
        elif isinstance(data, list):
            return tuple(self.convert_lists_to_tuples(v) for v in data)
        else:
            return data

    def predict(
        self, payload: Union[Dict, InferRequest], headers: Dict[str, str] = None
    ) -> Union[Dict, InferResponse]:

        logger.info(f'Payload for generating the answers -> {payload}')
        payload = self.convert_lists_to_tuples(payload)

        # Create the answers, without refiner if not needed
        generators = self.pipeline(**payload)
        
        return {
            "predictions": [
                {
                    "model_name": self.model_id,
                    "text": payload["text"],
                    "generated_text": generators[0].text,
                }
            ]
        }


parser = argparse.ArgumentParser(parents=[model_server.parser])

parser.add_argument(
    "--model_id",
    type=str,
    help="Model ID to load (default: /mnt/models, adapt if you use the refiner model)",
)
parser.add_argument(
    "--lora_dir",
    type=str,
    help="LoRA weights will be loaded from this directory (no default value)",
)
parser.add_argument(
    "--device",
    type=str,
    help="Device to use, including offloading. Valid values are: 'cuda' (default), 'enable_model_cpu_offload', 'enable_sequential_cpu_offload', 'cpu' (works but unusable...)",
)
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    print(f' -> Creating an instance of DiffusersModel - modelName=[{args.model_name}]...')
    model = TransformersModel(args.model_name)
    # model.load()      # model is loaded from init

    print(' -> Calling start() on ModelServer...')
    ModelServer().start([model])
