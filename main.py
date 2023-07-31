from asyncio import threads
import os
import torch
import time
import json
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

from fastapi import Body, FastAPI, Request
from parallelformers import parallelize
from typing import Any, Dict, AnyStr, List, Union
import ast
import logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)


def load_models():
    global model_dict
    model_name_or_path = "TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ"
    model_basename = "Wizard-Vicuna-7B-Uncensored-GPTQ-4bit-128g.no-act.order"
    print('\tLoading model: Wizard-Vicuna-7B-Uncensored-GPTQ')

    use_triton = False

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
                                               model_basename=model_basename,
                                               use_safetensors=True,
                                               trust_remote_code=True,
                                               device="cuda:0",
                                               use_triton=use_triton,
                                               quantize_config=None)
    model_dict['gptq_model'] = model
    model_dict['gptq_tokenizer'] = tokenizer
    return model_dict





def extractArgumentsFromJson(jsonString):
    jsonData = jsonString["data"]
    print(type(jsonData))
    return jsonData


async def GenerateTextByPayload(request):
    global model_dict
    payload = request.json()

    start_time = time.time()

    model = model_dict['gptq_model']
    tokenizer = model_dict['gptq_tokenizer']

    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

    payloadArguments = extractArgumentsFromJson(payload)
    output = generator(**payloadArguments)

    end_time = time.time()

    full_output = output
    output = output[0]['generated_text']
    result = {'inference_time': end_time - start_time,
              'result': output,
              'full_output': full_output
              }
    return result


def TestGenerateTextByPayload(payload):
    payloadArguments = extractArgumentsFromJson(payload)
    return payloadArguments


model_dict = load_models()

app = FastAPI(title="Inference Threaded API",
              description="fast Inference endpoint",
              version="1.0.0", )

JSONObject = Dict[AnyStr, Any]
JSONArray = List[Any]
JSONStructure = Union[JSONArray, JSONObject]


@app.get("/")
async def root():
    return {"message": "run /generate?text=&tokens=2048"}


@app.post("/inference")
async def inference(request: Request):
    return await GenerateTextByPayload(request)


if __name__ == "__main__":
    uvicorn.run("main:app", port=9000, reload=True)
