# my_ai_studio/qwen_model_handler.py
import torch
from transformers import AutoProcessor, TextIteratorStreamer # AutoModelForCausalLM removed as not used
from transformers import Qwen2_5_VLForConditionalGeneration
from threading import Thread
import logging

from qwen_vl_utils import process_vision_info # Ensure qwen_vl_utils.py is present

logger = logging.getLogger(__name__)

loaded_models = {}

def get_model_and_processor(model_id):
    if model_id in loaded_models:
        return loaded_models[model_id]

    logger.info(f"Loading model and processor for: {model_id}")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto", 
            device_map="auto",
            # trust_remote_code=True, # Add if model requires
        )
        # Qwen often needs trust_remote_code for processor
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) 
        
        loaded_models[model_id] = (model, processor)
        logger.info(f"Successfully loaded: {model_id}")
        return model, processor
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {e}", exc_info=True)
        raise

def generate_chat_response_stream(model_id, messages_for_model, temperature, max_new_tokens=2048):
    model, processor = get_model_and_processor(model_id)

    try:
        text_prompt = processor.apply_chat_template(
            messages_for_model,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        logger.error(f"Error in apply_chat_template for {model_id} with messages: {messages_for_model}. Error: {e}", exc_info=True)
        raise ValueError(f"Error preparing prompt with apply_chat_template: {e}. Check message format.")

    image_inputs_processed, _ = process_vision_info(messages_for_model)

    inputs = processor(
        text=[text_prompt], 
        images=image_inputs_processed,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True if temperature > 0.01 else False,
        temperature=max(temperature, 0.01), 
        top_k=50 if temperature > 0.01 else 1,
        pad_token_id=pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id
    )
    
    if isinstance(processor.tokenizer.eos_token_id, list):
         generation_kwargs["eos_token_id"] = processor.tokenizer.eos_token_id[0]

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    logger.info(f"Starting generation for model {model_id} with temp {temperature}")
    buffer = ""
    for new_text_chunk in streamer:
        if new_text_chunk:
            buffer += new_text_chunk
            if ' ' in buffer or '\n' in buffer or len(buffer) > 10:
                yield buffer
                buffer = ""
    if buffer:
        yield buffer
    logger.info(f"Finished generation for model {model_id}")