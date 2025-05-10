# my_ai_studio/qwen_model_handler.py
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, TextIteratorStreamer
# For Qwen2.5-VL, the specific class might be Qwen2_5_VLForConditionalGeneration
# Using AutoModelForCausalLM should generally work if the config is set up correctly for VL models.
# Let's try with the specific one if AutoModelForCausalLM gives issues.
from transformers import Qwen2_5_VLForConditionalGeneration

from threading import Thread
import logging

# Local import (make sure qwen_vl_utils.py is in the same directory or accessible)
from qwen_vl_utils import process_vision_info


logger = logging.getLogger(__name__)

# Cache for loaded models and processors
loaded_models = {}

def get_model_and_processor(model_id):
    if model_id in loaded_models:
        return loaded_models[model_id]

    logger.info(f"Loading model and processor for: {model_id}")
    try:
        # For Unsloth 4-bit models, bnb_4bit_compute_dtype=torch.float16 is often recommended
        # and device_map="auto" for multi-GPU or efficient single-GPU use.
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto", # or torch.bfloat16 / torch.float16
            device_map="auto",
            # For Unsloth with BitsAndBytes, trust_remote_code might be needed by some models
            # trust_remote_code=True, # Uncomment if required by the model
            # attn_implementation="flash_attention_2" # If supported and beneficial
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) # Qwen often needs trust_remote_code for processor
        
        loaded_models[model_id] = (model, processor)
        logger.info(f"Successfully loaded: {model_id}")
        return model, processor
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {e}", exc_info=True)
        raise

def generate_chat_response_stream(model_id, messages_for_model, temperature, max_new_tokens=2048):
    """
    Generates a chat response from the Qwen model using streaming.

    Args:
        model_id (str): The Hugging Face model ID.
        messages_for_model (list): The conversation history formatted for the Qwen model.
                                   Each message is a dict with "role" and "content".
                                   Content for "user" role should be a list: [{"type": "text", "text": "..."}]
        temperature (float): The generation temperature.
        max_new_tokens (int): Maximum number of new tokens to generate.

    Yields:
        str: Chunks of the generated text.
    """
    model, processor = get_model_and_processor(model_id)

    # `apply_chat_template` is the standard way to prepare inputs for chat.
    # It handles the roles and special tokens.
    # For Qwen-VL, `messages_for_model` should already be in the expected list-of-dicts format.
    try:
        text_prompt = processor.apply_chat_template(
            messages_for_model,
            tokenize=False,
            add_generation_prompt=True # Important for instruct/chat models
        )
    except Exception as e:
        logger.error(f"Error in apply_chat_template for {model_id} with messages: {messages_for_model}. Error: {e}", exc_info=True)
        # Try to provide more context on the error if possible.
        # Common issue: `messages_for_model` not in the expected format.
        # Qwen expects: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]
        # For Qwen-VL specifically, user/assistant content might be a list of dicts: [{"type":"text", "text":"..."}]
        # If an image were involved: [{"type":"image", "image":"path_or_url"}, {"type":"text", "text":"..."}]
        raise ValueError(f"Error preparing prompt with apply_chat_template: {e}. Check message format.")


    # The `process_vision_info` is for when images are explicitly part of the `messages_for_model`
    # and need to be extracted and passed to `processor(images=...)`.
    # For text-only from the frontend, or if images are URLs within the text content
    # that `apply_chat_template` handles, `image_inputs_processed` might be None.
    image_inputs_processed, _ = process_vision_info(messages_for_model)


    # Prepare inputs for the model
    # Note: If `image_inputs_processed` is not None, it means we have actual image data
    # (e.g., PIL Images or tensors) to pass. If it's None, and images were URLs in text,
    # `apply_chat_template` might have handled them by embedding in `text_prompt`.
    inputs = processor(
        text=[text_prompt], # text_prompt should be a list of strings if batching
        images=image_inputs_processed, # Pass processed images if any
        return_tensors="pt",
        padding=True
    ).to(model.device)

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    # Ensure pad_token_id is set, Qwen often uses eos_token_id for padding
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True if temperature > 0.01 else False, # do_sample=False for temp ~0
        temperature=max(temperature, 0.01), # Temperature must be > 0 for sampling
        top_k=50 if temperature > 0.01 else 1,
        pad_token_id=pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id # Model might have multiple EOS tokens
    )
    
    # Some models might have a list of EOS tokens. Pick one, or let the model default.
    # For Qwen, <|endoftext|> is common. processor.tokenizer.eos_token might be it.
    if isinstance(processor.tokenizer.eos_token_id, list):
         generation_kwargs["eos_token_id"] = processor.tokenizer.eos_token_id[0]


    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    logger.info(f"Starting generation for model {model_id} with temp {temperature}")
    buffer = ""
    for new_text_chunk in streamer:
        # logger.debug(f"Streamer yielded: '{new_text_chunk}'") # Can be very verbose
        if new_text_chunk:
            buffer += new_text_chunk
            # Yield complete words or after a short pause to avoid too many tiny chunks
            # This is a simple heuristic; more complex buffering might be needed
            if ' ' in buffer or '\n' in buffer or len(buffer) > 10:
                yield buffer
                buffer = ""
    if buffer: # Yield any remaining part
        yield buffer
    logger.info(f"Finished generation for model {model_id}")