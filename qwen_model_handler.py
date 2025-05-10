# my_ai_studio/qwen_model_handler.py
import torch
from transformers import AutoProcessor, TextIteratorStreamer
from transformers import Qwen2_5_VLForConditionalGeneration # Qwen2ForConditionalGeneration is for text only
from threading import Thread
import logging

from qwen_vl_utils import extract_pil_images_from_messages # Ensure qwen_vl_utils.py is present

logger = logging.getLogger(__name__)

loaded_models = {}

def get_model_and_processor(model_id):
    if model_id in loaded_models:
        logger.info(f"Returning cached model and processor for: {model_id}")
        return loaded_models[model_id]

    logger.info(f"Loading model and processor for: {model_id}")
    try:
        # For Qwen-VL models, ensure the correct class is used
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True, # Crucial for many Hugging Face models, especially custom ones
        )
        # Qwen often needs trust_remote_code for processor as well
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        loaded_models[model_id] = (model, processor)
        logger.info(f"Successfully loaded model and processor: {model_id}")
        return model, processor
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {e}", exc_info=True)
        raise

def generate_chat_response_stream(model_id, messages_for_model, temperature, max_new_tokens=2048):
    model, processor = get_model_and_processor(model_id)

    # 1. Extract PIL images from messages
    # This should happen BEFORE apply_chat_template if the template needs to know about images
    # or if images are passed separately to the processor.
    # For Qwen-VL, `apply_chat_template` usually expects messages to contain image references
    # (like `image_url` dicts) and it generates text with placeholders (e.g. `<img>`).
    pil_images = extract_pil_images_from_messages(messages_for_model)
    if pil_images:
        logger.info(f"Extracted {len(pil_images)} image(s) for processing.")
    else:
        logger.info("No images found or extracted for this request.")


    # 2. Apply chat template
    # The `messages_for_model` should be structured so that `apply_chat_template`
    # can correctly create a prompt that accounts for text and images.
    # For Qwen-VL, this often means `messages_for_model` contains `{"type": "image_url", ...}`
    try:
        text_prompt = processor.apply_chat_template(
            messages_for_model,
            tokenize=False,
            add_generation_prompt=True # Important for instruct/chat models
        )
        logger.info(f"Generated text_prompt via apply_chat_template for {model_id}")
    except Exception as e:
        logger.error(f"Error in apply_chat_template for {model_id} with messages: {messages_for_model}. Error: {e}", exc_info=True)
        raise ValueError(f"Error preparing prompt with apply_chat_template: {e}. Check message format and image references.")

    # 3. Process inputs (text and images)
    # The `images` argument takes the PIL images loaded earlier.
    # The `text_prompt` from `apply_chat_template` should contain necessary placeholders if images are used.
    try:
        inputs = processor(
            text=[text_prompt], # `text` should be a list of strings
            images=pil_images if pil_images else None, # Pass loaded PIL images
            return_tensors="pt",
            padding=True # Consider `padding="longest"` or specific strategy if issues arise
        ).to(model.device)
        logger.info(f"Inputs processed for model {model_id}. Image tensor shape: {inputs.get('pixel_values').shape if pil_images and 'pixel_values' in inputs else 'No images'}")
    except Exception as e:
        logger.error(f"Error processing inputs with processor for {model_id}. Text: '{text_prompt}', Images present: {bool(pil_images)}. Error: {e}", exc_info=True)
        raise ValueError(f"Error in processor call: {e}. Check text prompt and image data.")


    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id
        logger.warning(f"pad_token_id was None, using eos_token_id: {pad_token_id}")
    
    # Ensure eos_token_id is a single ID, not a list, for generation_kwargs
    eos_token_id_for_generation = processor.tokenizer.eos_token_id
    if isinstance(eos_token_id_for_generation, list):
        # For Qwen-VL, <|im_end|> (ID: 151645) is often a primary stop token.
        # Or just pick the first one. Check tokenizer_config.json or model card.
        # Let's find a common one or default to the first if not obvious.
        im_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_token_id != processor.tokenizer.unk_token_id: # Check if token exists
             eos_token_id_for_generation = im_end_token_id
             logger.info(f"Using specific eos_token_id '<|im_end|>' ({eos_token_id_for_generation}) for generation.")
        else:
            eos_token_id_for_generation = eos_token_id_for_generation[0]
            logger.warning(f"eos_token_id is a list, using the first one: {eos_token_id_for_generation}")


    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True if temperature > 0.01 else False,
        temperature=max(temperature, 0.01), # Clamp temperature to avoid issues with 0.0
        top_p=0.8 if temperature > 0.01 else None, # Conditional top_p
        top_k=None, # Usually top_p is preferred over top_k for diverse sampling
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id_for_generation
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    logger.info(f"Starting generation for model {model_id} with temp {temperature}, max_new_tokens {max_new_tokens}")
    buffer = ""
    for new_text_chunk in streamer:
        if new_text_chunk:
            buffer += new_text_chunk
            # Yield more frequently for better perceived responsiveness
            if ' ' in buffer or '\n' in buffer or len(buffer) > 5: # Reduced buffer length
                yield buffer
                buffer = ""
    if buffer: # Yield any remaining buffer content
        yield buffer
    logger.info(f"Finished generation stream for model {model_id}")