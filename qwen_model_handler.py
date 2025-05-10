# my_ai_studio/qwen_model_handler.py
import torch
from transformers import AutoProcessor, TextIteratorStreamer
from transformers import Qwen2_5_VLForConditionalGeneration
from threading import Thread
import logging

from qwen_vl_utils import extract_pil_images_from_messages

logger = logging.getLogger(__name__)

loaded_models = {}

# Helper to check CUDA availability for bitsandbytes (simplified)
def is_bnb_cuda_available():
    try:
        # A more robust check might involve trying a small bnb operation
        # For now, rely on torch's CUDA check as a proxy, though bnb has its own compiled status
        if not torch.cuda.is_available():
            return False
        # Further, bnb might be installed without GPU support even if torch sees CUDA
        # This is hard to check without attempting a bnb-specific GPU operation
        # For now, let's assume if torch sees CUDA, we can try GPU,
        # and if bnb then complains, the from_pretrained will error out.
        return True # Placeholder, true check is more complex
    except Exception:
        return False

def get_model_and_processor(model_id):
    if model_id in loaded_models:
        logger.info(f"Returning cached model and processor for: {model_id}")
        return loaded_models[model_id]

    logger.info(f"Loading model and processor for: {model_id}")

    # Determine device and quantization strategy
    device = "cpu"
    load_in_4bit_gpu = False
    load_in_8bit_gpu = False # You could add an 8-bit option too
    torch_dtype = torch.float32 # Default for CPU

    can_try_gpu = torch.cuda.is_available()
    model_is_bnb_quantized = "bnb" in model_id.lower() # Heuristic

    if can_try_gpu and model_is_bnb_quantized:
        logger.info("CUDA is available and model name suggests BNB quantization. Attempting GPU 4-bit load.")
        # We will let from_pretrained try with device_map="auto" and implicit 4-bit
        # If bitsandbytes isn't compiled for CUDA, it will raise an error during from_pretrained
        device = "auto" # Let transformers decide, likely cuda
        torch_dtype = "auto" # Or torch.bfloat16 / torch.float16 for GPU
        load_in_4bit_gpu = True # We'll pass this to from_pretrained if needed for explicit control
    elif model_is_bnb_quantized:
        logger.warning("CUDA not available, but model name suggests BNB quantization. "
                       "Attempting to load on CPU. This might be slow or unsupported for 4-bit.")
        # For CPU, we typically don't use 4-bit bnb. We might need to disable it.
        # If the model config *forces* 4-bit, this might still fail.
    else: # Not a bnb model or no CUDA
        logger.info(f"CUDA not available or model not explicitly BNB. Loading on CPU with {torch_dtype}.")


    try:
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
        }

        if device == "auto" and load_in_4bit_gpu:
            model_kwargs["device_map"] = "auto"
            # For "unsloth/...-bnb-4bit" models, the config.json likely has quantization_config.
            # `from_pretrained` should pick this up.
            # Explicitly setting load_in_4bit=True can sometimes be helpful or required.
            # model_kwargs["load_in_4bit"] = True # Try this if auto doesn't work
            logger.info(f"Attempting to load {model_id} with device_map='auto' (expecting GPU 4-bit).")
        else: # CPU
            model_kwargs["device_map"] = "cpu"
            logger.info(f"Attempting to load {model_id} on CPU with dtype {torch_dtype}.")
            # If it's a BNB model and we're on CPU, we might need to tell it NOT to load in 4-bit
            # if the config would otherwise try to. This is tricky as the config might be fixed.
            # model_kwargs["quantization_config"] = None # May or may not work
            # model_kwargs["load_in_4bit"] = False # More direct

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            **model_kwargs
        )
        
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        loaded_models[model_id] = (model, processor)
        logger.info(f"Successfully loaded model and processor: {model_id} on device: {model.device}")
        return model, processor
    except RuntimeError as e:
        if "CUDA is required but not available for bitsandbytes" in str(e) and device != "cpu":
            logger.warning(f"GPU load failed due to bitsandbytes CUDA issue: {e}. Retrying on CPU...")
            try:
                # Force CPU load, and try to disable 4-bit if that's the issue
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True,
                    # load_in_4bit=False, # Explicitly try to disable 4-bit for CPU
                                            # This argument might not exist or apply if quantization
                                            # is hardcoded in the model's config.json
                )
                processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                loaded_models[model_id] = (model, processor)
                logger.info(f"Successfully loaded model (fallback to CPU): {model_id} on device: {model.device}")
                return model, processor
            except Exception as cpu_e:
                logger.error(f"CPU fallback loading also failed for {model_id}: {cpu_e}", exc_info=True)
                raise cpu_e
        else:
            logger.error(f"Error loading model {model_id}: {e}", exc_info=True)
            raise e
    except Exception as e: # Catch other general errors
        logger.error(f"General error loading model {model_id}: {e}", exc_info=True)
        raise e


def generate_chat_response_stream(model_id, messages_for_model, temperature, max_new_tokens=2048):
    model, processor = get_model_and_processor(model_id) # This now handles device logic

    pil_images = extract_pil_images_from_messages(messages_for_model)
    if pil_images:
        logger.info(f"Extracted {len(pil_images)} image(s) for processing.")
    else:
        logger.info("No images found or extracted for this request.")

    try:
        text_prompt = processor.apply_chat_template(
            messages_for_model,
            tokenize=False,
            add_generation_prompt=True
        )
        logger.info(f"Generated text_prompt via apply_chat_template for {model_id}")
    except Exception as e:
        logger.error(f"Error in apply_chat_template for {model_id} with messages: {messages_for_model}. Error: {e}", exc_info=True)
        raise ValueError(f"Error preparing prompt with apply_chat_template: {e}. Check message format and image references.")

    try:
        # Inputs should be sent to the same device the model is on.
        # model.device will tell us where the model ended up.
        target_device = model.device
        inputs = processor(
            text=[text_prompt],
            images=pil_images if pil_images else None,
            return_tensors="pt",
            padding=True
        ).to(target_device) # Send inputs to the model's device
        logger.info(f"Inputs processed for model {model_id} and sent to {target_device}. Image tensor shape: {inputs.get('pixel_values').shape if pil_images and 'pixel_values' in inputs else 'No images'}")
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

    eos_token_id_for_generation = processor.tokenizer.eos_token_id
    if isinstance(eos_token_id_for_generation, list):
        im_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_token_id != processor.tokenizer.unk_token_id:
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
        temperature=max(temperature, 0.01),
        top_p=0.8 if temperature > 0.01 else None,
        top_k=None,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id_for_generation
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    logger.info(f"Starting generation for model {model_id} with temp {temperature}, max_new_tokens {max_new_tokens} on device {model.device}")
    buffer = ""
    for new_text_chunk in streamer:
        if new_text_chunk:
            buffer += new_text_chunk
            if ' ' in buffer or '\n' in buffer or len(buffer) > 5:
                yield buffer
                buffer = ""
    if buffer:
        yield buffer
    logger.info(f"Finished generation stream for model {model_id}")