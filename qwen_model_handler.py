# my_ai_studio/qwen_model_handler.py
import torch
from transformers import AutoProcessor, TextIteratorStreamer
from transformers import Qwen2_5_VLForConditionalGeneration
from threading import Thread
import logging
import numpy # Explicitly import numpy at the top level for early check

from qwen_vl_utils import extract_pil_images_from_messages

logger = logging.getLogger(__name__)
try:
    logger.info(f"NumPy version imported by qwen_model_handler: {numpy.__version__}") 
except NameError:
    logger.error("NumPy failed to import in qwen_model_handler top level!")


loaded_models = {}

def get_model_and_processor(model_id):
    if model_id in loaded_models:
        logger.info(f"Returning cached model and processor for: {model_id}")
        return loaded_models[model_id]

    logger.info(f"Loading model and processor for: {model_id}")

    model_kwargs = {
        "trust_remote_code": True,
    }

    can_use_gpu = torch.cuda.is_available()
    model_is_bnb_quantized = "bnb" in model_id.lower() 

    if can_use_gpu:
        if model_is_bnb_quantized:
            logger.info(f"CUDA available and model {model_id} suggests BNB quantization. Attempting GPU 4-bit load.")
            model_kwargs["torch_dtype"] = "auto" # Let transformers decide, usually bfloat16/float16 for BNB
            model_kwargs["device_map"] = {"": 0} # Target GPU 0, accelerate handles offload if needed
            # The UserWarning previously indicated the model has its own quantization_config.
            # We rely on from_pretrained to use that internal config.
            logger.info(
                f"Configured for GPU 4-bit load with device_map={model_kwargs['device_map']}, "
                f"torch_dtype={model_kwargs['torch_dtype']}. Model's internal quantization_config will be used."
            )
        else: 
            logger.info(f"CUDA available. Loading non-BNB model {model_id} on GPU.")
            model_kwargs["device_map"] = "auto" 
            model_kwargs["torch_dtype"] = "auto" 
            logger.info(f"Configured for GPU load with device_map='auto', torch_dtype='auto'.")
    else: 
        logger.info(f"CUDA not available. Attempting to load {model_id} on CPU.")
        model_kwargs["device_map"] = "cpu"
        model_kwargs["torch_dtype"] = torch.float32
        if model_is_bnb_quantized:
            logger.warning(
                f"Model {model_id} suggests BNB quantization. "
                "BNB 4-bit is generally not supported or performant on CPU. Load may fail or be very slow."
            )

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            **model_kwargs
        )
        
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        loaded_models[model_id] = (model, processor)
        device_info = model.hf_device_map if hasattr(model, 'hf_device_map') and model.hf_device_map else model.device
        logger.info(f"Successfully loaded model and processor: {model_id} on device(s): {device_info}")
        return model, processor
    
    except RuntimeError as e:
        if "Numpy is not available" in str(e):
            logger.critical(
                "CRITICAL RuntimeError: 'Numpy is not available' from bitsandbytes. "
                "ACTION: \n"
                "1. Activate your Python environment. \n"
                "2. Reinstall NumPy and bitsandbytes: \n"
                "   pip uninstall numpy bitsandbytes -y \n"
                "   pip install numpy \n"
                "   pip install bitsandbytes \n"
                "3. If using GPU, ensure your CUDA toolkit and drivers are compatible with the installed bitsandbytes version. ",
                exc_info=True
            )
        elif "CUDA is required" in str(e) or "bitsandbytes" in str(e).lower():
            logger.warning(
                f"bitsandbytes/CUDA related RuntimeError during load attempt for {model_id}: {e}. "
                f"Attempting CPU fallback if not already on CPU."
            )
            if model_kwargs.get("device_map") != "cpu":
                cpu_fallback_kwargs = {
                    "trust_remote_code": True, "device_map": "cpu", "torch_dtype": torch.float32,
                }
                if model_is_bnb_quantized: logger.info("CPU fallback for a BNB model: quantization_config will be omitted.")
                try:
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **cpu_fallback_kwargs)
                    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                    loaded_models[model_id] = (model, processor)
                    logger.info(f"Successfully loaded model (fallback to CPU): {model_id} on device: {model.device}")
                    return model, processor
                except Exception as cpu_e:
                    logger.error(f"CPU fallback loading also failed for {model_id}: {cpu_e}", exc_info=True)
                    raise RuntimeError(f"GPU load failed: {e}. CPU fallback also failed: {cpu_e}") from cpu_e
            else: # Already on CPU, and it failed
                logger.error(f"Runtime error loading model {model_id} on CPU: {e}", exc_info=True)
        else: 
            logger.error(f"Unhandled RuntimeError loading model {model_id}: {e}", exc_info=True)
        raise e # Re-raise the caught runtime error

    except ValueError as ve: # Catch ValueErrors like the offloading one
        logger.error(f"ValueError during model loading for {model_id}: {ve}", exc_info=True)
        if "Some modules are dispatched on the CPU or the disk" in str(ve):
             logger.critical(
                "CRITICAL ValueError: The 'llm_int8_enable_fp32_cpu_offload=True' (or equivalent from model's internal config) "
                "is still not preventing this error when modules are offloaded. "
                "This could be due to the model's internal quantization_config not correctly supporting this, "
                "or a deeper issue in the libraries. "
                "Consider checking the model card for Unsloth for specific loading instructions or known issues with offloading."
             )
        raise ve
    except Exception as e: 
        logger.error(f"General error loading model {model_id}: {e}", exc_info=True)
        raise e


def generate_chat_response_stream(model_id, messages_for_model, temperature, max_new_tokens=2048):
    # Ensure numpy is importable at the start of this function too, for sanity check
    try:
        import numpy as np_check # Use a different alias to avoid conflict if numpy was already imported
        logger.debug(f"NumPy version in generate_chat_response_stream: {np_check.__version__}")
    except ImportError:
        logger.critical("NumPy cannot be imported in generate_chat_response_stream! This is a fundamental environment issue.")
        raise RuntimeError("NumPy not found during stream generation setup.")

    model, processor = get_model_and_processor(model_id) 

    pil_images = extract_pil_images_from_messages(messages_for_model)
    if pil_images:
        logger.info(f"Extracted {len(pil_images)} PIL image(s) for model input.")
    else:
        logger.info("No PIL images extracted for model input for this request.")

    try:
        text_prompt = processor.apply_chat_template(
            messages_for_model,
            tokenize=False,
            add_generation_prompt=True
        )
        logger.info(f"Generated text_prompt via apply_chat_template for {model_id}")
    except Exception as e:
        logger.error(f"Error in apply_chat_template for {model_id} with messages: {json.dumps(messages_for_model, indent=2)}. Error: {e}", exc_info=True)
        raise ValueError(f"Error preparing prompt with apply_chat_template: {e}. Check message format and image references.")

    try:
        # Determine the target device from the model's device map or device attribute
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            if isinstance(model.device, torch.device) and model.device.type != 'meta':
                 target_device = model.device
            else: 
                 target_device = next(model.parameters()).device if len(list(model.parameters())) > 0 else \
                                (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
                 logger.info(f"Model on mixed devices or meta. Inferred target_device for inputs: {target_device}")
        else: 
            target_device = model.device

        inputs = processor(
            text=[text_prompt], # Must be a list for batching, even if single prompt
            images=pil_images if pil_images else None, # Pass loaded PIL images
            return_tensors="pt",
            padding=True 
        ).to(target_device) 
        
        pixel_values_info = inputs.get('pixel_values') # PyTorch tensor or None
        image_info_str = f"Image tensor shape: {pixel_values_info.shape}, dtype: {pixel_values_info.dtype}" if pil_images and pixel_values_info is not None else "No image tensors"

        logger.info(f"Inputs processed for model {model_id} and sent to {target_device}. {image_info_str}")
    except Exception as e:
        logger.error(f"Error processing inputs with processor for {model_id}. Text: '{text_prompt[:200]}...', Images present: {bool(pil_images)}. Error: {e}", exc_info=True)
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
        if pad_token_id is None: 
            logger.error(f"CRITICAL: Both pad_token_id and eos_token_id are None for tokenizer of {model_id}.")
            # Attempt to find a common EOS token as a last resort, e.g., for Qwen
            try_eos_tokens = ["<|endoftext|>", "<|im_end|>"] # Common EOS tokens
            for token_str in try_eos_tokens:
                token_id = processor.tokenizer.convert_tokens_to_ids(token_str)
                if token_id != processor.tokenizer.unk_token_id:
                    pad_token_id = token_id
                    logger.warning(f"Tokenizer missing pad_token_id and eos_token_id. Using '{token_str}' ({pad_token_id}) as fallback pad_token_id.")
                    break
            if pad_token_id is None: # Still None
                 raise ValueError("Tokenizer missing pad_token_id and eos_token_id, and fallbacks failed.")
        else:
            logger.warning(f"pad_token_id was None, using eos_token_id: {pad_token_id}")


    eos_token_id_for_generation = processor.tokenizer.eos_token_id
    if isinstance(eos_token_id_for_generation, list):
        # For Qwen-VL, <|im_end|> (ID: 151645) or other specific end tokens might be preferred.
        im_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_token_id != processor.tokenizer.unk_token_id: # Check if token exists
             eos_token_id_for_generation = im_end_token_id
             logger.info(f"Using specific eos_token_id '<|im_end|>' ({eos_token_id_for_generation}) for generation.")
        else:
            # Fallback: use the first ID in the list if specific one not found or not applicable
            eos_token_id_for_generation = eos_token_id_for_generation[0]
            logger.warning(f"eos_token_id is a list, specific '<|im_end|>' not applicable/found, using the first one: {eos_token_id_for_generation}")
    
    if eos_token_id_for_generation is None: 
        logger.warning(f"eos_token_id_for_generation is None for {model_id}. Using pad_token_id ({pad_token_id}) as fallback for eos_token_id.")
        eos_token_id_for_generation = pad_token_id # Fallback to pad_token_id if EOS is truly missing
        if eos_token_id_for_generation is None: # Absolute last resort
            raise ValueError("eos_token_id_for_generation cannot be None and fallback failed.")


    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True if temperature > 0.01 else False,
        temperature=max(temperature, 0.01), # Clamp temperature
        top_p=0.8 if temperature > 0.01 else None, # Conditional top_p
        top_k=None, # Usually top_p is preferred
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id_for_generation
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    model_device_info = model.hf_device_map if hasattr(model, 'hf_device_map') and model.hf_device_map else model.device
    logger.info(f"Starting generation for model {model_id} with temp {temperature}, max_new_tokens {max_new_tokens} on device(s) {model_device_info}")
    
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