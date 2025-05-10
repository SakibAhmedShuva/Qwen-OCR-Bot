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
    logger.info(f"Finished generation for model {model_id}")# my_ai_studio/app.py
import os
import uuid
import json
import time
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename
import logging

# Local imports
from qwen_model_handler import generate_chat_response_stream, get_model_and_processor, loaded_models

# --- Flask App Setup ---
app = Flask(__name__)
app.secret_key = os.urandom(24) # For session management if Flask sessions are used

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- In-memory store for session histories ---
# session_histories = {
#    "session_id_1": {
#        "messages": [
#            {"role": "system", "content": "You are helpful."},
#            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
#            {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]}
#        ],
#        "last_activity": timestamp,
#        "model_id": "currently_used_model_for_this_session" # Optional
#    }
# }
session_histories = {}
MAX_HISTORY_LENGTH = 20 # Max number of messages (user + assistant pairs) to keep

# --- Helper Functions ---
def get_session_history(session_id):
    if session_id not in session_histories:
        session_histories[session_id] = {"messages": [], "last_activity": time.time()}
    session_histories[session_id]["last_activity"] = time.time()
    return session_histories[session_id]["messages"]

def add_to_session_history(session_id, role, text_content):
    history = get_session_history(session_id)
    # Ensure content is in Qwen-VL's list-of-dicts format for user/assistant
    qwen_content_format = [{"type": "text", "text": text_content}]
    
    if role == "system": # System prompt is a simple string
         history.append({"role": role, "content": text_content})
    else:
         history.append({"role": role, "content": qwen_content_format})

    # Trim history if it gets too long
    if len(history) > MAX_HISTORY_LENGTH * 2: # roughly pairs
        # Keep system prompt if present, then trim older messages
        new_history = []
        system_prompts = [m for m in history if m['role'] == 'system']
        if system_prompts:
            new_history.extend(system_prompts) # Keep all system prompts or just the latest? For now, all.
        
        # Keep the last MAX_HISTORY_LENGTH user/assistant messages
        non_system_messages = [m for m in history if m['role'] != 'system']
        new_history.extend(non_system_messages[-MAX_HISTORY_LENGTH*2:])
        session_histories[session_id]["messages"] = new_history


def format_messages_for_qwen(session_id, system_prompt_override, user_prompt_text):
    """
    Prepares the list of messages in the format Qwen expects,
    including the system prompt and the new user prompt.
    """
    current_history = get_session_history(session_id).copy() # Work with a copy
    messages_for_model = []

    # Handle system prompt
    # If override is provided, it takes precedence.
    # If not, check if a system prompt already exists in history.
    # If neither, a default one can be added.
    has_system_prompt_in_history = any(m['role'] == 'system' for m in current_history)

    if system_prompt_override:
        messages_for_model.append({"role": "system", "content": system_prompt_override})
        # Remove any existing system prompts from history if override is used
        current_history = [m for m in current_history if m['role'] != 'system']
    elif has_system_prompt_in_history:
        # If no override, but system prompt in history, it will be added from current_history
        pass
    else:
        # Default system prompt if no override and none in history
        messages_for_model.append({"role": "system", "content": "You are a helpful AI assistant."})

    # Add historical messages
    for msg in current_history:
        # Ensure content format consistency if needed, though add_to_session_history should handle it
        messages_for_model.append(msg)
    
    # Add current user prompt
    messages_for_model.append({
        "role": "user",
        "content": [{"type": "text", "text": user_prompt_text}]
    })
    return messages_for_model

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/create-session', methods=['POST'])
def create_session():
    session_id = str(uuid.uuid4())
    session_histories[session_id] = {"messages": [], "last_activity": time.time()}
    logger.info(f"Created new backend session: {session_id}")
    return jsonify({"status": "success", "session_id": session_id})

@app.route('/clear-backend-history', methods=['POST'])
def clear_backend_history():
    data = request.json
    session_id = data.get('session_id')
    if session_id and session_id in session_histories:
        session_histories[session_id]["messages"] = [] # Keep session, clear messages
        logger.info(f"Cleared backend history for session: {session_id}")
        return jsonify({"status": "success", "message": "Backend session memory (history) cleared."})
    elif session_id:
        logger.warning(f"Attempt to clear history for non-existent session: {session_id}")
        return jsonify({"status": "error", "message": "Session not found."}), 404
    else:
        return jsonify({"status": "error", "message": "Session ID not provided."}), 400

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        session_id = data.get('session_id')
        user_prompt = data.get('prompt')
        system_prompt = data.get('system_prompt', '').strip() # User-defined system prompt
        temperature = float(data.get('temperature', 0.7))
        model_id = data.get('model_id')

        if not all([session_id, user_prompt, model_id]):
            return jsonify({"error": "Missing required parameters (session_id, prompt, model_id)."}), 400

        logger.info(f"Chat request for session {session_id}, model {model_id}")
        add_to_session_history(session_id, 'user', user_prompt)
        
        messages_for_model = format_messages_for_qwen(session_id, system_prompt, user_prompt)

        def generate_sse_stream():
            full_ai_response = ""
            try:
                for chunk in generate_chat_response_stream(model_id, messages_for_model, temperature):
                    if chunk: # Ensure chunk is not empty
                        full_ai_response += chunk
                        # SSE format: data: {"text_chunk": "...", "is_final": false}\n\n
                        sse_data = {"text_chunk": chunk, "is_final": False}
                        yield f"data: {json.dumps(sse_data)}\n\n"
                
                # After stream is finished
                if full_ai_response:
                     add_to_session_history(session_id, 'assistant', full_ai_response)
                
                # Send final message marker
                final_data = {"text_chunk": "", "full_response": full_ai_response, "is_final": True}
                yield f"data: {json.dumps(final_data)}\n\n"
                logger.info(f"Stream finished for session {session_id}. Full response length: {len(full_ai_response)}")

            except Exception as e:
                logger.error(f"Error during SSE generation for session {session_id}: {e}", exc_info=True)
                error_data = {"error": f"Model generation error: {str(e)}", "is_final": True}
                yield f"data: {json.dumps(error_data)}\n\n"

        return Response(stream_with_context(generate_sse_stream()), mimetype='text/event-stream')

    except Exception as e:
        logger.error(f"Error in /chat endpoint: {e}", exc_info=True)
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# --- Pre-load default model (optional, can speed up first request) ---
# This is illustrative. You might want to load models on first use in get_model_and_processor
# Or have a separate admin endpoint to trigger loading.
def preload_models():
    logger.info("Preloading default models...")
    default_model_id_from_frontend = "unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit" # Match JS
    try:
        get_model_and_processor(default_model_id_from_frontend)
        # You could preload other models too if desired
        # get_model_and_processor("unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit")
        logger.info("Default model(s) preloaded (or loading initiated).")
    except Exception as e:
        logger.error(f"Failed to preload default model {default_model_id_from_frontend}: {e}", exc_info=True)


if __name__ == '__main__':
    # preload_models() # Uncomment to preload models on startup
    port = int(os.environ.get("PORT", 5000))
    # Consider using a more production-ready WSGI server like Gunicorn for deployment
    # For development:
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
    # use_reloader=False is often good when dealing with large models to avoid reloading them on code changes.
    # For production, Gunicorn/uWSGI would handle workers.