# my_ai_studio/app.py

import os
import uuid
import json
import time
import re
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import logging

# Local imports
from qwen_model_handler import generate_chat_response_stream, get_model_and_processor

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app) 

app.secret_key = os.urandom(24) 

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

session_histories = {}
MAX_HISTORY_LENGTH = 20 

def get_session_history(session_id):
    if session_id not in session_histories:
        session_histories[session_id] = {"messages": [], "last_activity": time.time()}
    session_histories[session_id]["last_activity"] = time.time()
    return session_histories[session_id]["messages"]

def add_to_session_history(session_id, role, content_data):
    history = get_session_history(session_id)
    # Ensure content_data for user/assistant is always a list of content blocks
    if role == "system":
        history.append({"role": role, "content": content_data}) # System prompt is a string
    else:
        if isinstance(content_data, list): # Already formatted as list of content blocks
            history.append({"role": role, "content": content_data})
        elif isinstance(content_data, str): # Simple text string
            history.append({"role": role, "content": [{"type": "text", "text": content_data}]})
        else:
            logger.error(f"Unsupported content_data type for role {role}: {type(content_data)}. Expected list or str.")
            return

    if len(history) > MAX_HISTORY_LENGTH * 2: # System prompts + user/assistant pairs
        new_history = []
        system_prompts = [m for m in history if m['role'] == 'system']
        if system_prompts:
            new_history.extend(system_prompts)
        
        non_system_messages = [m for m in history if m['role'] != 'system']
        new_history.extend(non_system_messages[-MAX_HISTORY_LENGTH*2:])
        session_histories[session_id]["messages"] = new_history
        logger.info(f"History for session {session_id} truncated.")


def format_messages_for_qwen(session_id, system_prompt_override, user_prompt_text, uploaded_image_data_url=None):
    current_history = get_session_history(session_id).copy()
    messages_for_model = []
    
    # Handle System Prompt
    has_system_prompt_in_history = any(m['role'] == 'system' for m in current_history)
    if system_prompt_override:
        # If a new system prompt is provided, it replaces any existing ones for this call
        messages_for_model.append({"role": "system", "content": system_prompt_override})
        # Remove old system prompts from current_history before extending
        current_history = [m for m in current_history if m['role'] != 'system']
    elif has_system_prompt_in_history:
        messages_for_model.extend([m for m in current_history if m['role'] == 'system'])
        current_history = [m for m in current_history if m['role'] != 'system'] # Remove system prompts after adding them
    else:
        # Default system prompt if none exists and none overridden
        messages_for_model.append({"role": "system", "content": "You are a helpful AI assistant."})

    # Add existing user/assistant messages from history
    messages_for_model.extend(current_history) 
    
    # Construct current user message content
    user_content_list = []
    
    # Prioritize explicitly uploaded image
    if uploaded_image_data_url:
        user_content_list.append({"type": "image_url", "image_url": {"url": uploaded_image_data_url}})
        logger.info(f"Using explicitly uploaded image data URL for Qwen-VL.")
        # The user_prompt_text can then be caption/instruction for this image
        if user_prompt_text and user_prompt_text.strip():
            user_content_list.append({"type": "text", "text": user_prompt_text.strip()})
    else:
        # Fallback: try to find an image URL in the text prompt
        # Regex updated to better handle various image extensions and data URLs
        url_pattern = r'(https?://[^\s/$.?#].[^\s]*\.(?:jpg|jpeg|png|gif|webp|avif))|(data:image/[a-zA-Z]+;base64,[A-Za-z0-9+/=]+)'
        
        parts = []
        last_end = 0
        for match in re.finditer(url_pattern, user_prompt_text, re.IGNORECASE):
            # Text before the URL
            if match.start() > last_end:
                stripped_pre_text = user_prompt_text[last_end:match.start()].strip()
                if stripped_pre_text:
                    parts.append({"type": "text", "text": stripped_pre_text})
            
            # The URL itself (either http or data)
            url = match.group(0) # Full match
            user_content_list.append({"type": "image_url", "image_url": {"url": url}})
            logger.info(f"Image URL found in text and formatted for Qwen-VL: {url[:60]}...")
            last_end = match.end()

        # Text after the last URL, or the whole prompt if no URL
        remaining_text = user_prompt_text[last_end:].strip()
        if remaining_text:
            parts.append({"type": "text", "text": remaining_text})
        
        # Consolidate text parts if they were split by URL parsing
        # and add them to user_content_list if they aren't already there (as image URLs)
        for p in parts:
            if p["type"] == "text" and p["text"]:
                 # Avoid adding duplicate text if it was the entire prompt and no URL was found
                 # or if it's already somehow in user_content_list
                is_duplicate = any(
                    existing_item["type"] == "text" and existing_item["text"] == p["text"] 
                    for existing_item in user_content_list
                )
                if not is_duplicate:
                    user_content_list.append(p)


        # If after all that, user_content_list is empty but user_prompt_text exists, add it as text.
        if not user_content_list and user_prompt_text and user_prompt_text.strip():
            user_content_list.append({"type": "text", "text": user_prompt_text.strip()})
        elif not user_content_list and (not user_prompt_text or not user_prompt_text.strip()):
            # Edge case: empty prompt, no uploaded image.
            user_content_list.append({"type": "text", "text": "Describe the image."}) # Default if truly empty
            logger.warning("User prompt was empty and no image provided. Added default text.")


    messages_for_model.append({
        "role": "user",
        "content": user_content_list
    })
    
    logger.debug(f"Formatted messages for Qwen: {json.dumps(messages_for_model, indent=2)}")
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
        current_messages = session_histories[session_id]["messages"]
        # Retain only system prompts, clear user/assistant messages
        system_prompts = [m for m in current_messages if m['role'] == 'system']
        session_histories[session_id]["messages"] = system_prompts
        logger.info(f"Cleared user/assistant messages from backend history for session: {session_id}. System prompts retained.")
        return jsonify({"status": "success", "message": "Backend session memory (user/assistant messages) cleared."})
    elif session_id:
        logger.warning(f"Attempt to clear history for non-existent session: {session_id}")
        return jsonify({"status": "error", "message": "Session not found."}), 404
    else:
        return jsonify({"status": "error", "message": "Session ID not provided."}), 400

@app.route('/chat', methods=['POST'])
def chat():
    request_start_time = time.monotonic() # Start timing the request
    try:
        data = request.json
        session_id = data.get('session_id')
        user_prompt_text = data.get('prompt', '') # Can be empty if image is primary
        system_prompt_override = data.get('system_prompt', '').strip()
        temperature = float(data.get('temperature', 0.7))
        model_id = data.get('model_id')
        image_data_url = data.get('image_data_url') # New field for base64 image

        if not session_id or not model_id: # Prompt or image_data_url must exist
            logger.error(f"Missing required parameters: session_id={session_id}, model_id={model_id}")
            return jsonify({"error": "Missing required parameters (session_id, model_id)."}), 400
        
        if not user_prompt_text and not image_data_url:
            logger.error(f"Missing content: Both prompt text and image_data_url are empty for session {session_id}.")
            return jsonify({"error": "Prompt text or an image upload is required."}), 400


        logger.info(f"Chat request for session {session_id}, model {model_id}. User prompt: '{user_prompt_text[:100]}...'. Image uploaded: {bool(image_data_url)}")
        
        messages_for_model = format_messages_for_qwen(session_id, system_prompt_override, user_prompt_text, image_data_url)
        
        if messages_for_model and messages_for_model[-1]["role"] == "user":
             add_to_session_history(session_id, 'user', messages_for_model[-1]["content"])
        else:
             logger.error("Failed to correctly format user message for history. Last model message was not user.")
             add_to_session_history(session_id, 'user', [{"type": "text", "text": user_prompt_text or "Image interaction"}])


        @stream_with_context
        def generate_sse_stream_with_timing():
            sse_setup_start_time = time.monotonic()
            full_ai_response_text = ""
            response_metadata = {"first_token_time_ms": -1, "full_generation_time_ms": -1}
            first_token_received = False
            generation_start_time = 0 

            try:
                for chunk in generate_chat_response_stream(model_id, messages_for_model, temperature):
                    if not first_token_received:
                        generation_start_time = time.monotonic() 
                        # Time from sse_setup_start_time to first token includes model call setup, 
                        # processor, and actual first token latency from model.
                        response_metadata["first_token_time_ms"] = round((generation_start_time - sse_setup_start_time) * 1000)
                        first_token_received = True

                    if chunk:
                        full_ai_response_text += chunk
                        sse_data = {"text_chunk": chunk, "is_final": False}
                        yield f"data: {json.dumps(sse_data)}\n\n"
                
                if full_ai_response_text:
                    add_to_session_history(session_id, 'assistant', full_ai_response_text)
                
                if first_token_received: 
                    response_metadata["full_generation_time_ms"] = round((time.monotonic() - generation_start_time) * 1000)
                else: 
                    # If no tokens, full_generation_time is effectively the time spent trying.
                    response_metadata["full_generation_time_ms"] = round((time.monotonic() - sse_setup_start_time) * 1000)
                    if response_metadata["first_token_time_ms"] == -1: # Ensure TTFT is also set if no tokens
                         response_metadata["first_token_time_ms"] = response_metadata["full_generation_time_ms"]


                final_data = {
                    "text_chunk": "", 
                    "full_response": full_ai_response_text, 
                    "is_final": True,
                    "response_time_stats": response_metadata 
                }
                yield f"data: {json.dumps(final_data)}\n\n"
                
                stream_total_duration_ms = round((time.monotonic() - sse_setup_start_time) * 1000)
                logger.info(
                    f"Stream finished for session {session_id}. Response length: {len(full_ai_response_text)}. "
                    f"TTFT: {response_metadata['first_token_time_ms']}ms, "
                    f"Gen time: {response_metadata['full_generation_time_ms']}ms. "
                    f"Total stream handler time: {stream_total_duration_ms}ms."
                )

            except ValueError as ve: 
                logger.error(f"ValueError during SSE gen for session {session_id}: {ve}", exc_info=False)
                error_data = {"error": f"Model input error: {str(ve)}", "is_final": True}
                yield f"data: {json.dumps(error_data)}\n\n"
            except Exception as e:
                logger.error(f"Error during SSE gen for session {session_id}: {e}", exc_info=True)
                error_data = {"error": f"Model generation error: {str(e)}", "is_final": True}
                yield f"data: {json.dumps(error_data)}\n\n"
        
        response = Response(generate_sse_stream_with_timing(), mimetype='text/event-stream')
        request_duration_ms = round((time.monotonic() - request_start_time) * 1000)
        logger.info(f"/chat request setup for session {session_id} completed in {request_duration_ms}ms. Streaming response initiated.")
        return response

    except Exception as e:
        request_duration_ms = round((time.monotonic() - request_start_time) * 1000)
        logger.error(f"Error in /chat endpoint after {request_duration_ms}ms: {e}", exc_info=True)
        return jsonify({"error": f"Server error: {str(e)}"}), 500

def preload_models(): # Optional: Preload default model on startup
    logger.info("Attempting to preload default model...")
    default_model_id = "unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit" # Or your preferred default
    try:
        get_model_and_processor(default_model_id)
        logger.info(f"Default model {default_model_id} preloading initiated/completed.")
    except Exception as e:
        logger.error(f"Failed to preload default model {default_model_id}: {e}", exc_info=True)


if __name__ == '__main__':
    # preload_models() # Uncomment if you want to preload on start
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)