import os
import uuid
import json
import time
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename # This was missing from your app.py but is not used, so it's fine.
import logging

# Local imports
# This line is CRUCIAL and assumes qwen_model_handler.py is a separate, correct file.
from qwen_model_handler import generate_chat_response_stream, get_model_and_processor 
# loaded_models is not directly used by app.py but by qwen_model_handler, so importing it here isn't strictly necessary for app.py

# --- Flask App Setup ---
app = Flask(__name__)
app.secret_key = os.urandom(24) # For session management if Flask sessions are used

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- In-memory store for session histories ---
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
    qwen_content_format = [{"type": "text", "text": text_content}]
    
    if role == "system":
         history.append({"role": role, "content": text_content})
    else:
         history.append({"role": role, "content": qwen_content_format})

    if len(history) > MAX_HISTORY_LENGTH * 2: 
        new_history = []
        system_prompts = [m for m in history if m['role'] == 'system']
        if system_prompts:
            new_history.extend(system_prompts)
        
        non_system_messages = [m for m in history if m['role'] != 'system']
        new_history.extend(non_system_messages[-MAX_HISTORY_LENGTH*2:])
        session_histories[session_id]["messages"] = new_history


def format_messages_for_qwen(session_id, system_prompt_override, user_prompt_text):
    current_history = get_session_history(session_id).copy() 
    messages_for_model = []
    has_system_prompt_in_history = any(m['role'] == 'system' for m in current_history)

    if system_prompt_override:
        messages_for_model.append({"role": "system", "content": system_prompt_override})
        current_history = [m for m in current_history if m['role'] != 'system']
    elif has_system_prompt_in_history:
        pass
    else:
        messages_for_model.append({"role": "system", "content": "You are a helpful AI assistant."})

    for msg in current_history:
        messages_for_model.append(msg)
    
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
        session_histories[session_id]["messages"] = [] 
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
        system_prompt = data.get('system_prompt', '').strip() 
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
                    if chunk: 
                        full_ai_response += chunk
                        sse_data = {"text_chunk": chunk, "is_final": False}
                        yield f"data: {json.dumps(sse_data)}\n\n"
                
                if full_ai_response:
                     add_to_session_history(session_id, 'assistant', full_ai_response)
                
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

def preload_models():
    logger.info("Preloading default models...")
    default_model_id_from_frontend = "unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit"
    try:
        get_model_and_processor(default_model_id_from_frontend)
        logger.info("Default model(s) preloaded (or loading initiated).")
    except Exception as e:
        logger.error(f"Failed to preload default model {default_model_id_from_frontend}: {e}", exc_info=True)


if __name__ == '__main__':
    # preload_models() # It's good to keep this commented out during initial debugging
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)