# Ensure necessary libraries are installed
# !pip install gradio torch transformers bitsandbytes accelerate sentencepiece Pillow

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import time
import os
import gradio as gr
import tempfile # For handling temporary image files

# --- Check for necessary libraries ---
try:
    import bitsandbytes
except ImportError:
    raise ImportError("bitsandbytes is required for 4-bit loading. Please install it: pip install bitsandbytes")
try:
    import accelerate
except ImportError:
    raise ImportError("accelerate is required for device_map='auto'. Please install it: pip install accelerate")

# --- qwen_vl_utils.py Placeholder ---
try:
    from qwen_vl_utils import process_vision_info
    print("Successfully imported process_vision_info from qwen_vl_utils.")
    QWEN_PROCESS_VISION_INFO_AVAILABLE = True
except ImportError:
    print("Warning: Could not import process_vision_info from qwen_vl_utils.")
    print("Using a placeholder function for process_vision_info.")
    QWEN_PROCESS_VISION_INFO_AVAILABLE = False

    def process_vision_info(messages):
        image_inputs_list = []
        video_inputs_list = []

        for msg in messages:
            if msg['role'] == 'user' and isinstance(msg['content'], list):
                for item in msg['content']:
                    if item['type'] == 'image':
                        image_data = item.get('image')
                        if isinstance(image_data, str) and os.path.exists(image_data):
                            try:
                                img = Image.open(image_data).convert('RGB')
                                image_inputs_list.append(img)
                                print(f"Placeholder: Loaded image from path {image_data}")
                            except Exception as e:
                                print(f"Placeholder: Error loading image from path {image_data}: {e}")
                        elif isinstance(image_data, Image.Image):
                            image_inputs_list.append(image_data.convert('RGB')) # Ensure RGB
                            print("Placeholder: Received PIL image directly.")
                        else:
                            print(f"Placeholder: Image data not found, not a valid path, or not a PIL image: {image_data}")
        if not image_inputs_list:
            print("Placeholder: No images were processed by process_vision_info.")
        return image_inputs_list, video_inputs_list

# --- Configuration ---
MODEL_ID = "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit"

# --- Model and Processor Loading ---
print(f"Loading model: {MODEL_ID} with 4-bit quantization...")
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print(f"Using compute dtype: {compute_dtype}")

try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        load_in_4bit=True,
        device_map="auto",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    print("Model loaded.")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("Processor loaded.")
    MODEL_LOADED_SUCCESSFULLY = True
    TARGET_DEVICE = next(model.parameters()).device
    print(f"Model is on device: {TARGET_DEVICE}")

except Exception as e:
    print(f"Error during model or processor loading: {e}")
    MODEL_LOADED_SUCCESSFULLY = False
    model = None
    processor = None
    TARGET_DEVICE = torch.device("cpu")

# --- System Prompts ---
SYSTEM_PROMPT_OCR = """Extract ALL text, numbers, in all sides. You MUST ONLY output the extracted text. Do NOT add any introductory phrases, explanations, summaries, confidence scores, meta-comments, or any text whatsoever that is not directly read from the image.
Be extremely thorough and do not miss any textual element, no matter how small, faint, or seemingly insignificant.
"""

SYSTEM_PROMPT_CHAT = """You are a helpful multimodal assistant. You have been provided with an image and its OCR-extracted text.
Answer the user's questions based on the visual information in the image and the provided text.
If the question is about the text, refer to the OCR text. If it's about visual elements not in text, describe the image.
Be concise and helpful.
"""

# --- Core OCR Logic ---
def _perform_ocr_core(image_pil_for_ocr):
    if not MODEL_LOADED_SUCCESSFULLY:
        return "Error: Model could not be loaded."
    if image_pil_for_ocr is None:
        return "Error: No image provided for OCR."

    temp_image_path_ocr = None
    try:
        # Forcing temp file usage as the real qwen_vl_utils might expect paths
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_ocr:
            image_pil_for_ocr.convert('RGB').save(tmp_ocr, format="JPEG")
            temp_image_path_ocr = tmp_ocr.name
        print(f"OCR: Temporary image saved to: {temp_image_path_ocr}")

        messages_ocr = [
            {"role": "system", "content": SYSTEM_PROMPT_OCR},
            {"role": "user", "content": [
                {"type": "image", "image": temp_image_path_ocr},
                {"type": "text", "text": "Perform a complete and verbatim Optical Character Recognition (OCR) of this entire document image. Extract every single word, number, symbol, and punctuation mark. Maintain the original layout, including line breaks and spacing, as accurately as possible."},
            ]},
        ]

        image_inputs_ocr, _ = process_vision_info(messages_ocr)
        if not image_inputs_ocr:
            msg = "Error: process_vision_info (OCR) did not return any image data."
            if not QWEN_PROCESS_VISION_INFO_AVAILABLE: msg += " (Using placeholder process_vision_info)"
            return msg

        text_ocr_prompt = processor.apply_chat_template(
            messages_ocr, tokenize=False, add_generation_prompt=True
        )
        inputs_ocr = processor(
            text=[text_ocr_prompt], images=image_inputs_ocr, padding=True, return_tensors="pt"
        ).to(TARGET_DEVICE)

        print("OCR: Starting text generation...")
        start_time = time.perf_counter()
        generated_ids_ocr = model.generate(
            **inputs_ocr,
            max_new_tokens=4096,
            do_sample=False, # Crucial for OCR
            temperature=0.01,
            top_k=1,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id
        )
        generated_ids_trimmed_ocr = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_ocr.input_ids, generated_ids_ocr)
        ]
        output_text_raw = processor.batch_decode(
            generated_ids_trimmed_ocr, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        end_time = time.perf_counter()
        print(f"OCR: Time taken: {end_time - start_time:.4f} seconds")
        return output_text_raw

    except Exception as e:
        print(f"An error occurred during OCR core processing: {e}")
        import traceback
        traceback.print_exc()
        return f"An error occurred during OCR: {str(e)}"
    finally:
        if temp_image_path_ocr and os.path.exists(temp_image_path_ocr):
            os.remove(temp_image_path_ocr)
            print(f"OCR: Temporary image {temp_image_path_ocr} deleted.")

# --- Gradio UI Functions ---
def process_uploaded_image_for_ui(uploaded_image_pil):
    if not MODEL_LOADED_SUCCESSFULLY:
        error_msg = "Model not loaded. Cannot perform OCR."
        return error_msg, None, None, [(None, error_msg)], "Chat disabled: Model not loaded."

    if uploaded_image_pil is None:
        return "Please upload an image first.", None, None, [], "Chat disabled: Upload an image and perform OCR."

    print("UI: Performing OCR...")
    ocr_text_result = _perform_ocr_core(uploaded_image_pil)

    if "Error:" in ocr_text_result or not ocr_text_result.strip():
        err_msg = ocr_text_result if "Error:" in ocr_text_result else "OCR resulted in empty text or an error."
        print(f"UI: OCR Error or empty result: {err_msg}")
        # Still store the image if it was uploaded, but OCR text is None
        return err_msg, uploaded_image_pil, None, [(None, err_msg)], "Chat disabled: OCR failed or no text found."

    initial_chat_history = [(None, "OCR complete. Extracted text is shown. You can now ask questions about the image or its content.")]
    print("UI: OCR successful.")
    return ocr_text_result, uploaded_image_pil, ocr_text_result, initial_chat_history, "Ask a question about the image or text..."

def handle_chat_message_for_ui(user_query, chat_history, image_pil_from_state, ocr_text_from_state):
    if not MODEL_LOADED_SUCCESSFULLY:
        chat_history.append((user_query, "Model not loaded. Cannot process chat."))
        return chat_history, ""

    if image_pil_from_state is None:
        chat_history.append((user_query, "No image has been processed yet. Please upload an image and perform OCR first."))
        return chat_history, ""
    if ocr_text_from_state is None: # OCR might have failed, but image is present
        chat_history.append((user_query, "OCR was not successful or yielded no text. Chatting might be limited to visual aspects of the image if available."))
        # Fallback to allow asking about image even if OCR failed.
        # For robust handling, we'd construct a prompt without ocr_text_from_state or with a note about its absence.
        # For now, let's proceed, but the prompt will include "None" or empty OCR text.

    print(f"UI Chat: Query: '{user_query}'")
    chat_history.append((user_query, None)) # Show user query immediately

    temp_image_path_chat = None
    try:
        # Forcing temp file usage for process_vision_info consistency
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_chat:
            image_pil_from_state.convert('RGB').save(tmp_chat, format="JPEG")
            temp_image_path_chat = tmp_chat.name
        print(f"Chat: Temporary image for chat saved to: {temp_image_path_chat}")

        user_content_list_chat = [
            {"type": "image", "image": temp_image_path_chat},
            {"type": "text", "text": f"Contextual OCR Text (if available):\n---\n{ocr_text_from_state or 'No text extracted or OCR failed.'}\n---\nBased on the image and the provided text, answer this question: {user_query}"}
        ]

        # Simple history: just the last exchange for context, or build full history.
        # For this example, each chat turn is mostly independent but uses the full context.
        messages_for_chat = [{"role": "system", "content": SYSTEM_PROMPT_CHAT}]

        # Add previous turns from chat_history to messages_for_chat for better context
        # This can get long, so typically summarize or take last N turns.
        # For now, keep it simple: current query with full image/OCR context.
        # If you want to build history:
        # for human, ai in chat_history_for_model: # chat_history_for_model is chat_history[:-1]
        #     messages_for_chat.append({"role": "user", "content": human})
        #     if ai: messages_for_chat.append({"role": "assistant", "content": ai})

        messages_for_chat.append({"role": "user", "content": user_content_list_chat})


        image_inputs_chat, _ = process_vision_info(messages_for_chat)
        if not image_inputs_chat:
            msg = "Error: process_vision_info (Chat) did not return any image data."
            if not QWEN_PROCESS_VISION_INFO_AVAILABLE: msg += " (Using placeholder process_vision_info)"
            chat_history[-1] = (user_query, msg)
            return chat_history, ""

        text_chat_prompt = processor.apply_chat_template(
            messages_for_chat, tokenize=False, add_generation_prompt=True
        )
        inputs_chat = processor(
            text=[text_chat_prompt], images=image_inputs_chat, padding=True, return_tensors="pt"
        ).to(TARGET_DEVICE)

        print("Chat: Starting model generation for chat...")
        start_time = time.perf_counter()
        generated_ids_chat = model.generate(
            **inputs_chat,
            max_new_tokens=1024, # Shorter for chat
            do_sample=True,     # More creative for chat
            temperature=0.6,
            top_p=0.9,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id
        )
        generated_ids_trimmed_chat = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_chat.input_ids, generated_ids_chat)
        ]
        model_response = processor.batch_decode(
            generated_ids_trimmed_chat, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        end_time = time.perf_counter()
        print(f"Chat: Time taken: {end_time - start_time:.4f} seconds")

        chat_history[-1] = (user_query, model_response) # Update placeholder with actual response

    except Exception as e:
        print(f"An error occurred during chat processing: {e}")
        import traceback
        traceback.print_exc()
        chat_history[-1] = (user_query, f"An error occurred during chat: {str(e)}")
    finally:
        if temp_image_path_chat and os.path.exists(temp_image_path_chat):
            os.remove(temp_image_path_chat)
            print(f"Chat: Temporary image {temp_image_path_chat} deleted.")

    return chat_history, "" # Clear chat input textbox

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Qwen2.5-VL: OCR & Chat")
    gr.Markdown(
        "Upload an image, perform OCR, and then chat with the model about the image and its extracted text. "
        "Model: `unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit`"
    )

    # State variables to store image and OCR text between interactions
    stored_image_pil = gr.State()
    stored_ocr_text = gr.State()

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="1. Upload Image")
            ocr_button = gr.Button("2. Perform OCR", variant="primary")
            gr.Markdown("### Extracted Text (OCR)")
            ocr_output_textbox = gr.Textbox(lines=15, label="OCR Output", interactive=False, placeholder="OCR results will appear here...")

        with gr.Column(scale=2):
            gr.Markdown("### 3. Chat about the Image/Text")
            chatbot_display = gr.Chatbot(label="Conversation", height=500, bubble_full_width=False)
            chat_input_textbox = gr.Textbox(
                label="Your Question:",
                placeholder="Chat disabled until OCR is performed successfully.",
                interactive=True # Will be enabled/disabled via logic if needed
            )
            chat_send_button = gr.Button("Send Message", variant="primary")

    # OCR button action
    ocr_button.click(
        fn=process_uploaded_image_for_ui,
        inputs=[image_input],
        outputs=[ocr_output_textbox, stored_image_pil, stored_ocr_text, chatbot_display, chat_input_textbox] # Update chat input placeholder too
    )

    # Chat submission (button or enter key)
    chat_params = {
        "fn": handle_chat_message_for_ui,
        "inputs": [chat_input_textbox, chatbot_display, stored_image_pil, stored_ocr_text],
        "outputs": [chatbot_display, chat_input_textbox] # Update chatbot, clear chat input
    }
    chat_send_button.click(**chat_params)
    chat_input_textbox.submit(**chat_params)

    if not MODEL_LOADED_SUCCESSFULLY:
        gr.Error("Critical Error: Model and/or Processor failed to load. The application will not function correctly. Please check the console logs.")

# --- Launching the Gradio App ---
if __name__ == '__main__':
    if MODEL_LOADED_SUCCESSFULLY:
        print("Attempting to launch Gradio interface...")
        try:
            demo.launch(inline=True, debug=True, share=True)
            print("Gradio app running. Check the output for the URL (usually http://127.0.0.1:7860 or 7861).")
        except Exception as e:
            print(f"Could not launch Gradio interface automatically: {e}")
            print("You can try running `demo.launch()` manually in a new cell if in a notebook.")
    else:
        print("Gradio interface not launched due to model loading failure. Please check console output for errors.")

# For Jupyter Notebook, you might run this in a cell:
# if MODEL_LOADED_SUCCESSFULLY:
#     demo.launch(inline=True, debug=True)
# else:
#     print("Model not loaded, Gradio interface cannot be started.")