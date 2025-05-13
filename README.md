# Qwen OCR Bot

[![GitHub license](https://img.shields.io/github/license/SakibAhmedShuva/Qwen-OCR-Bot)](https://github.com/SakibAhmedShuva/Qwen-OCR-Bot/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Qwen2.5--VL-yellow)](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

A powerful, self-hosted OCR (Optical Character Recognition) and image analysis application built on Unsloth's quantized Qwen2.5-VL vision-language model. This application provides both a user-friendly web interface and a backend API for extracting text from images and analyzing visual content.

![Qwen OCR Bot Demo](https://raw.githubusercontent.com/SakibAhmedShuva/Qwen-OCR-Bot/main/docs/demo-screenshot.png)

![image](https://github.com/user-attachments/assets/f15e6d14-104e-46a7-801a-02168022170c)

## Features

- **OCR Capabilities**: Extract text from documents, receipts, business cards, and more
- **Image Analysis**: Get detailed descriptions of image content
- **Streaming Responses**: Fast, token-by-token streaming interface for immediate feedback
- **Low Resource Requirements**: Runs efficiently on consumer hardware with 4-bit quantization
- **Fully Self-Hosted**: Complete privacy with no data sent to external APIs
- **Web UI & API**: Use via browser or integrate with your own applications
- **Session Management**: Save and retrieve conversation history
- **Multi-image Support**: Process multiple images in a single conversation

## Requirements

- Python 3.8 or higher
- CUDA-compatible GPU with at least 6GB VRAM (for GPU acceleration) or 16GB+ RAM for CPU-only mode
- Approximately 8GB disk space for the model weights

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/SakibAhmedShuva/Qwen-OCR-Bot.git
cd Qwen-OCR-Bot
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

The web interface will be available at http://localhost:5001

## Usage

### Web Interface

1. Open your browser and navigate to http://localhost:5001
2. Click "Create New Session" to start
3. Upload an image or paste an image URL
4. Type a prompt like "Extract all text from this image" or "What does this image show?"
5. View the streamed response from the model

### API Endpoints

The application provides the following API endpoints:

- `POST /create-session`: Creates a new conversation session
- `POST /chat`: Submit images and prompts to get OCR/analysis results
- `POST /clear-backend-history`: Clears conversation history for a session

Example curl request:

```bash
curl -X POST http://localhost:5001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your-session-id",
    "prompt": "Extract all text from this image",
    "model_id": "unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit",
    "image_data_url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
  }'
```

## Supported Models

The application is designed to work with:

- `unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit` (default, 4-bit quantized)
- `Qwen/Qwen2.5-VL-3B-Instruct` (original model)

Additional models can be added by updating the model selection dropdown in the UI and ensuring compatibility in the backend.

## Performance Considerations

- **GPU Mode**: For best performance, use a CUDA-capable GPU with at least 6GB VRAM.
- **CPU Mode**: Will run on CPU-only systems but significantly slower. Requires 16GB+ system RAM.
- **First Run**: The first query will be slower as the model needs to be loaded into memory.
- **Memory Usage**: The quantized model requires ~3-4GB of GPU memory.

## Customization

### System Prompts

You can customize system prompts to specialize the model for different tasks:

```
You are an expert OCR assistant that specializes in accurate text extraction from images. Focus on extracting all text with proper formatting.
```

### Temperature Settings

- Lower temperature (0.1-0.3): More deterministic results for accurate OCR
- Higher temperature (0.7-0.9): More creative descriptions for image analysis

## Troubleshooting

### Common Issues

1. **"CUDA out of memory"**: Your GPU doesn't have enough VRAM. Try using CPU mode or a smaller model.
2. **"NumPy is not available"**: Reinstall NumPy and bitsandbytes as described in the error message.
3. **Slow first response**: Normal as the model is loaded into memory on first use.
4. **Missing image data**: Check image format and size; larger images (>10MB) may cause issues.

## Project Structure

- `app.py`: Main Flask application and API endpoints
- `qwen_model_handler.py`: Model loading and text generation
- `qwen_vl_utils.py`: Utility functions for image processing
- `static/`: Frontend assets and JavaScript
- `templates/`: HTML templates for the web interface

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for their excellent quantization techniques
- [Qwen Team](https://github.com/QwenLM/Qwen-VL) for the Qwen2.5-VL model
- [HuggingFace](https://huggingface.co/) for model hosting and transformers library

---

Created by [Sakib Ahmed Shuva](https://github.com/SakibAhmedShuva)
