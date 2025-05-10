# my_ai_studio/qwen_vl_utils.py
import logging
import re
from PIL import Image
import requests
from io import BytesIO

logger = logging.getLogger(__name__)

def _load_image_from_url(url):
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        # Ensure image is in RGB format, as some models expect this
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download image from {url}: {e}")
    except IOError as e:
        logger.error(f"Failed to open image from {url} (possibly corrupted or not an image): {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading image {url}: {e}")
    return None

def extract_pil_images_from_messages(messages):
    """
    Extracts image URLs from the 'image_url' content type in messages,
    loads them as PIL Images.

    Args:
        messages (list): A list of message dictionaries.
                         Each message with an image should have a content list,
                         e.g., [{"type": "image_url", "image_url": {"url": "http://..."}}]

    Returns:
        list: A list of PIL.Image objects. Returns empty list if no valid images found.
    """
    pil_images = []
    if not messages:
        return pil_images

    for message in messages:
        if message.get("role") == "user" and isinstance(message.get("content"), list):
            for content_item in message["content"]:
                if content_item.get("type") == "image_url":
                    image_url_data = content_item.get("image_url")
                    if isinstance(image_url_data, dict) and "url" in image_url_data:
                        url = image_url_data["url"]
                        # Basic check for common image extensions or data URI
                        if re.match(r'^https?://.*\.(jpeg|jpg|png|webp|gif)(\?.*)?$', url, re.IGNORECASE) or \
                           url.startswith('data:image'):
                            img = _load_image_from_url(url)
                            if img:
                                pil_images.append(img)
                        else:
                            logger.warning(f"URL {url} does not look like a direct image URL. Skipping.")
                    elif isinstance(image_url_data, str): # Support direct URL string for simplicity
                        url = image_url_data
                        if re.match(r'^https?://.*\.(jpeg|jpg|png|webp|gif)(\?.*)?$', url, re.IGNORECASE) or \
                           url.startswith('data:image'):
                            img = _load_image_from_url(url)
                            if img:
                                pil_images.append(img)
                        else:
                            logger.warning(f"URL {url} does not look like a direct image URL. Skipping.")


    # If no `image_url` types are found, try a fallback for simple text URLs in the last user message.
    # This is less robust but can catch cases where the frontend didn't format perfectly.
    if not pil_images and messages and messages[-1].get("role") == "user":
        last_content = messages[-1].get("content")
        if isinstance(last_content, list) and last_content and last_content[0].get("type") == "text":
            text_prompt = last_content[0].get("text", "")
            # A simple regex to find URLs in text. This could be improved.
            # It will take the first one found.
            found_urls = re.findall(r'https?://[^\s/$.?#].[^\s]*\.(?:jpg|jpeg|png|gif|webp)', text_prompt, re.IGNORECASE)
            if found_urls:
                logger.info(f"Fallback: Found image URL '{found_urls[0]}' in text of last user message. Attempting to load.")
                img = _load_image_from_url(found_urls[0])
                if img:
                    pil_images.append(img)
                    # Optionally, you might want to modify the message to replace the text URL
                    # with a placeholder if your apply_chat_template doesn't handle it.
                    # For Qwen-VL, the template usually expects explicit image markers.
                    # This part is tricky without knowing exactly how apply_chat_template works.
                    # For now, we just load the image and let apply_chat_template deal with the text.
                    # The `images` param to processor() is the key.

    if pil_images:
        logger.info(f"Successfully loaded {len(pil_images)} PIL image(s).")
    return pil_images