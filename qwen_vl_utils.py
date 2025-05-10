# my_ai_studio/qwen_vl_utils.py
import logging
import re
from PIL import Image
import requests
from io import BytesIO
import base64 

logger = logging.getLogger(__name__)

def _load_image_from_url(url):
    try:
        if url.startswith('data:image'):
            # Expected format: data:image/png;base64,iVBORw0KGgo...
            try:
                header, encoded = url.split(',', 1)
                image_data = base64.b64decode(encoded)
                img = Image.open(BytesIO(image_data))
                logger.info(f"Successfully loaded image from data URL (type: {header.split(';')[0].split('/')[1]}).")
            except ValueError as e: 
                logger.error(f"Invalid data URL format for {url[:60]}...: {e}")
                return None
            except base64.binascii.Error as e: 
                logger.error(f"Failed to decode base64 image data from {url[:60]}...: {e}")
                return None
            except Exception as e: # Catch other potential errors during data URL processing
                logger.error(f"Error processing data URL {url[:60]}...: {e}")
                return None
        elif url.startswith('http://') or url.startswith('https://'):
            response = requests.get(url, stream=True, timeout=15) 
            response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
            img = Image.open(BytesIO(response.content))
            logger.info(f"Successfully loaded image from HTTP/S URL: {url}")
        else:
            logger.warning(f"Unsupported URL scheme or format, cannot load image: {url[:60]}...")
            return None

        # Ensure image is in RGB format, as some models expect this
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download image from {url}: {e}")
    except IOError as e: 
        logger.error(f"Failed to open image from {url[:60]}... (PIL error: possibly corrupted, not an image, or unsupported format): {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading image {url[:60]}...: {e}")
    return None

def extract_pil_images_from_messages(messages):
    """
    Extracts image URLs (http, https, or data URIs) from the 'image_url' content type in messages,
    loads them as PIL Images.
    """
    pil_images = []
    if not messages:
        return pil_images

    for message_idx, message in enumerate(messages): 
        if message.get("role") == "user" and isinstance(message.get("content"), list):
            for content_item_idx, content_item in enumerate(message["content"]): 
                if content_item.get("type") == "image_url":
                    image_url_data = content_item.get("image_url")
                    url_to_load = None
                    if isinstance(image_url_data, dict) and "url" in image_url_data:
                        url_to_load = image_url_data["url"]
                    elif isinstance(image_url_data, str): 
                        url_to_load = image_url_data
                    
                    if url_to_load:
                        is_http_url = url_to_load.startswith('http://') or url_to_load.startswith('https://')
                        is_data_url = url_to_load.startswith('data:image')

                        if is_http_url or is_data_url:
                            img = _load_image_from_url(url_to_load)
                            if img:
                                pil_images.append(img)
                        else:
                            logger.warning(f"URL '{url_to_load[:60]}...' in msg {message_idx}, content {content_item_idx} "
                                           f"is not a recognized HTTP/S or Data URL. Skipping.")
                    else:
                        logger.warning(f"Malformed image_url_data in msg {message_idx}, content {content_item_idx}: {image_url_data}")


    # Fallback for URLs in text content (kept for now, but less preferred with explicit upload/data URI)
    if not pil_images and messages and messages[-1].get("role") == "user":
        last_user_message_content = messages[-1].get("content")
        # Check if content is a list and first item is text
        if isinstance(last_user_message_content, list) and last_user_message_content and \
           isinstance(last_user_message_content[0], dict) and last_user_message_content[0].get("type") == "text":
            
            text_prompt = last_user_message_content[0].get("text", "")
            # Regex for http/https URLs common image extensions
            found_urls = re.findall(r'https?://[^\s/$.?#].[^\s]*\.(?:jpg|jpeg|png|gif|webp|avif)', text_prompt, re.IGNORECASE)
            if found_urls:
                # Try to load the first valid one found in text
                for url_in_text in found_urls:
                    logger.info(f"Fallback: Found image URL '{url_in_text}' in text of last user message. Attempting to load.")
                    img = _load_image_from_url(url_in_text)
                    if img:
                        pil_images.append(img)
                        break # Load only the first one found via fallback
    
    if pil_images:
        logger.info(f"Successfully loaded {len(pil_images)} PIL image(s) for model processing.")
    else:
        logger.info("No PIL images were loaded for this request (either none provided or failed to load).")
        
    return pil_images