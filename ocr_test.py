#coding:utf8
import os
import json
import base64
import argparse
import requests

API_URL = os.environ.get("OCRFLUX_API_URL", "http://ip:port/v1/chat/completions")
MODEL = os.environ.get("OCRFLUX_MODEL_PATH", "model_path")


def call_ocr(image_path: str, api_url: str = API_URL, model: str = MODEL, timeout: float = 30.0) -> str:
    """
    Call the OCRFlux API to perform text recognition on an image.

    :param image_path: Path to the image file
    :param api_url: API endpoint, defaults to env OCRFLUX_API_URL or the hardcoded default
    :param model: Model path, defaults to env OCRFLUX_MODEL_PATH or the hardcoded default
    :param timeout: Request timeout in seconds
    :return: Recognized text; if extraction fails, returns the raw JSON string
    """
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,  # Must match MODEL_PATH in docker-compose.server.yml and model_path in ocrflux/server.sh
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": (
                            "请逐字逐句识别图像中的所有可见文本，不要总结、不要编造、不要产生幻觉。"
                        ),
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "seed": 42,
        "max_tokens": 2048,
    }

    resp = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    # Try to extract text and be compatible with multiple response formats
    text = None
    try:
        # OpenAI-style: choices[0].message.content
        if isinstance(data, dict) and "choices" in data and data["choices"]:
            choice0 = data["choices"][0]
            if isinstance(choice0, dict):
                msg = choice0.get("message") or {}
                text = msg.get("content")
                if not text:
                    # Some implementations may put it directly in 'content'
                    text = choice0.get("content")
    except Exception:
        pass

    if isinstance(text, str) and text.strip():
        return text.strip()

    # If text cannot be extracted, return the raw JSON (for debugging)
    return json.dumps(data, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple OCRFlux test script")
    parser.add_argument("image", nargs="?", default="jpg_test.jpg", help="Path to the image to be recognized, default is jpg_test.jpg")
    parser.add_argument("--api-url", default=API_URL, help="Service URL, default reads OCRFLUX_API_URL or file default value")
    parser.add_argument("--model", default=MODEL, help="Model path, default reads OCRFLUX_MODEL_PATH or file default value")
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout in seconds, default is 30")
    args = parser.parse_args()

    try:
        result = call_ocr(args.image, api_url=args.api_url, model=args.model, timeout=args.timeout)
        print(result)
    except requests.HTTPError as e:
        print(f"HTTP error: {e} - Response content: {getattr(e.response, 'text', '')}")
    except Exception as e:
        print(f"Call failed: {e}")