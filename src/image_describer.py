"""
Image Describer Module
Uses OpenAI Vision API to generate descriptions for images.
"""

import base64
from typing import Optional
from openai import OpenAI


class ImageDescriber:
    """Generates text descriptions for images using OpenAI Vision API."""

    DEFAULT_PROMPT = """이 이미지를 자세히 분석하고 설명해주세요. 다음 항목들을 포함해주세요:

1. 이미지의 전체적인 내용과 주제
2. 주요 객체, 텍스트, 다이어그램 등의 세부 사항
3. 이미지에 포함된 텍스트가 있다면 그 내용
4. 이미지의 맥락이나 목적 (가능한 경우)

간결하면서도 핵심 정보를 담아 설명해주세요."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_tokens: int = 1000
    ):
        """
        Initialize the ImageDescriber.

        Args:
            api_key: OpenAI API key
            model: Vision model to use (default: gpt-4o-mini)
            max_tokens: Maximum tokens for the response
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def _get_mime_type(self, extension: str) -> str:
        """Get MIME type from file extension."""
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp'
        }
        return mime_types.get(extension.lower(), 'image/png')

    def describe_from_base64(
        self,
        base64_image: str,
        extension: str = '.png',
        custom_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a description for a base64-encoded image.

        Args:
            base64_image: Base64-encoded image string
            extension: File extension to determine MIME type
            custom_prompt: Custom prompt for the description

        Returns:
            Text description of the image
        """
        mime_type = self._get_mime_type(extension)
        prompt = custom_prompt or self.DEFAULT_PROMPT

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=self.max_tokens
        )

        return response.choices[0].message.content

    def describe_from_file(
        self,
        file_path: str,
        custom_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a description for an image file.

        Args:
            file_path: Path to the image file
            custom_prompt: Custom prompt for the description

        Returns:
            Text description of the image
        """
        from pathlib import Path

        path = Path(file_path)
        with open(path, 'rb') as f:
            image_data = f.read()

        base64_image = base64.b64encode(image_data).decode('utf-8')
        return self.describe_from_base64(base64_image, path.suffix, custom_prompt)

    def describe_from_url(
        self,
        image_url: str,
        custom_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a description for an image from URL.

        Args:
            image_url: URL of the image
            custom_prompt: Custom prompt for the description

        Returns:
            Text description of the image
        """
        prompt = custom_prompt or self.DEFAULT_PROMPT

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=self.max_tokens
        )

        return response.choices[0].message.content


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        describer = ImageDescriber(api_key)
        print("ImageDescriber initialized successfully")
    else:
        print("OPENAI_API_KEY not found in environment")
