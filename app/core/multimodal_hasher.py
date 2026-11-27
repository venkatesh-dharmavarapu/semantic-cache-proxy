import base64
import hashlib
import io
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)


class PerceptualImageHasher:
    """
    Computes perceptual difference hashes (dHash) for images to ensure
    caching is robust against resizing, JPEG artifacts, and format transcoding.
    """

    @classmethod
    def compute_dhash(cls, image_bytes: bytes, hash_size: int = 8) -> str:
        """
        Computes 64-bit difference hash (dHash) from image bytes.
        1. Resize to (hash_size + 1, hash_size) -> (9, 8)
        2. Convert to grayscale (L)
        3. Compare adjacent pixels horizontally
        """
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                # Convert to grayscale and resize to 9x8
                resized = img.convert("L").resize(
                    (hash_size + 1, hash_size), Image.Resampling.LANCZOS
                )
                pixels = list(resized.getdata())

                # Compare pixel values across adjacent columns
                diff_bits = []
                for row in range(hash_size):
                    for col in range(hash_size):
                        idx = row * (hash_size + 1) + col
                        diff_bits.append("1" if pixels[idx] > pixels[idx + 1] else "0")

                # Convert 64 bits to hexadecimal string
                bit_string = "".join(diff_bits)
                hex_hash = f"{int(bit_string, 2):016x}"
                return hex_hash
        except Exception as exc:
            logger.warning("Failed to compute perceptual hash, falling back to SHA-256: %s", exc)
            return hashlib.sha256(image_bytes).hexdigest()[:16]

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """Computes bit-level Hamming distance between two hexadecimal hashes."""
        val1 = int(hash1, 16)
        val2 = int(hash2, 16)
        xor_val = val1 ^ val2
        return bin(xor_val).count("1")

    @classmethod
    def extract_images_and_text(
        cls, messages: List[Dict[str, Any]]
    ) -> Tuple[str, List[str], bool]:
        """
        Extracts combined text content and computes perceptual hashes for all images
        found in OpenAI multi-modal messages format.

        Returns:
            (extracted_text, list_of_image_hashes, is_multimodal)
        """
        text_segments: List[str] = []
        image_hashes: List[str] = []
        is_multimodal = False

        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                text_segments.append(content)
            elif isinstance(content, list):
                # Multi-part content block
                for part in content:
                    part_type = part.get("type")
                    if part_type == "text":
                        text_segments.append(part.get("text", ""))
                    elif part_type == "image_url":
                        is_multimodal = True
                        image_info = part.get("image_url", {})
                        url_or_b64 = image_info.get("url", "")
                        if url_or_b64.startswith("data:image"):
                            # Base64 data URI
                            b64_data = re.sub(r"^data:image/.+;base64,", "", url_or_b64)
                            try:
                                raw_bytes = base64.b64decode(b64_data)
                                img_hash = cls.compute_dhash(raw_bytes)
                                image_hashes.append(img_hash)
                            except Exception as e:
                                logger.warning("Failed to decode base64 image: %s", e)
                        else:
                            # External URL: hash the normalized URL string
                            url_hash = hashlib.sha256(url_or_b64.strip().encode("utf-8")).hexdigest()[:16]
                            image_hashes.append(url_hash)

        combined_text = " ".join(t.strip() for t in text_segments if t.strip())
        return combined_text, image_hashes, is_multimodal