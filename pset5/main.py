#!/usr/bin/env python3
"""Standalone multimodal smolagents runner for Homework 5.

This script lifts the usable agent logic from
`Copy_of_Homework_5_AI_Agents.ipynb` into a local Python entrypoint.
It keeps the industrial-design inspiration agent prompt from the notebook,
uses `gpt-5.4-mini` as the default OpenAI-backed agent model, and adds custom
tools for extracting webpage images, downloading them, and evaluating them
with a vision-capable OpenAI model.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import shutil
import sys
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from markdownify import markdownify as md
from openai import OpenAI
from PIL import Image, ImageDraw, ImageOps

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env", override=False)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "downloaded_images"
CURATED_DIR = PROJECT_DIR / "curated"
SELECTION_BOARD_DIR = PROJECT_DIR / "selection_boards"
DEFAULT_AGENT_MODEL = os.getenv("OPENAI_AGENT_MODEL", "gpt-5.4-mini")
DEFAULT_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", DEFAULT_AGENT_MODEL)

SYSTEM_INSTRUCTIONS = (
    "You are a helpful industrial design inspiration research agent. "
    "Your job is to help users find visual references for product design, "
    "industrial design, CMF, form studies, mechanisms, and interaction design. "
    "Use your tools to search the web, visit relevant pages, extract candidate "
    "images, download them when useful, and evaluate them against the design brief. "
    "If the user asks for images, inspiration references, moodboards, or visual "
    "precedents, you must inspect at least one real candidate image with "
    "extract_images_from_webpage, download_image, and evaluate_image_with_vlm "
    "before giving a final answer, unless every candidate page is inaccessible. "
    "After downloading candidate images, use evaluate_design_quality to screen "
    "how polished, intentional, and design-relevant they look. Prefer approved "
    "images over flagged ones, and do not present flagged images as primary "
    "references unless no better option exists. "
    "Do not answer image requests from search snippets alone. "
    "Use concise search queries based on the product/topic itself, not the full "
    "formatting instructions from the user's prompt. "
    "If a search query returns zero useful results twice, stop repeating it and "
    "broaden or switch sources. "
    "Prioritize design-oriented sources such as Pinterest, Behance, Core77, "
    "Yanko Design, Dezeen, Design Milk, Designboom, product portfolios, studio "
    "websites, and manufacturer pages. "
    "For each result, summarize why it is visually relevant to the user's request. "
    "When possible, include visual tags such as form language, material, color, "
    "finish, affordance, interaction, and product category. "
    "Do not claim that you inspected an image unless you actually used the "
    "image-extraction, download, and image-evaluation tools. "
    "If a site is inaccessible, JavaScript-rendered, login-gated, or unclear, say "
    "so explicitly. Cite sources as URLs. If uncertain, say so explicitly."
)

DEFAULT_QUERY = (
    "Find industrial design inspiration images for a compact countertop coffee "
    "grinder with a soft rounded aluminum body, playful CMF, and simple physical "
    "controls. Search design-oriented sites such as Pinterest, Behance, Core77, "
    "Yanko Design, Dezeen, Design Milk, Designboom, and product design portfolios. "
    "Return 3 strong inspiration references. For each one, include: title or page "
    "name, source URL, why it is relevant, and visual tags. Be explicit if a "
    "source is inaccessible, login-gated, or not visually inspectable."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Homework 5 multimodal smolagents workflow.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=DEFAULT_QUERY,
        help="User query to send to the agent.",
    )
    parser.add_argument(
        "--agent-model",
        default=DEFAULT_AGENT_MODEL,
        help="OpenAI model used by smolagents for tool orchestration.",
    )
    parser.add_argument(
        "--vision-model",
        default=DEFAULT_VISION_MODEL,
        help="OpenAI model used inside the image evaluation tool.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where downloaded images should be stored.",
    )
    parser.add_argument(
        "--build-selection-board",
        type=int,
        default=0,
        help="Build a selection board from the latest curated/downloaded images.",
    )
    parser.add_argument(
        "--interactive-pick",
        action="store_true",
        help="Prompt in the terminal for which board image to refine from.",
    )
    parser.add_argument(
        "--selected-index",
        type=int,
        default=0,
        help="1-based candidate index to refine from after building a board.",
    )
    parser.add_argument(
        "--run-refined-round",
        action="store_true",
        help="After selecting a board image, run another agent round using the refined direction.",
    )
    return parser.parse_args()


def ensure_openai_api_key() -> str:
    load_dotenv(PROJECT_DIR / ".env", override=False)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Copy pset5/.env.example to pset5/.env "
            "or export the variable before running main.py."
        )
    return api_key


def require_smolagents() -> tuple[Any, Any, Any]:
    try:
        from smolagents import (
            OpenAIServerModel,
            Tool,
            ToolCallingAgent,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "smolagents is not installed in this environment. Activate pset5/.venv "
            "and install dependencies with `pip install -r pset5/requirements.txt`."
        ) from exc

    return OpenAIServerModel, Tool, ToolCallingAgent


def response_to_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text.strip()

    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)

    return "\n".join(part for part in parts if part).strip()


def normalize_jsonish_text(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return cleaned


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def review_sidecar_path(image_path: Path) -> Path:
    return image_path.with_suffix(image_path.suffix + ".review.json")


def load_review_sidecar(image_path: Path) -> dict[str, Any] | None:
    sidecar = review_sidecar_path(image_path)
    if not sidecar.exists():
        return None
    try:
        return json.loads(sidecar.read_text())
    except Exception:
        return None


def condense_design_request(user_request: str) -> str:
    normalized = re.sub(r"\s+", " ", user_request).strip()
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    skip_markers = (
        "return ",
        "for each ",
        "be explicit ",
        "include:",
        "search design-oriented sites",
        "cite sources",
    )

    kept: list[str] = []
    for sentence in sentences:
        lower = sentence.lower().strip()
        if any(marker in lower for marker in skip_markers):
            continue
        kept.append(sentence.strip())

    candidate = " ".join(kept).strip() or normalized
    candidate = re.sub(r"^(find|show me|search for|get me)\s+", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(
        r"^(industrial design inspiration images for|industrial design inspiration for|inspiration images for)\s+",
        "",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = candidate.rstrip(".")
    if len(candidate) > 180:
        candidate = candidate[:180].rsplit(" ", 1)[0]
    return candidate


def discover_candidate_images(limit: int, output_dir: Path) -> list[Path]:
    candidate_dirs = [
        CURATED_DIR / "approved",
        CURATED_DIR / "flagged",
        output_dir,
        PROJECT_DIR / "downloads",
    ]
    image_suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    seen: set[str] = set()
    images: list[Path] = []

    for directory in candidate_dirs:
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in image_suffixes:
                review = load_review_sidecar(path)
                canonical_key = str(review.get("image_path")) if isinstance(review, dict) and review.get("image_path") else str(path.resolve())
                if canonical_key not in seen:
                    seen.add(canonical_key)
                    images.append(path)

    images.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return images[:limit]


def build_tools(client: OpenAI, vision_model: str, default_output_dir: Path) -> list[Any]:
    OpenAIServerModel, Tool, ToolCallingAgent = require_smolagents()
    _ = (OpenAIServerModel, ToolCallingAgent)

    class SearchWebTool(Tool):
        name = "search_web"
        description = (
            "Search the public web using DuckDuckGo's HTML endpoint and return "
            "result titles, URLs, and snippets."
        )
        inputs = {
            "query": {
                "type": "string",
                "description": "The search query to run.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of search results to return.",
                "nullable": True,
            },
        }
        output_type = "string"

        def forward(self, query: str, max_results: int = 5) -> str:
            try:
                response = requests.post(
                    "https://html.duckduckgo.com/html/",
                    data={"q": query},
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=20,
                )
                response.raise_for_status()
            except Exception as exc:
                return json.dumps(
                    {"query": query, "error": f"Search failed: {exc}"},
                    indent=2,
                )

            soup = BeautifulSoup(response.text, "html.parser")
            results: list[dict[str, str]] = []
            for result in soup.select(".result")[:max_results]:
                link = result.select_one(".result__a")
                snippet = result.select_one(".result__snippet")
                if not link or not link.get("href"):
                    continue
                href = link.get("href")
                if any(
                    bad_fragment in href
                    for bad_fragment in ("duckduckgo.com/y.js", "bing.com/aclick", "amazon.com")
                ):
                    continue
                results.append(
                    {
                        "title": link.get_text(" ", strip=True),
                        "url": href,
                        "snippet": snippet.get_text(" ", strip=True) if snippet else "",
                    }
                )

            return json.dumps(
                {
                    "query": query,
                    "num_results": len(results),
                    "results": results,
                },
                indent=2,
            )

    class FetchWebpageTextTool(Tool):
        name = "fetch_webpage_text"
        description = (
            "Fetch a webpage with a browser-like user agent and return the title "
            "plus a markdown/text extraction for downstream reasoning."
        )
        inputs = {
            "url": {
                "type": "string",
                "description": "The webpage URL to fetch.",
            }
        }
        output_type = "string"

        def forward(self, url: str) -> str:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            }
            try:
                response = requests.get(url, headers=headers, timeout=20)
                response.raise_for_status()
            except Exception as exc:
                return json.dumps(
                    {"url": url, "error": f"Failed to fetch webpage: {exc}"},
                    indent=2,
                )

            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.title.text.strip() if soup.title and soup.title.text else ""
            markdown = md(response.text)
            markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()

            return json.dumps(
                {
                    "url": url,
                    "title": title,
                    "text_excerpt": markdown[:8000],
                },
                indent=2,
            )

    class DesignSearchPromptTool(Tool):
        name = "design_search_prompt"
        description = (
            "Turn a raw industrial or product design inspiration request into "
            "targeted web search queries and evaluation criteria."
        )
        inputs = {
            "user_request": {
                "type": "string",
                "description": "The user's original design inspiration request.",
            }
        }
        output_type = "string"

        def forward(self, user_request: str) -> str:
            request = condense_design_request(user_request)
            design_sites = [
                "pinterest.com",
                "behance.net",
                "core77.com",
                "yankodesign.com",
                "dezeen.com",
                "design-milk.com",
                "designboom.com",
                "lemanoosh.com",
                "notcot.org",
                "designwanted.com",
            ]
            search_angles = [
                "industrial design inspiration",
                "product design inspiration",
                "CMF material finish color",
                "form study product design",
                "concept product design",
                "minimal physical interface",
                "soft rounded product design",
                "consumer electronics design",
                "mechanism design detail",
                "design portfolio case study",
            ]
            visual_attributes = [
                "form language",
                "material",
                "finish",
                "color",
                "texture",
                "scale",
                "proportion",
                "interaction",
                "physical controls",
                "affordance",
                "manufacturing detail",
            ]

            queries: list[str] = []
            for site in design_sites:
                queries.append(f"{request} site:{site}")
            for angle in search_angles[:6]:
                queries.append(f"{request} {angle}")
            queries.extend(
                [
                    f"{request} moodboard industrial design references",
                    f"{request} visual references product design",
                    f"{request} product design case study images",
                    f"{request} CMF moodboard product design",
                    f"{request} design language reference images",
                ]
            )

            return json.dumps(
                {
                    "interpreted_request": request,
                    "recommended_search_strategy": (
                        "Start with concise design-site searches on the product/topic "
                        "itself, then broaden into CMF, form language, interaction "
                        "details, and case studies."
                    ),
                    "visual_attributes_to_evaluate": visual_attributes,
                    "search_queries": queries[:16],
                },
                indent=2,
            )

    class ExtractImagesFromWebpageTool(Tool):
        name = "extract_images_from_webpage"
        description = (
            "Extract candidate image URLs from a webpage after it has been found or visited."
        )
        inputs = {
            "url": {
                "type": "string",
                "description": "The webpage URL to inspect for candidate images.",
            },
            "max_images": {
                "type": "integer",
                "description": "Maximum number of image candidates to return.",
                "nullable": True,
            },
        }
        output_type = "string"

        def forward(self, url: str, max_images: int = 25) -> str:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            }
            try:
                response = requests.get(url, headers=headers, timeout=20)
                response.raise_for_status()
            except Exception as exc:
                return json.dumps(
                    {
                        "page_url": url,
                        "error": f"Failed to fetch webpage: {exc}",
                    },
                    indent=2,
                )

            soup = BeautifulSoup(response.text, "html.parser")
            images: list[dict[str, Any]] = []
            title = soup.title.text.strip() if soup.title and soup.title.text else ""

            for tag_name, attrs, source_type in (
                ("meta", {"property": "og:image"}, "og:image"),
                ("meta", {"name": "twitter:image"}, "twitter:image"),
            ):
                tag = soup.find(tag_name, attrs=attrs)
                if tag and tag.get("content"):
                    images.append(
                        {
                            "image_url": urljoin(url, tag["content"]),
                            "source_type": source_type,
                            "alt": "",
                            "page_url": url,
                        }
                    )

            for img in soup.find_all("img"):
                src = (
                    img.get("src")
                    or img.get("data-src")
                    or img.get("data-original")
                    or img.get("data-lazy-src")
                    or img.get("data-pin-media")
                )
                if not src and img.get("srcset"):
                    src = img.get("srcset").split(",")[-1].strip().split(" ")[0]
                if not src:
                    continue

                full_url = urljoin(url, src)
                if full_url.startswith("data:"):
                    continue

                images.append(
                    {
                        "image_url": full_url,
                        "source_type": "img",
                        "alt": img.get("alt", ""),
                        "width": img.get("width", ""),
                        "height": img.get("height", ""),
                        "page_url": url,
                    }
                )

            unique_images: list[dict[str, Any]] = []
            seen: set[str] = set()
            for image in images:
                image_url = image["image_url"]
                if image_url in seen:
                    continue
                seen.add(image_url)
                unique_images.append(image)

            return json.dumps(
                {
                    "page_url": url,
                    "page_title": title,
                    "num_images_found": len(unique_images),
                    "images": unique_images[:max_images],
                    "note": (
                        "Static HTML extraction works best. JavaScript-heavy sites may "
                        "hide or lazy-load the actual image assets."
                    ),
                },
                indent=2,
            )

    class DownloadImageTool(Tool):
        name = "download_image"
        description = (
            "Download an image URL to disk so a vision model can inspect the file."
        )
        inputs = {
            "image_url": {
                "type": "string",
                "description": "The image URL to download.",
            },
            "output_dir": {
                "type": "string",
                "description": "Directory where the image should be saved.",
                "nullable": True,
            },
        }
        output_type = "string"

        def __init__(self, base_output_dir: Path):
            super().__init__()
            self.base_output_dir = base_output_dir

        def forward(self, image_url: str, output_dir: str = "") -> str:
            target_dir = Path(output_dir).expanduser() if output_dir else self.base_output_dir
            target_dir.mkdir(parents=True, exist_ok=True)

            try:
                response = requests.get(
                    image_url,
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=20,
                )
                response.raise_for_status()
            except Exception as exc:
                return json.dumps(
                    {
                        "image_url": image_url,
                        "error": f"Failed to download image: {exc}",
                    },
                    indent=2,
                )

            content_type = response.headers.get("Content-Type", "")
            if "image" not in content_type:
                return json.dumps(
                    {
                        "image_url": image_url,
                        "error": f"URL did not return an image. Content-Type: {content_type}",
                    },
                    indent=2,
                )

            try:
                image = Image.open(BytesIO(response.content)).convert("RGB")
            except Exception as exc:
                return json.dumps(
                    {
                        "image_url": image_url,
                        "error": f"Could not decode downloaded content as an image: {exc}",
                    },
                    indent=2,
                )

            parsed = urlparse(image_url)
            stem = re.sub(r"[^a-zA-Z0-9_-]", "_", Path(parsed.path).stem) or "image"
            file_path = target_dir / f"{stem}.jpg"
            counter = 1
            while file_path.exists():
                file_path = target_dir / f"{stem}_{counter}.jpg"
                counter += 1

            image.save(file_path, format="JPEG", quality=95)
            return json.dumps(
                {
                    "image_url": image_url,
                    "local_path": str(file_path),
                    "width": image.width,
                    "height": image.height,
                    "content_type": content_type,
                },
                indent=2,
            )

    class EvaluateImageWithVLMTool(Tool):
        name = "evaluate_image_with_vlm"
        description = (
            "Evaluate a downloaded image against the design brief with an OpenAI "
            "vision-capable model and return a structured assessment."
        )
        inputs = {
            "image_path": {
                "type": "string",
                "description": "Local path to the downloaded image.",
            },
            "user_request": {
                "type": "string",
                "description": "The original user design request or brief.",
            },
        }
        output_type = "string"

        def __init__(self, api_client: OpenAI, model_id: str):
            super().__init__()
            self.client = api_client
            self.model_id = model_id

        def _image_to_data_url(self, image_path: str) -> str:
            path = Path(image_path)
            with path.open("rb") as handle:
                encoded = base64.b64encode(handle.read()).decode("utf-8")

            suffix = path.suffix.lower()
            mime = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
            }.get(suffix, "image/jpeg")
            return f"data:{mime};base64,{encoded}"

        def forward(self, image_path: str, user_request: str) -> str:
            path = Path(image_path)
            if not path.exists():
                return json.dumps(
                    {
                        "image_path": image_path,
                        "error": "Image file does not exist.",
                    },
                    indent=2,
                )

            prompt = f"""
You are evaluating whether this image is useful as industrial design inspiration.

User request:
{user_request}

Return strict JSON with these fields:
- relevance_score: integer from 0 to 10
- keep: true or false
- visual_summary: short description of what is visible
- why_relevant: why this image matches or does not match the request
- useful_design_attributes: list of useful reference qualities
- visual_tags: list of tags such as material, color, finish, form language, controls, scale, or product type
- concerns: list of issues such as wrong category, too generic, low image quality, unclear product, or weak design signal
""".strip()

            try:
                response = self.client.responses.create(
                    model=self.model_id,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {
                                    "type": "input_image",
                                    "image_url": self._image_to_data_url(image_path),
                                },
                            ],
                        }
                    ],
                )
            except Exception as exc:
                return json.dumps(
                    {
                        "image_path": image_path,
                        "error": f"VLM evaluation failed: {exc}",
                    },
                    indent=2,
                )

            evaluation_text = response_to_text(response)
            return json.dumps(
                {
                    "image_path": image_path,
                    "user_request": user_request,
                    "vision_model": self.model_id,
                    "evaluation": normalize_jsonish_text(evaluation_text),
                },
                indent=2,
            )

    class EvaluateDesignQualityTool(Tool):
        name = "evaluate_design_quality"
        description = (
            "Score how polished, intentional, and design-strong a downloaded image "
            "looks, then archive it as approved or flagged for downstream curation."
        )
        inputs = {
            "image_path": {
                "type": "string",
                "description": "Local path to the downloaded image.",
            },
            "user_request": {
                "type": "string",
                "description": "The original user design request or brief.",
            },
            "min_design_score": {
                "type": "integer",
                "description": "Minimum overall design score required for approval.",
                "nullable": True,
            },
            "archive_flagged": {
                "type": "boolean",
                "description": "Whether to archive flagged images into a flagged bucket.",
                "nullable": True,
            },
        }
        output_type = "string"

        def __init__(self, api_client: OpenAI, model_id: str):
            super().__init__()
            self.client = api_client
            self.model_id = model_id

        def _image_to_data_url(self, image_path: str) -> str:
            path = Path(image_path)
            with path.open("rb") as handle:
                encoded = base64.b64encode(handle.read()).decode("utf-8")

            mime = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
            }.get(path.suffix.lower(), "image/jpeg")
            return f"data:{mime};base64,{encoded}"

        def forward(
            self,
            image_path: str,
            user_request: str,
            min_design_score: int = 7,
            archive_flagged: bool = True,
        ) -> str:
            path = Path(image_path)
            if not path.exists():
                return json.dumps(
                    {"image_path": image_path, "error": "Image file does not exist."},
                    indent=2,
                )

            prompt = f"""
You are screening this image as a design-reference candidate.

User request:
{user_request}

Return strict JSON with these fields:
- overall_design_score: integer from 0 to 10
- polish_score: integer from 0 to 10
- form_language_score: integer from 0 to 10
- cmf_score: integer from 0 to 10
- originality_score: integer from 0 to 10
- category_fit_score: integer from 0 to 10
- keep: true or false
- design_direction_summary: short summary of the visual direction
- strengths: list of what looks strong or polished
- weaknesses: list of what looks weak, generic, noisy, or off-brief
- recommended_tags: list of useful downstream search/style tags
""".strip()

            try:
                response = self.client.responses.create(
                    model=self.model_id,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {
                                    "type": "input_image",
                                    "image_url": self._image_to_data_url(image_path),
                                },
                            ],
                        }
                    ],
                )
            except Exception as exc:
                return json.dumps(
                    {"image_path": image_path, "error": f"Design-quality evaluation failed: {exc}"},
                    indent=2,
                )

            evaluation = normalize_jsonish_text(response_to_text(response))
            keep = False
            status = "flagged"

            if isinstance(evaluation, dict):
                overall_score = int(evaluation.get("overall_design_score", 0) or 0)
                category_fit_score = int(evaluation.get("category_fit_score", overall_score) or 0)
                suggested_keep = evaluation.get("keep")
                keep = bool(suggested_keep) if isinstance(suggested_keep, bool) else False
                keep = keep or (overall_score >= min_design_score and category_fit_score >= max(4, min_design_score - 2))
                status = "approved" if keep else "flagged"

            curated_bucket = CURATED_DIR / status
            curated_bucket.mkdir(parents=True, exist_ok=True)
            archived_copy_path = unique_path(curated_bucket / path.name)
            shutil.copy2(path, archived_copy_path)

            review_payload = {
                "image_path": str(path),
                "user_request": user_request,
                "vision_model": self.model_id,
                "min_design_score": min_design_score,
                "status": status,
                "keep": keep,
                "archived_copy_path": str(archived_copy_path),
                "evaluation": evaluation,
            }

            source_review_path = review_sidecar_path(path)
            source_review_path.write_text(json.dumps(review_payload, indent=2))

            archived_review_path = review_sidecar_path(archived_copy_path)
            archived_review_path.write_text(json.dumps(review_payload, indent=2))

            if not keep and not archive_flagged:
                try:
                    archived_copy_path.unlink()
                    archived_review_path.unlink(missing_ok=True)
                    review_payload["archived_copy_path"] = None
                except Exception:
                    pass

            return json.dumps(review_payload, indent=2)

    class BuildImageSelectionBoardTool(Tool):
        name = "build_image_selection_board"
        description = (
            "Create a labeled contact-sheet board from several downloaded images so "
            "a user can pick a direction to explore further."
        )
        inputs = {
            "image_paths_json": {
                "type": "string",
                "description": "JSON array of local image paths to include on the board.",
            },
            "title": {
                "type": "string",
                "description": "Optional title shown at the top of the board.",
                "nullable": True,
            },
            "max_images": {
                "type": "integer",
                "description": "Maximum number of images to include on the board.",
                "nullable": True,
            },
        }
        output_type = "string"

        def forward(
            self,
            image_paths_json: str,
            title: str = "Direction options",
            max_images: int = 3,
        ) -> str:
            try:
                parsed_paths = json.loads(image_paths_json)
            except json.JSONDecodeError:
                return json.dumps(
                    {"error": "image_paths_json must be a valid JSON array of image paths."},
                    indent=2,
                )

            if not isinstance(parsed_paths, list):
                return json.dumps(
                    {"error": "image_paths_json must decode to a list of image paths."},
                    indent=2,
                )

            image_paths = [str(path) for path in parsed_paths]
            valid_paths = [Path(path) for path in image_paths if Path(path).exists()]
            valid_paths = valid_paths[:max_images]
            if not valid_paths:
                return json.dumps(
                    {"error": "No valid image paths were provided for the selection board."},
                    indent=2,
                )

            SELECTION_BOARD_DIR.mkdir(parents=True, exist_ok=True)
            cell_width = 360
            image_height = 320
            label_height = 92
            padding = 20
            cols = min(3, len(valid_paths))
            rows = math.ceil(len(valid_paths) / cols)
            board_width = cols * cell_width + padding * (cols + 1)
            board_height = 90 + rows * (image_height + label_height + padding) + padding

            board = Image.new("RGB", (board_width, board_height), color=(245, 244, 240))
            draw = ImageDraw.Draw(board)
            draw.text((padding, 24), title, fill=(20, 20, 20))

            candidates: list[dict[str, Any]] = []
            for index, path in enumerate(valid_paths, start=1):
                row = (index - 1) // cols
                col = (index - 1) % cols
                x = padding + col * (cell_width + padding)
                y = 70 + row * (image_height + label_height + padding)

                with Image.open(path) as raw_image:
                    tile = ImageOps.fit(raw_image.convert("RGB"), (cell_width, image_height), method=Image.Resampling.LANCZOS)
                board.paste(tile, (x, y))

                review = load_review_sidecar(path)
                score = None
                if review and isinstance(review.get("evaluation"), dict):
                    score = review["evaluation"].get("overall_design_score")
                draw.rectangle(
                    [(x, y + image_height), (x + cell_width, y + image_height + label_height)],
                    fill=(255, 255, 255),
                    outline=(210, 210, 210),
                )
                score_text = f" | design score {score}/10" if score is not None else ""
                draw.text(
                    (x + 12, y + image_height + 12),
                    f"{index}. {path.name}{score_text}",
                    fill=(25, 25, 25),
                )
                if review and isinstance(review.get("evaluation"), dict):
                    summary = review["evaluation"].get("design_direction_summary") or review["evaluation"].get("visual_summary")
                    if summary:
                        draw.text((x + 12, y + image_height + 40), str(summary)[:64], fill=(85, 85, 85))

                candidates.append(
                    {
                        "index": index,
                        "image_path": str(path),
                        "review_path": str(review_sidecar_path(path)) if review_sidecar_path(path).exists() else None,
                        "design_score": score,
                    }
                )

            board_path = unique_path(SELECTION_BOARD_DIR / "selection_board.jpg")
            board.save(board_path, quality=92)
            manifest_path = board_path.with_suffix(".json")
            manifest = {
                "title": title,
                "board_path": str(board_path),
                "candidates": candidates,
            }
            manifest_path.write_text(json.dumps(manifest, indent=2))
            manifest["manifest_path"] = str(manifest_path)
            return json.dumps(manifest, indent=2)

    class RefineDirectionFromSelectedImageTool(Tool):
        name = "refine_direction_from_selected_image"
        description = (
            "Take one chosen reference image and turn it into a tighter design "
            "direction brief plus refined search queries for the next round."
        )
        inputs = {
            "image_path": {
                "type": "string",
                "description": "Local path to the selected reference image.",
            },
            "original_request": {
                "type": "string",
                "description": "The original user request or design brief.",
            },
            "num_queries": {
                "type": "integer",
                "description": "Number of refined search queries to return.",
                "nullable": True,
            },
        }
        output_type = "string"

        def __init__(self, api_client: OpenAI, model_id: str):
            super().__init__()
            self.client = api_client
            self.model_id = model_id

        def _image_to_data_url(self, image_path: str) -> str:
            path = Path(image_path)
            with path.open("rb") as handle:
                encoded = base64.b64encode(handle.read()).decode("utf-8")

            mime = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
            }.get(path.suffix.lower(), "image/jpeg")
            return f"data:{mime};base64,{encoded}"

        def forward(self, image_path: str, original_request: str, num_queries: int = 6) -> str:
            path = Path(image_path)
            if not path.exists():
                return json.dumps(
                    {"image_path": image_path, "error": "Image file does not exist."},
                    indent=2,
                )

            prompt = f"""
You are converting a chosen visual reference into a refined design-search direction.

Original request:
{original_request}

Return strict JSON with these fields:
- direction_name: short name for the chosen direction
- refined_brief: one paragraph describing the refined direction to pursue next
- preserve: list of visual qualities to preserve from the chosen image
- push_further: list of qualities to exaggerate or explore further
- avoid: list of qualities to avoid in the next search round
- search_queries: list of {num_queries} concise follow-up search queries
""".strip()

            try:
                response = self.client.responses.create(
                    model=self.model_id,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {
                                    "type": "input_image",
                                    "image_url": self._image_to_data_url(image_path),
                                },
                            ],
                        }
                    ],
                )
            except Exception as exc:
                return json.dumps(
                    {"image_path": image_path, "error": f"Direction refinement failed: {exc}"},
                    indent=2,
                )

            return json.dumps(
                {
                    "image_path": str(path),
                    "original_request": original_request,
                    "refinement": normalize_jsonish_text(response_to_text(response)),
                },
                indent=2,
            )

    return [
        SearchWebTool(),
        FetchWebpageTextTool(),
        DesignSearchPromptTool(),
        ExtractImagesFromWebpageTool(),
        DownloadImageTool(default_output_dir),
        EvaluateImageWithVLMTool(client, vision_model),
        EvaluateDesignQualityTool(client, vision_model),
        BuildImageSelectionBoardTool(),
        RefineDirectionFromSelectedImageTool(client, vision_model),
    ]


def build_agent(agent_model: str, vision_model: str, output_dir: Path) -> Any:
    api_key = ensure_openai_api_key()
    client = OpenAI(api_key=api_key)
    OpenAIServerModel, _, ToolCallingAgent = require_smolagents()
    model = OpenAIServerModel(model_id=agent_model, api_key=api_key)
    tools = build_tools(client=client, vision_model=vision_model, default_output_dir=output_dir)
    return ToolCallingAgent(tools=tools, model=model, max_steps=12, verbosity_level=0)


def run_selection_board_flow(
    *,
    agent_model: str,
    vision_model: str,
    output_dir: Path,
    user_request: str,
    board_size: int,
    selected_index: int,
    interactive_pick: bool,
    run_refined_round: bool,
) -> int:
    api_key = ensure_openai_api_key()
    client = OpenAI(api_key=api_key)
    tools = build_tools(client=client, vision_model=vision_model, default_output_dir=output_dir)
    tool_map = {tool.name: tool for tool in tools}

    candidate_paths = discover_candidate_images(limit=max(board_size, 1), output_dir=output_dir)
    if not candidate_paths:
        print("No candidate images found. Run the agent first so it downloads or curates some images.")
        return 1

    board_payload = json.loads(
        tool_map["build_image_selection_board"].forward(
            image_paths_json=json.dumps([str(path) for path in candidate_paths]),
            title="Pick a direction to explore further",
            max_images=board_size,
        )
    )
    print(json.dumps(board_payload, indent=2))

    chosen_index = selected_index
    if interactive_pick and chosen_index <= 0:
        raw_choice = input("Pick a candidate index to refine from: ").strip()
        try:
            chosen_index = int(raw_choice)
        except ValueError:
            print("Invalid selection. Expected an integer index.")
            return 1

    if chosen_index <= 0:
        return 0

    candidates = board_payload.get("candidates", [])
    selected = next((candidate for candidate in candidates if candidate.get("index") == chosen_index), None)
    if selected is None:
        print(f"Selection index {chosen_index} is not present on the board.")
        return 1

    refinement_payload = json.loads(
        tool_map["refine_direction_from_selected_image"].forward(
            image_path=selected["image_path"],
            original_request=user_request,
            num_queries=6,
        )
    )
    print(json.dumps(refinement_payload, indent=2))

    if not run_refined_round:
        return 0

    refinement = refinement_payload.get("refinement", {})
    refined_query = user_request
    if isinstance(refinement, dict):
        refined_query = (
            f"{user_request}\n\n"
            f"Chosen direction: {refinement.get('direction_name', 'selected reference')}\n"
            f"Refined brief: {refinement.get('refined_brief', '')}\n"
            f"Preserve: {json.dumps(refinement.get('preserve', []))}\n"
            f"Push further: {json.dumps(refinement.get('push_further', []))}\n"
            f"Avoid: {json.dumps(refinement.get('avoid', []))}\n"
            f"Suggested search queries: {json.dumps(refinement.get('search_queries', []))}"
        )

    agent = build_agent(agent_model=agent_model, vision_model=vision_model, output_dir=output_dir)
    final_prompt = f"{SYSTEM_INSTRUCTIONS}\n\nUser query: {refined_query}"
    response = agent.run(final_prompt)
    print(response)
    return 0


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.build_selection_board > 0:
        return run_selection_board_flow(
            agent_model=args.agent_model,
            vision_model=args.vision_model,
            output_dir=output_dir,
            user_request=args.query,
            board_size=args.build_selection_board,
            selected_index=args.selected_index,
            interactive_pick=args.interactive_pick,
            run_refined_round=args.run_refined_round,
        )

    agent = build_agent(
        agent_model=args.agent_model,
        vision_model=args.vision_model,
        output_dir=output_dir,
    )
    final_prompt = f"{SYSTEM_INSTRUCTIONS}\n\nUser query: {args.query}"
    response = agent.run(final_prompt)
    print(response)
    return 0


if __name__ == "__main__":
    sys.exit(main())
