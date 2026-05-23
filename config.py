# -*- coding: utf-8 -*-
"""
Central config for the portfolio app.
Edit models, profile copy, and defaults here — secrets stay in .env / Streamlit secrets.
"""

import os
from typing import List

# --- App ---

PAGE_TITLE = "Huy Nguyen Portfolio"
PAGE_ICON = "👨‍💻"
PAGE_LAYOUT = "wide"

NAV_PAGES = ["Home", "Search", "Chat", "YouTube Summary"]

SITE_PASSWORD_KEY = "SITE_PASSWORD"

# --- Profile ---

PROFILE_IMAGE = os.path.join(os.path.dirname(__file__), "assets", "images", "IMG_0399.jpg")

PROFILE_INFO = {
    "name": "Huy Nguyen",
    "tagline": "AI engineer. I build things that ship.",
    "about": [
        "Came to the U.S. from Vietnam in 2014. Spent 10 years in the nail industry and built a salon business from scratch.",
        "Now AI Engineer at Confie — Python, Snowflake, production ML. Remote from Florida. I lead a small team.",
    ],
    "featured_work": {
        "title": "Enterprise Document QC",
        "org": "Confie",
        "period": "2025 — present",
        "summary": (
            "Automated insurance document quality control. In production — "
            "98%+ accuracy, ~90% less manual review."
        ),
        "details": [
            "Lead a small engineering team; main technical contact for ops and leadership",
            "Own AI governance and architecture for compliance and scale",
        ],
    },
    "site_tools": [
        ("Search", "Web + YouTube research, summarized."),
        ("Chat", "Multi-model chat with web search."),
        ("YouTube Summary", "Paste a link, get a summary."),
    ],
    "linkedin_url": "",
}

# --- Models (OpenRouter slugs) ---
# Gemini *-latest aliases (~ prefix) auto-resolve to the newest version on OpenRouter.

GEMINI_PRO = "~google/gemini-pro-latest"
GEMINI_FLASH = "~google/gemini-flash-latest"
GEMINI_FLASH_LITE = "google/gemini-3.1-flash-lite"  # no lite-latest alias on OpenRouter yet

ALL_MODELS: List[str] = [
    "openai/gpt-5.5",
    "anthropic/claude-opus-4.7",
    "anthropic/claude-sonnet-4.6",
    GEMINI_PRO,
    GEMINI_FLASH,
    GEMINI_FLASH_LITE,
    "x-ai/grok-4.3",
]

# Per-app model settings: options shown in UI + default selection.
APP_MODELS = {
    "search": {
        "options": ALL_MODELS,
        "default": GEMINI_FLASH,
    },
    "chat": {
        "options": ALL_MODELS,
        "default": GEMINI_FLASH,
    },
    "youtube": {
        "options": ALL_MODELS,
        "default": GEMINI_FLASH,
    },
    "chat_web_search": {
        "default": GEMINI_FLASH,
    },
}


def get_app_model_options(app: str) -> List[str]:
    """
    Input: App key (search, chat, youtube, chat_web_search).
    Process: Returns the model list for that app, or empty list if app has no options.
    Output: List of OpenRouter model slugs.
    """
    cfg = APP_MODELS[app]
    return cfg.get("options", [])


def get_app_default_model(app: str) -> str:
    """
    Input: App key (search, chat, youtube, chat_web_search).
    Process: Returns the configured default model for that app.
    Output: OpenRouter model slug.
    """
    cfg = APP_MODELS[app]
    if "default" in cfg:
        return cfg["default"]
    options = cfg.get("options", [])
    return options[0] if options else GEMINI_FLASH


def get_model_select_index(app: str, model: str | None = None) -> int:
    """
    Input: App key and optional current model slug.
    Process: Finds a safe st.selectbox index for the app model list.
    Output: Zero-based index into that app's options.
    """
    options = get_app_model_options(app)
    if not options:
        return 0
    target = model if model in options else get_app_default_model(app)
    try:
        return options.index(target)
    except ValueError:
        return 0
