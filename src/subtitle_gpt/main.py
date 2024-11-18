# I want to create a script to translate the subtitles of a video
# to any languages

from typing import List, Dict, Any, Tuple
import json
import os
from openai import OpenAI
import logging
import time
from tqdm import tqdm
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler("subtitle_translation.log"),  # Output to file
    ],
)

# %%

FILE_PATH = "data/subtitles/The.Count.Of.Monte-Cristo.2024.1080p.BluRay.x264.AAC5.1-[YTS.MX].srt"
OUTPUT_DIR = "outputs/tmp"

TARGET_LANGUAGE = "CN"
SOURCE_LANGUAGE = "EN"
MOVIE_TITLE = "Le Comte de Monte-Cristo"
MOVIE_GENRE = "Drama/Historical"
LLM_MODEL = "gpt-4o-mini"
CONTEXT_LINES = 3

OUTPUT_FILE = f"outputs/{MOVIE_TITLE}-{TARGET_LANGUAGE}-EN.srt"

# Add BATCH_SIZE to the global parameters at the top of the file
BATCH_SIZE = 10

# %%

SYSTEM_PROMPT = """You are a professional subtitle translator specializing in {source_language} to {target_language} translations.
You must provide precise, culturally-appropriate translations that preserve the original meaning, tone, and emotional impact.

Content Information:
- Title: {movie_title}
- Genre: {genre}

Core Translation Guidelines:
1. BREVITY: Maximum 2 lines per subtitle, optimized for quick reading
2. CONSISTENCY: Use identical translations for:
   - Character names and titles
   - Recurring phrases
   - Technical/specific terms
3. AUTHENTICITY: Maintain the speaker's:
   - Social status indicators
   - Speech patterns
   - Emotional tone
4. TIMING: Ensure translations can be read within the original subtitle duration
5. CULTURAL ADAPTATION:
   - For names/terms requiring translation: Use format "translation (original)"
   - Adapt idioms and cultural references appropriately
   - Preserve honorifics with suitable target language equivalents

STRICT PROHIBITIONS:
× NO translator notes or explanations
× NO alterations to original meaning
× NO changes to character names unless culturally necessary
× NO additions of context not present in source"""


USER_PROMPT = """
SOURCE MOVIE LINES IN LIST: {current_lines}

CONTEXT:
↑ Previous Lines:
{previous_lines}

↓ Following Lines:
{following_lines}

Previous Existing Translations for terms and names:
{previous_translations_terms}

REQUIREMENTS:
- Provide ONLY the direct translation
- Match the original line's length and timing
- Preserve the speaker's tone and style
- Use appropriate target language punctuation

RETURN FORMAT IN JSON:
{{
    "translation": ["line1", "line2", ...],
    "translation_terms": {{
        "original1": "translation1",
        "original2": "translation2",
    }},
}}
"""

# %%


@dataclass
class TranslationJob:
    """
    A job to translate a batch of subtitles.
    """

    current_lines: List[str]
    previous_lines: List[str]
    following_lines: List[str]

    numbers: List[int]
    timestamps: List[str]
    translations: List[str] = field(default_factory=list)


def load_video_subtitles(
    file_path: str, batch_size: int = BATCH_SIZE
) -> List[TranslationJob]:
    """
    Load subtitles from an SRT file in a robust way.
    Returns a list of tuples: (subtitle_number, timestamp, text)
    """
    with open(file_path, "rt", encoding="utf-8") as f:
        content = f.read().strip()

    subtitles = []
    blocks = content.split("\n\n")

    for block in blocks:
        if not block.strip():
            continue

        lines = block.split("\n")
        if len(lines) < 3:  # Skip invalid blocks
            continue

        try:
            # Parse subtitle number
            number = lines[0].strip().lstrip("\ufeff")

            # Parse timestamp line
            timestamp = lines[1].strip()
            if " --> " not in timestamp:  # Verify timestamp format
                continue

            # Combine all remaining lines as text (handles multi-line subtitles)
            text = " ".join(line.strip() for line in lines[2:])

            subtitles.append((number, timestamp, text))

        except Exception as e:
            logging.warning(f"Error parsing subtitle block: {block}\nError: {str(e)}")
            continue

    # convert to TranslationJob, create batch of 10
    translation_jobs = []
    for i in range(0, len(subtitles), batch_size):
        batch = subtitles[i : i + batch_size]

        # Get previous CONTEXT_LINES lines
        start_idx = max(0, i - CONTEXT_LINES)
        previous_lines = [subtitle[2] for subtitle in subtitles[start_idx:i]]

        # Get following CONTEXT_LINES lines
        end_idx = min(len(subtitles), i + batch_size + CONTEXT_LINES)
        following_lines = [
            subtitle[2] for subtitle in subtitles[i + batch_size : end_idx]
        ]

        translation_jobs.append(
            TranslationJob(
                current_lines=[subtitle[2] for subtitle in batch],
                previous_lines=previous_lines,
                following_lines=following_lines,
                numbers=[subtitle[0] for subtitle in batch],
                timestamps=[subtitle[1] for subtitle in batch],
            )
        )

    return translation_jobs


def translate_subtitles(translation_jobs: List[TranslationJob]) -> List[TranslationJob]:
    # prompt the LLM

    client = OpenAI()

    translated_terms = {}

    for job in translation_jobs:
        system_prompt = SYSTEM_PROMPT.format(
            source_language=SOURCE_LANGUAGE,
            target_language=TARGET_LANGUAGE,
            movie_title=MOVIE_TITLE,
            genre=MOVIE_GENRE,
        )

        user_prompt = USER_PROMPT.format(
            current_lines=job.current_lines,
            previous_lines=job.previous_lines,
            following_lines=job.following_lines,
            previous_translations_terms=json.dumps(translated_terms),
        )

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )

        response_json = response.choices[0].message.content
        response_data = json.loads(response_json)

        job.translations = response_data["translation"]
        translated_terms.update(response_data["translation_terms"])

    return translation_jobs


def save_subtitles(translation_jobs: List[TranslationJob], output_file: str):
    with open(output_file, "wt", encoding="utf-8") as f:
        for job in translation_jobs:
            for number, timestamp, translation, current_line in zip(
                job.numbers, job.timestamps, job.translations, job.current_lines
            ):
                f.write(f"{number}\n{timestamp}\n{translation}\n{current_line}\n\n")


# %%


if __name__ == "__main__":
    translation_jobs = load_video_subtitles(FILE_PATH)
    print(translation_jobs[0])

    # for testing
    TESTING = True

    if TESTING:
        translation_jobs = translation_jobs[:10]

    # translate the subtitles
    translation_jobs = translate_subtitles(translation_jobs)

    # save the subtitles
    save_subtitles(translation_jobs, OUTPUT_FILE)
