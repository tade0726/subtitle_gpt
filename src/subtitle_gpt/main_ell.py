# %%
from openai import OpenAI

import logging
from pydantic import BaseModel, Field
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import logging
from collections import deque
from pprint import pprint
import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import ell
import json


# %%

# logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler("subtitle_translation.log"),  # Output to file
    ],
)


ell.init(store="./logdir")


# %%

# parameters

SUB_DIR = "data/subtitles/"
OUTPUT_DIR = "outputs"

TARGET_LANGUAGE = "CN"
BATCH_SIZE = 100
CONTEXT_LINES = 5

LLM_MODEL = "gpt-4o"

# %%

# data structure


class Terminology(BaseModel):
    original: str
    translation: str
    type: str = Field(
        description="Type of terminology",
        enum=["location", "people", "companies", "events", "objects"],
    )


class Translation(BaseModel):
    original: str
    translation: str


class Translations(BaseModel):
    terminologies: List[Terminology] = Field(
        description="Dictionary mapping original terms (names, locations, etc.) to their translations",
    )
    translations: List[Translation] = Field(
        description="List of translated subtitle lines in chronological order",
    )


class TranslationJob(BaseModel):
    current_lines: List[str]
    previous_lines: List[str]
    following_lines: List[str]
    numbers: List[int]
    timestamps: List[str]


class SubtitleParsingError(Exception):
    """Custom exception for subtitle parsing errors."""

    pass


# %%


def parse_subtitle_block(block: str) -> Optional[Tuple[int, str, str]]:
    """
    Parse a single subtitle block into its components.

    Args:
        block: Raw subtitle block text

    Returns:
        Tuple of (number, timestamp, text) if successful, None if invalid
    """
    lines = block.strip().split("\n")
    if len(lines) < 3:
        return None

    try:
        # Remove BOM and whitespace from number and convert to int
        number = int(lines[0].strip().lstrip("\ufeff"))

        # Validate timestamp format
        timestamp = lines[1].strip()
        if " --> " not in timestamp:
            return None

        # Join multiple text lines, removing any empty lines
        text_lines = [line.strip() for line in lines[2:] if line.strip()]
        text = " ".join(text_lines)

        return (number, timestamp, text)
    except (ValueError, Exception) as e:
        logging.debug(f"Failed to parse block: {block!r}, error: {str(e)}")
        return None


def load_video_subtitles(
    file_path: str | Path,
    batch_size: int = BATCH_SIZE,
    context_lines: int = CONTEXT_LINES,
) -> List[TranslationJob]:
    """
    Load and parse subtitles from an SRT file, returning batched translation jobs.

    Args:
        file_path: Path to the SRT file
        batch_size: Number of subtitles per translation job
        context_lines: Number of context lines before and after each batch

    Returns:
        List of TranslationJob objects containing batched subtitles with context

    Raises:
        FileNotFoundError: If the subtitle file doesn't exist
        SubtitleParsingError: If the file is empty or severely malformed
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Subtitle file not found: {file_path}")

    try:
        with open(file_path, "rt", encoding="utf-8") as f:
            content = f.read().strip()
    except UnicodeDecodeError:
        # Fallback to different encodings commonly used in subtitles
        for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
            try:
                with open(file_path, "rt", encoding=encoding) as f:
                    content = f.read().strip()
                break
            except UnicodeDecodeError:
                continue
        else:
            raise SubtitleParsingError(
                "Failed to decode subtitle file with known encodings"
            )

    if not content:
        raise SubtitleParsingError("Empty subtitle file")

    # Parse all subtitle blocks
    subtitles = []
    blocks = content.split("\n\n")

    for block in blocks:
        if result := parse_subtitle_block(block):
            subtitles.append(result)

    if not subtitles:
        raise SubtitleParsingError("No valid subtitles found in file")

    return create_translation_jobs(subtitles, batch_size, context_lines)


def create_translation_jobs(
    subtitles: List[tuple[int, str, str]],
    batch_size: int = BATCH_SIZE,
    context_lines: int = CONTEXT_LINES,
) -> List[TranslationJob]:
    """
    Create batched translation jobs from parsed subtitles with context windows.

    Args:
        subtitles: List of (number, timestamp, text) tuples
        batch_size: Number of subtitles per translation job
        context_lines: Number of context lines before and after each batch

    Returns:
        List of TranslationJob objects containing batched subtitles with context
    """
    if not subtitles:
        return []

    translation_jobs = []
    for i in range(0, len(subtitles), batch_size):
        batch = subtitles[i : i + batch_size]

        # Get context windows
        start_idx = max(0, i - context_lines)
        end_idx = min(len(subtitles), i + batch_size + context_lines)

        translation_jobs.append(
            TranslationJob(
                current_lines=[s[2] for s in batch],
                previous_lines=[s[2] for s in subtitles[start_idx:i]],
                following_lines=[s[2] for s in subtitles[i + batch_size : end_idx]],
                numbers=[s[0] for s in batch],
                timestamps=[s[1] for s in batch],
            )
        )

    return translation_jobs


@ell.complex(model=LLM_MODEL, response_format=Translations)
def translate(
    lines: List[str],
    previous_lines: List[str],
    following_lines: List[str],
    previous_translations_terms: dict[str, str],
) -> Translations:
    """As a professional {TARGET_LANGUAGE} subtitle translator, your task is to translate subtitles while maintaining meaning, tone, and cultural relevance.

    [KEY REQUIREMENTS]
    • Match the original tone and emotion (formal/casual)
    • Use appropriate {TARGET_LANGUAGE} honorifics
    • Ensure dialogue flows naturally within context
    • Adapt cultural references for {TARGET_LANGUAGE} audience
    • Maintain consistency with provided TERMINOLOGY
    • Reflect character relationships accurately
    • Ensure translations have the same numbers of the original subtitlescle

    [OUTPUT FORMAT]
    1. terms: List of term translations, only extracting characters names and locations
    2. lines: List of translated subtitles
    """

    return f"""[SOURCE]
Lines to translate:
{json.dumps(lines, indent=2, ensure_ascii=False)}

[TARGET LANGUAGE]
{TARGET_LANGUAGE}

[CONTEXT]
Previous lines:
{json.dumps(previous_lines, indent=2, ensure_ascii=False)}

Following lines:
{json.dumps(following_lines, indent=2, ensure_ascii=False)}

[EXISTING TERMINOLOGY TRANSLATIONS]
Previously translated terms:
{json.dumps(previous_translations_terms, indent=2, ensure_ascii=False)}
"""


def translate_subtitles(jobs: List[TranslationJob], output_path: str):
    """
    Translate subtitles and write them to a file in SRT format with both translated and original text.
    """
    previous_translations_terms = {}
    all_translations = []

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for job in jobs:
        # Get translation messages
        translation_msg: Translations = translate(
            lines=job.current_lines,
            previous_lines=job.previous_lines,
            following_lines=job.following_lines,
            previous_translations_terms=json.dumps(previous_translations_terms),
        )

        # Parse the response into our model
        translation_parsed = translation_msg.parsed

        # Update terms dictionary
        for term in translation_parsed.terminologies:
            previous_translations_terms[term.original] = term.translation

        # Store translations with metadata
        for number, timestamp, orig_text, translation in zip(
            job.numbers,
            job.timestamps,
            job.current_lines,
            translation_parsed.translations,
        ):
            all_translations.append(
                (number, timestamp, translation.original, translation.translation)
            )

    # Sort and write to file
    all_translations.sort(key=lambda x: x[0])

    with open(output_path, "w", encoding="utf-8") as f:
        for number, timestamp, orig_text, trans_text in all_translations:
            f.write(f"{number}\n")
            f.write(f"{timestamp}\n")
            f.write(f"{trans_text}\n")
            f.write(f"{orig_text}\n\n")


def translate_subtitle(file_path: str, output_path: str):
    translation_jobs = load_video_subtitles(file_path)
    # extra file_name
    file_name = Path(file_path).stem
    translate_subtitles(translation_jobs, f"{output_path}/{file_name}.str")


def process_subtitle(sub_file):
    """Process a single subtitle file"""
    logging.info(f"Starting translation for: {Path(sub_file).name}")
    return translate_subtitle(sub_file, OUTPUT_DIR)


def scan_subtitles(file_dir: str):
    """Process subtitles in parallel using multiprocessing"""
    subtitle_files = glob.glob(f"{file_dir}/*.str")
    logging.info(f"Found {len(subtitle_files)} subtitle files to process")

    # Use number of CPU cores for parallel processing
    num_processes = cpu_count()
    logging.info(f"Using {num_processes} processes for parallel processing")

    with Pool(processes=num_processes) as pool:
        list(pool.imap_unordered(process_subtitle, subtitle_files))


# %%


if __name__ == "__main__":
    scan_subtitles(SUB_DIR)
