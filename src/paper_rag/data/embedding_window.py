from typing import Any

from tqdm import tqdm

from paper_rag.schemas import EmbeddingWindow, SentenceTokenChunk


def _to_list(value: Any) -> list:
    """Convert tokenizer output to a regular Python list."""

    if hasattr(value, "tolist"):
        value = value.tolist()

    if not isinstance(value, list):
        value = list(value)

    return value


def _tokenize(
    tokenizer: Any,
    text: str,
    add_special_tokens: bool,
    return_offsets_mapping: bool = False,
) -> tuple[list[int], list[tuple[int, int]] | None]:
    """Tokenize without truncation and optionally return character offsets."""

    encoded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        truncation=False,
        return_offsets_mapping=return_offsets_mapping,
    )

    input_ids = _to_list(encoded["input_ids"])

    if input_ids and isinstance(input_ids[0], list):
        raise ValueError("Expected a single unbatched tokenizer input.")

    token_ids = [int(token_id) for token_id in input_ids]

    if not return_offsets_mapping:
        return token_ids, None

    raw_offsets = _to_list(encoded["offset_mapping"])

    if raw_offsets and isinstance(raw_offsets[0], list) and len(raw_offsets[0]) != 2:
        raise ValueError("Invalid tokenizer offset mapping.")

    offsets = [(int(offset[0]), int(offset[1])) for offset in raw_offsets]

    if len(token_ids) != len(offsets):
        raise ValueError("Token IDs and offset mapping lengths differ.")

    for start, end in offsets:
        if not 0 <= start < end <= len(text):
            raise ValueError(
                f"Invalid token offset: ({start}, {end}) for text length {len(text)}."
            )

    return token_ids, offsets


def _get_character_bounds(
    text: str,
    offsets: list[tuple[int, int]],
    token_start: int,
    token_end: int,
) -> tuple[int, int]:
    """Convert token boundaries to character boundaries."""

    start_char = 0 if token_start == 0 else offsets[token_start][0]

    end_char = len(text) if token_end == len(offsets) else offsets[token_end - 1][1]

    if not 0 <= start_char < end_char <= len(text):
        raise ValueError("Invalid embedding window character boundaries.")

    return start_char, end_char


def _count_model_input_tokens(
    tokenizer: Any,
    text: str,
) -> int:
    """Count actual model input tokens, including special tokens."""

    token_ids, _ = _tokenize(
        tokenizer=tokenizer,
        text=text,
        add_special_tokens=True,
    )

    return len(token_ids)


def build_embedding_windows(
    chunks: list[SentenceTokenChunk],
    tokenizer: Any,
    max_sequence_length: int,
    overlap_tokens: int = 32,
    show_progress_bar: bool = True,
) -> list[EmbeddingWindow]:
    """Build token-limited embedding windows from complete parent chunks.

    Parent chunks remain unchanged.

    Long parent chunks are divided into overlapping token windows.
    Each window retains its parent chunk ID and character offsets.

    The actual model input length, including special tokens, is
    checked before a window is accepted.
    """

    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive.")

    if not chunks:
        return []

    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer is required for reliable offset mapping.")

    if not callable(getattr(tokenizer, "num_special_tokens_to_add", None)):
        raise TypeError("Tokenizer cannot calculate special token count.")

    special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))

    payload_budget = max_sequence_length - special_tokens

    if payload_budget <= 0:
        raise ValueError("Maximum sequence length is too small for special tokens.")

    if not 0 <= overlap_tokens < payload_budget:
        raise ValueError(
            "overlap_tokens must be non-negative and smaller "
            "than the available token budget."
        )

    windows: list[EmbeddingWindow] = []

    iterator = tqdm(
        chunks,
        desc="Building embedding windows",
        disable=not show_progress_bar,
    )

    for chunk in iterator:
        if not chunk.text.strip():
            raise ValueError(f"Parent chunk {chunk.chunk_id} contains empty text.")

        # Tokenize the complete parent chunk without truncation.
        token_ids, offsets = _tokenize(
            tokenizer=tokenizer,
            text=chunk.text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        if not token_ids or offsets is None:
            raise ValueError(f"Cannot tokenize parent chunk {chunk.chunk_id}.")

        total_tokens = len(token_ids)
        token_start = 0
        window_index = 0

        while token_start < total_tokens:
            token_end = min(
                token_start + payload_budget,
                total_tokens,
            )

            # Reduce the window if re-tokenizing its exact text
            # would exceed the model's maximum input length.
            while token_end > token_start:
                start_char, end_char = _get_character_bounds(
                    text=chunk.text,
                    offsets=offsets,
                    token_start=token_start,
                    token_end=token_end,
                )

                window_text = chunk.text[start_char:end_char]

                actual_token_count = _count_model_input_tokens(
                    tokenizer=tokenizer,
                    text=window_text,
                )

                if actual_token_count <= max_sequence_length:
                    break

                token_end -= 1

            if token_end <= token_start:
                raise ValueError(
                    f"Cannot construct a valid embedding window "
                    f"for parent chunk {chunk.chunk_id}."
                )

            # Record the accepted window.
            windows.append(
                EmbeddingWindow(
                    window_id=(f"{chunk.chunk_id}:embedding:{window_index}"),
                    parent_chunk_id=chunk.chunk_id,
                    paper_id=chunk.paper_id,
                    split=chunk.split,
                    section_index=chunk.section_index,
                    section_name=chunk.section_name,
                    window_index=window_index,
                    text=window_text,
                    start_char=start_char,
                    end_char=end_char,
                    token_start=token_start,
                    token_end=token_end,
                    token_count=actual_token_count,
                )
            )

            window_index += 1

            # The entire parent chunk has been covered.
            if token_end >= total_tokens:
                break

            # Move forward while retaining the configured overlap.
            next_start = token_end - overlap_tokens

            if next_start <= token_start:
                raise RuntimeError("Embedding window failed to make forward progress.")

            token_start = next_start

    # Check window ID uniqueness.
    window_ids = [window.window_id for window in windows]

    if len(window_ids) != len(set(window_ids)):
        raise ValueError("Duplicate embedding window IDs detected.")

    return windows
