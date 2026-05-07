"""
model.py — Tiny Qwen2-VL model with random weights for MME evaluation.

Since pre-trained weights are unavailable (403 on all model hubs), we construct
a small Qwen2VLForConditionalGeneration with random initialisation.  The model
architecture is identical to the real Qwen2.5-VL-2B; only the hyper-parameters
are reduced so the model fits in memory and runs quickly on CPU.

Key design choices
------------------
- hidden_size=96, num_hidden_layers=2, num_attention_heads=2
  → head_dim = 96/2 = 48
- mrope_section=[8, 8, 8]
  → mrope_section * 2 = [8,8,8,8,8,8], sum = 48 == head_dim  ✓
- vocab_size=256 (minimal, covers yes/no token IDs used by the dummy tokenizer)
- Vision encoder: depth=2, embed_dim=32, patch_size=14, spatial_merge_size=2
- IMAGE_TOKEN_ID=100, VISION_START=99, VISION_END=101
  (all within [0, vocab_size))

VLMEval reference
-----------------
The real Qwen2-VL-2B uses the same Qwen2VLForConditionalGeneration class as
VLMEval (https://github.com/open-compass/VLMEvalKit).  VLMEval's
``Qwen2VLChat`` wrapper calls ``model.generate()`` with the processor-prepared
inputs and then post-processes the raw string output for Yes/No extraction.
We replicate the same pipeline here but with random weights.

Usage
-----
    from vlm_eval.model import build_random_model, DummyProcessor
"""

from __future__ import annotations

import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLImageProcessor
from transformers import Qwen2VLConfig, Qwen2VLTextConfig, Qwen2VLVisionConfig
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from PIL import Image

# ---------------------------------------------------------------------------
# Token ID constants
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
IMAGE_TOKEN_ID = 100
VISION_START_ID = 99
VISION_END_ID = 101
YES_TOKEN_ID = 4
NO_TOKEN_ID = 5
PAD_TOKEN_ID = 1
BOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3
UNK_TOKEN_ID = 0


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _build_text_config() -> Qwen2VLTextConfig:
    """Return a minimal Qwen2VLTextConfig compatible with the MRoPE setup.

    head_dim = hidden_size // num_attention_heads = 96 // 2 = 48.
    mrope_section * 2 must sum to head_dim:
        [8, 8, 8] * 2 = [8, 8, 8, 8, 8, 8], sum = 48  ✓
    """
    return Qwen2VLTextConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=96,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
        use_sliding_window=False,
        max_window_layers=2,
        rms_norm_eps=1e-5,
        rope_parameters={
            "rope_theta": 10000.0,
            "rope_type": "default",
            # sum(mrope_section) == head_dim // 2  →  8+8+8 = 24 = 48//2
            "mrope_section": [8, 8, 8],
        },
    )


def _build_vision_config() -> Qwen2VLVisionConfig:
    """Return a minimal Qwen2VLVisionConfig.

    hidden_size must match the text model's hidden_size (96) so that the
    visual projection layer maps cleanly.
    """
    return Qwen2VLVisionConfig(
        depth=2,
        embed_dim=32,
        hidden_size=96,        # must equal text hidden_size
        num_heads=2,
        patch_size=14,
        spatial_merge_size=2,  # merge_size**2 = 4 patches → 1 token
        temporal_patch_size=2,
        in_channels=3,
    )


def build_random_model(seed: int = 0) -> Qwen2VLForConditionalGeneration:
    """Instantiate Qwen2VLForConditionalGeneration with random weights.

    No pre-trained checkpoint is loaded.  This is intentional: the purpose
    of this codebase is to validate the MME evaluation *pipeline*, not model
    accuracy.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        An un-trained model in eval mode on CPU.
    """
    torch.manual_seed(seed)

    config = Qwen2VLConfig(
        text_config=_build_text_config(),
        vision_config=_build_vision_config(),
        image_token_id=IMAGE_TOKEN_ID,
        vision_start_token_id=VISION_START_ID,
        vision_end_token_id=VISION_END_ID,
    )
    model = Qwen2VLForConditionalGeneration(config)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Dummy tokenizer
# ---------------------------------------------------------------------------

def _build_tokenizer() -> PreTrainedTokenizerFast:
    """Build a minimal fast tokenizer backed by WordPiece.

    Special token IDs are aligned with the constants above so that the model's
    logit-based Yes/No extraction works correctly.

    Token mapping (subset):
        0  [UNK]
        1  [PAD]
        2  [BOS]
        3  [EOS]
        4  yes
        5  no
        99  [VSTART]
        100 [IMG]
        101 [VEND]
    """
    # Build a vocab large enough to cover all special IDs (0-101 + filler)
    vocab: dict[str, int] = {f"[tok{i}]": i for i in range(VOCAB_SIZE)}
    # Override with meaningful tokens
    vocab.update({
        "[UNK]": UNK_TOKEN_ID,
        "[PAD]": PAD_TOKEN_ID,
        "[BOS]": BOS_TOKEN_ID,
        "[EOS]": EOS_TOKEN_ID,
        "yes": YES_TOKEN_ID,
        "no": NO_TOKEN_ID,
        "[VSTART]": VISION_START_ID,
        "[IMG]": IMAGE_TOKEN_ID,
        "[VEND]": VISION_END_ID,
    })

    tok_backend = Tokenizer(WordPiece(vocab=vocab, unk_token="[UNK]"))
    tok_backend.pre_tokenizer = Whitespace()

    return PreTrainedTokenizerFast(
        tokenizer_object=tok_backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )


# ---------------------------------------------------------------------------
# Dummy processor
# ---------------------------------------------------------------------------

class DummyProcessor:
    """Minimal VLM processor for MME-style evaluation.

    Encodes a PIL image + question text into the tensors expected by
    Qwen2VLForConditionalGeneration:

        input_ids             LongTensor  [1, seq_len]
        pixel_values          FloatTensor [num_patches, patch_dim]
        image_grid_thw        LongTensor  [1, 3]
        mm_token_type_ids     IntTensor   [1, seq_len]  (0=text, 1=image)
        attention_mask        LongTensor  [1, seq_len]

    The sequence layout is::

        [VISION_START] [IMG_TOKEN × n] [VISION_END] [question_token × m]

    where n = grid_t × grid_h × grid_w // (spatial_merge_size ** 2).

    VLMEval reference
    -----------------
    VLMEval's Qwen2VLChat processor produces the exact same keys via
    ``qwen_vl_utils.process_vision_info`` + ``processor()``.  The token type
    ids are generated in ``processing_qwen2_vl.py``.  Our ``DummyProcessor``
    replicates this schema without requiring network access.
    """

    def __init__(self) -> None:
        self.image_processor = Qwen2VLImageProcessor()
        self.tokenizer = _build_tokenizer()
        self._merge_size = self.image_processor.merge_size  # 2

    def __call__(self, image: Image.Image, text: str) -> dict[str, torch.Tensor]:
        """Process one image–text pair into model-ready tensors.

        Args:
            image: PIL RGB image (any size; will be resized by image_processor).
            text:  Question string (not used for token encoding in this dummy
                   implementation — kept for API compatibility with VLMEval).

        Returns:
            Dict with keys: input_ids, pixel_values, image_grid_thw,
            mm_token_type_ids, attention_mask.
        """
        # --- Visual features ---
        img_feats = self.image_processor(images=[image], return_tensors="pt")
        pixel_values: torch.Tensor = img_feats["pixel_values"]
        image_grid_thw: torch.Tensor = img_feats["image_grid_thw"]  # [1, 3]

        # Number of image tokens after spatial merging
        n_img = int(image_grid_thw[0].prod() // (self._merge_size ** 2))

        # --- Token sequence ---
        # [VISION_START] + [IMAGE_TOKEN]*n_img + [VISION_END] + [yes_id, no_id]
        # The trailing "yes no" acts as a stand-in for the question tokens and
        # ensures the model produces logits we can read for yes/no scoring.
        text_ids = [YES_TOKEN_ID, NO_TOKEN_ID]
        seq = (
            [VISION_START_ID]
            + [IMAGE_TOKEN_ID] * n_img
            + [VISION_END_ID]
            + text_ids
        )

        # mm_token_type_ids: 0=text, 1=image
        ttype = (
            [0]                 # VISION_START is a text-side special token
            + [1] * n_img       # image patches
            + [0]               # VISION_END
            + [0] * len(text_ids)
        )

        input_ids = torch.tensor([seq], dtype=torch.long)
        mm_token_type_ids = torch.tensor([ttype], dtype=torch.int)
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
            "attention_mask": attention_mask,
        }
