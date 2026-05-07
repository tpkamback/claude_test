# VLMEval MME Implementation Reference

Source repositories consulted:
- https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/dataset/image_yorn.py
- https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/dataset/utils/yorn.py

---

## 1. Yes/No Extraction

### VLMEval (`YOrN_Extraction` in `yorn.py`)

```python
def YOrN_Extraction(output):
    s = output.lower()
    words = process_punctuation(s).split()
    if 'yes' in words and 'no' not in words:
        return 'Yes'
    if 'yes' not in words and 'no' in words:
        return 'No'
    return 'Unknown'
```

### Our implementation (`extract_yesno` in `mme_eval.py`)

```python
def extract_yesno(response: str) -> str:
    s = response.lower()
    words = _process_punctuation(s).split()
    has_yes = "yes" in words
    has_no  = "no"  in words
    if has_yes and not has_no:
        return "Yes"
    if has_no  and not has_yes:
        return "No"
    return "Unknown"
```

**Verdict: Identical logic.** Both lowercase, strip punctuation, split on whitespace,
and use mutual exclusion to decide Yes/No/Unknown.

---

## 2. Score Calculation

### VLMEval (`MME_rating` in `yorn.py`)

```python
def acc(key, mode='normal'):
    res = stats[key]
    values = []
    for val in res.values():
        if mode == 'normal':
            values.extend(val)          # per-question accuracy
        elif mode == 'plus':
            values.append(val[0]*val[1])  # both Q correct for same image
    return np.mean(values) * 100

scores[k] = acc(k) + acc(k, 'plus')    # task_score = accuracy + accuracy_plus
```

The MME benchmark pairs two questions per image (a positive and a negative).
`accuracy_plus` requires **both** questions for the same image to be correct.

### Our implementation

```python
# accuracy      = correct / total * 100
# accuracy_plus = accuracy  (toy dataset has 1 question per image)
task_score = accuracy + accuracy   # = 2 * accuracy
```

**Difference noted:** VLMEval's real MME uses question **pairs** (two questions per
image), so `accuracy_plus` can be lower than `accuracy`. Our toy dataset has only
one question per image, so `accuracy_plus == accuracy` and `task_score = 2 * accuracy`.

To fully replicate VLMEval semantics, each image would need a paired
positive+negative question. The `dataset.py` design could be extended to add pairs,
but for pipeline validation this single-question approach is sufficient.

---

## 3. Prompt Format

### VLMEval (`image_yorn.py` → `image_base.py :: build_prompt`)

Messages list:
```python
[
    {"type": "image", "value": "<image_path>"},
    {"type": "text",  "value": "<question>"},  # no extra suffix for MME
]
```

For AMBER dataset only, VLMEval appends `"\nPlease answer yes or no."`.
For **MME**, the questions already contain "Please answer yes or no." in the
benchmark data itself, so no suffix is added.

### Our implementation

```python
_PROMPT_SUFFIX = "Answer the question using a single word or phrase."

def build_prompt_text(question: str) -> str:
    return f"{question}\n{_PROMPT_SUFFIX}"
```

**Minor difference:** We append `"Answer the question using a single word or phrase."`
as a suffix (following general VLMEval convention for short-answer tasks).
VLMEval's MME does NOT add a suffix because the MME questions already end with
"Please answer yes or no." The questions in our toy dataset also include that phrase,
so the extra suffix is redundant but harmless.

---

## 4. Model Input Construction

### VLMEval approach

VLMEval uses a model-specific `generate()` wrapper that internally:
1. Calls `build_prompt()` to get messages list
2. Passes messages through the model's `chat()` or `generate()` method
3. Each VLM wrapper handles its own processor/tokenizer

### Our approach

We manually construct token sequences:
```
[vision_start_id] [image_pad * 64] [vision_end_id] [dummy_text_tokens...]
```

This bypasses the need for a pretrained tokenizer (unavailable without downloading
Qwen2-VL-2B-Instruct weights). The pipeline is functionally equivalent for testing.

---

## 5. Expected Score with Random Weights

With random model weights:
- Model outputs arbitrary token IDs
- Most outputs won't decode to clean "yes" or "no" → high `Unknown` rate
- When `Unknown` is predicted, it scores 0 (neither Yes nor No)
- True random Yes/No: 50% accuracy → `task_score ≈ 100` per task
- With many `Unknown` outputs: `task_score < 100`

VLMEval would also score `Unknown` as 0, so the comparison is fair.

---

## 6. Summary Table

| Aspect                | VLMEval                          | Our Implementation              | Match? |
|-----------------------|----------------------------------|---------------------------------|--------|
| Yes/No extraction     | `YOrN_Extraction()`              | `extract_yesno()`               | Yes    |
| Punctuation removal   | `process_punctuation()`          | `str.translate(punctuation)`    | Yes    |
| Score formula         | `accuracy + accuracy_plus`       | `accuracy + accuracy` (approx)  | Approx |
| Question pairs        | 2 questions per image            | 1 question per image            | Diff   |
| Prompt suffix (MME)   | None (questions already have it) | Adds instruction suffix         | Minor  |
| Image format          | File path → loaded by model      | PIL.Image → ImageProcessor      | Equiv  |
| Tokenizer             | Full pretrained tokenizer        | Dummy token IDs (random model)  | N/A    |
