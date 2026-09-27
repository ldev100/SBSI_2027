# EPNI: Empirical Probabilistic Noise Injection for Brazilian Clinical Anamneses

Anonymous repository accompanying the paper **"Empirical Probabilistic Noise Injection for Brazilian Clinical Anamnesis: Learning Abbreviation Detection from Synthetic Text"**, currently under double-blind review. Author names, affiliations and links to non-anonymous resources were intentionally omitted.

> **Research artifact, not for clinical use.** The corpus, mapping tables and models in this repository were evaluated only on the anamneses described in the paper and have not undergone clinical validation. They must not support clinical decisions or process patient records in production without independent validation.

## Overview

Health professionals write anamneses under time pressure, so real Electronic Health Records contain abbreviations (e.g., *BEG* for *bom estado geral*, "good general condition") and typographical errors that models trained on clean text rarely see during training. EPNI produces noisy training data without using any real record for training:

1. **Clean narrative generation.** A Large Language Model (Gemini 2.0 Flash) writes emergency department anamneses in Brazilian Portuguese as continuous prose, with every clinical term in full form. The prompts contain no real record.
2. **Empirical Probabilistic Noise Injection.** The clean forms of the most frequent abbreviations and typos of an empirical noise taxonomy (Top-10, Top-15 or Top-20 forms per class) are replaced by their noisy counterparts, each eligible occurrence with a fixed probability (10%, 25% or 50%). Replaced tokens are labeled `ABBREV` or `TYPO`; all other tokens are `CLEAN`.
3. **Fine-tuning and evaluation.** BERTimbau and BioBERTpt are fine-tuned for token classification on the resulting text and evaluated on manually annotated real anamneses, which are not distributed (see [Real data](#real-data-not-distributed)).

The noise taxonomy comes from prior work (reference omitted for double-blind review).

### Main results

Word-level abbreviation F1 on 500 annotated real anamneses, mean over three seeds (Table 3 of the paper):

| Method | Training data or lexicon | Abbreviation F1 |
|---|---|---|
| DICT-Top20 | exact match on the 20 injected forms | 0.691 |
| EPNI, BERTimbau | synthetic text, Top-20, rate 50% | 0.802 |
| EPNI, BioBERTpt | synthetic text, Top-20, rate 25% | 0.814 |
| DICT-Full | exact match on all 1,540 abbreviation forms of the taxonomy | 0.877 |
| REAL, BERTimbau (reference) | 1,193 annotated real anamneses | 0.943 |

The EPNI models also detect abbreviations absent from the taxonomy (BERTimbau recovers 53% of these tokens). Detection relies on the uppercase form of abbreviations, and typo detection remains weak; both limitations are discussed in the paper.

## Repository structure

```
.
├── dados/
│   ├── clean_synthetic_anamneses_gold_standart.json  # 1,196 clean synthetic anamneses
│   ├── mapeamentos/
│   │   ├── template_mapeamento_top10.json           # EPNI mapping tables: noisy form -> clean form,
│   │   ├── template_mapeamento_top15.json           # for abbreviations and typos
│   │   └── template_mapeamento_top20.json
│   ├── potential_abbreviations.json                 # noise taxonomy (prior work): abbreviation forms
│   ├── potential_typos.json                         # noise taxonomy (prior work): typo forms
│   └── glossario/
│       └── ignore_list.txt                          # expanded medical vocabulary used to build the taxonomy
├── modelos-finais/
│   ├── BERTimbau_top20_rate50_seed456/              # fine-tuned BERTimbau (reference configuration)
│   └── BioBERTpt_top20_rate25_seed456/              # fine-tuned BioBERTpt (reference configuration)
├── prompt/
│   └── example_prompt.txt                           # example generation prompt (in Portuguese)
└── scripts/
    ├── injetar_ruido_ngram.py                       # EPNI: builds the CLEAN set and the nine noisy sets
    ├── fine_tuning_v2.py                            # fine-tuning grid and evaluation on the real test set
    ├── reavaliar_word_level.py                      # word-level metrics, dictionary baselines, paired bootstrap
    └── medir_faltantes.py                           # auxiliary analyses reported in the paper
```

Folder and file names are in Portuguese: *dados* (data), *mapeamentos* (mappings), *glossário* (glossary), *modelos finais* (final models).

## Data

**Synthetic corpus** (`dados/clean_synthetic_anamneses_gold_standart.json`). A JSON list of 1,196 records in the form `{"id": 1, "texto": "..."}`, with a mean length of 67.3 whitespace-separated tokens (SD 17.5, median 70). Each generation batch was required to include the expanded forms of the most frequent abbreviations of the taxonomy, so that EPNI finds the expressions it replaces. `prompt/example_prompt.txt` shows the prompt structure.

**Mapping tables** (`dados/mapeamentos/`). One file per vocabulary level (Top-10, Top-15, Top-20):

```json
{
  "config": {"top_n": 20, "instrucao": "..."},
  "abbreviations": {
    "BEG": {"significado": "Bom Estado Geral", "count": 3340}
  },
  "typos": {
    "irratacao": {"significado": "Irritação", "count": 57}
  }
}
```

Each key is a noisy form, `significado` is the clean form it replaces (a single word or a multi-word expression, matched case-insensitively) and `count` is its frequency in the source corpus.

**Noise taxonomy** (`dados/potential_abbreviations.json`, `dados/potential_typos.json`). Forms listed by the taxonomy of the prior work, with their frequencies (`{"total": ..., "tokens": {form: {"count": n, ...}}}`). They are used by the DICT-Full baseline and by the partition analyses.

**Token and label format.** The generated training sets and the test set share this format; tokenization follows the regular expression `\b[\w\-]+\b|[^\w\s]`:

```json
[
  {"tokens": ["Paciente", "em", "BEG", ","], "labels": ["CLEAN", "CLEAN", "ABBREV", "CLEAN"]}
]
```

### Real data (not distributed)

The annotated test set (`test_set_real.json`), the training set of the REAL reference and the source corpus contain real clinical records. They were anonymized before use and cannot be redistributed under the Brazilian General Data Protection Law (LGPD) and the data use agreement. The scripts accept any annotated set in the format above.

## Setup

Python 3.10 or later.

```bash
pip install torch "transformers==5.5.4" accelerate datasets evaluate seqeval \
            scikit-learn pandas numpy matplotlib seaborn huggingface_hub

# fine_tuning_v2.py runs in offline mode (HF_HUB_OFFLINE=1),
# so cache the base models and the seqeval metric beforehand:
python -c "from huggingface_hub import snapshot_download as d; d('neuralmind/bert-base-portuguese-cased'); d('pucpr/biobertpt-all')"
python -c "import evaluate; evaluate.load('seqeval')"

# model weights are stored with Git LFS
git lfs install
git lfs pull
```

The released models were saved with `transformers` 5.5.4 (recorded in their `config.json`); using the same version is recommended.

## Reproducing the experiments

The scripts read and write paths relative to the working directory:

```bash
mkdir work && cd work
cp ../dados/clean_synthetic_anamneses_gold_standart.json ../dados/mapeamentos/*.json ../dados/potential_*.json .

# 1. EPNI: CLEAN set + Top-10/15/20 x rates 10/25/50% (injection seed 42)
python ../scripts/injetar_ruido_ngram.py
mv datasets_experimento_ruido_1196 datasets_experimento_ruido   # folder name expected by step 2

# 2. Fine-tuning grid: {BERTimbau, BioBERTpt} x training sets x seeds {42, 123, 456}
#    requires ./test_set_real.json (annotated real anamneses, not distributed)
python ../scripts/fine_tuning_v2.py

# 3. Word-level metrics, DICT-Top20 and DICT-Full baselines, paired bootstrap (10,000 resamples)
python ../scripts/reavaliar_word_level.py --preds-dir resultados_experimento/predicoes --filtro BERTimbau_top20_rate50
python ../scripts/reavaliar_word_level.py --preds-dir resultados_experimento/predicoes --filtro BioBERTpt_top20_rate25 \
       --out resultados_word_level_biobertpt.json
```

Step 2 writes per-run results, test-set predictions (`resultados_experimento/predicoes/`), summary tables and figures to `resultados_experimento/`, and copies the best checkpoints to `modelos_finais/`. Word-level labels are obtained by majority vote over the sub-tokens of each word, with ties resolved by the first sub-token.

Step 2 also accepts two optional inputs:

- **RANDOM controls.** Files `datasets_experimento_ruido/dataset_random_rate{10,25,50}.json`, in the token and label format above, are added to the grid when present.
- **REAL reference.** Set `REAL_DATASET=/path/to/dataset_real.json`. The script refuses to train if any document of this set is identical to a test document.

### Hyperparameters

| Setting | Value |
|---|---|
| Base models | `neuralmind/bert-base-portuguese-cased` (BERTimbau), `pucpr/biobertpt-all` (BioBERTpt) |
| Learning rate and batch size | 2e-5 and 4 |
| Epochs | up to 10, early stopping with patience 3 |
| Maximum length | 300 sub-tokens in training, 512 in evaluation |
| Weight decay | 0.01 |
| Loss | weighted cross-entropy (CLEAN 0.2, TYPO 5.0, ABBREV 15.0) |
| Validation | 20% of the synthetic anamneses, drawn anew for each seed |
| Seeds | 42, 123, 456 |

### Auxiliary analyses

`scripts/medir_faltantes.py` groups the additional analyses of the paper. Run `python ../scripts/medir_faltantes.py <command> --help` for the options of each command.

| Command | What it computes |
|---|---|
| `densidade` | share of noisy tokens in each training set |
| `particoes` | abbreviation recall by vocabulary membership (injected, listed in the taxonomy, absent from the taxonomy) |
| `lowercase` | effect of lowercasing the test set on a trained model |
| `regex` | rule-based uppercase baselines, on the whole test set and outside the Top-20 forms |
| `split-dev` | development and held-out partitions used to select the reference configurations |
| `gerar-preds` | test-set predictions of a saved checkpoint |
| `reetiquetagem` | checks of the relabeling step of the annotation |
| `top-sem-teste`, `formas-exclusivas`, `termos` | checks that require the source corpus, which is not distributed |

## Using the released models

The two checkpoints in `modelos-finais/` correspond to the reference configurations of the paper, trained with seed 456 (one of the three seeds reported). Labels: `0 = CLEAN`, `1 = TYPO`, `2 = ABBREV`.

```python
import re
from collections import Counter

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

path = "modelos-finais/BERTimbau_top20_rate50_seed456"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForTokenClassification.from_pretrained(path).eval()

text = "Paciente em BEG, ACV: RCR 2T BNF, AP: MV+ sem RA."
words = re.findall(r"\b[\w\-]+\b|[^\w\s]", text)  # same tokenization as the scripts
enc = tokenizer(words, is_split_into_words=True, truncation=True,
                max_length=512, return_tensors="pt")
with torch.no_grad():
    pred = model(**enc).logits.argmax(-1)[0].tolist()

# word label: majority over its sub-tokens, ties resolved by the first sub-token
votes = {}
for i, w in enumerate(enc.word_ids()):
    if w is not None:
        votes.setdefault(w, []).append(pred[i])

def aggregate(v):
    counts = Counter(v)
    top = max(counts.values())
    tied = [label for label, n in counts.items() if n == top]
    return tied[0] if len(tied) == 1 else v[0]

for i, word in enumerate(words):
    if i in votes:
        print(word, model.config.id2label[aggregate(votes[i])])
```

## Ethics and privacy

- The prompts, the synthetic corpus and the training data of the EPNI models contain no real clinical record.
- Real anamneses were used only for evaluation and for the REAL reference; they are not distributed.
- The artifacts are intended for research. Any clinical use requires independent validation.

## License

Code: MIT License. Synthetic corpus and mapping tables: CC BY 4.0. The fine-tuned models derive from BERTimbau and BioBERTpt and follow the licenses of their base models.
