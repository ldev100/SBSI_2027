import json
import os
import gc
import hashlib
import platform
import random
import re
import shutil
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
from matplotlib.patches import Patch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    TrainerCallback,
)
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import evaluate

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if torch.backends.mps.is_available():
    print("Apple GPU (MPS) detectada")
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    print("CUDA GPU detectada")
    DEVICE = torch.device("cuda")
else:
    print("Usando CPU")
    DEVICE = torch.device("cpu")

DATASET_DIR = Path("datasets_experimento_ruido")
OUTPUT_DIR = Path("resultados_experimento")
CLEAN_DATASET = Path("clean_synthetic_anamneses_gold_standart_tokens.json")
REAL_DATASET = Path(os.environ.get("REAL_DATASET", "datasets_experimento_ruido/dataset_real.json"))
TEST_REAL_FILE = Path("test_set_real.json")
PREDS_DIR = OUTPUT_DIR / "predicoes"

GOLD_HASH = (
    hashlib.md5(TEST_REAL_FILE.read_bytes()).hexdigest()[:10]
    if TEST_REAL_FILE.exists() else "sem-gold"
)

MANTER_CHECKPOINT = {"top20_rate50", "top20_rate25", "top20_rate10", "real"}

EXIGIR_PREDICOES = os.environ.get("EXIGIR_PREDICOES", "0") == "1"

MODELS = {
    "BERTimbau": "neuralmind/bert-base-portuguese-cased",
    "BioBERTpt": "pucpr/biobertpt-all",
}

LABEL_LIST = ["CLEAN", "TYPO", "ABBREV"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}

BATCH_SIZE = 4
EVAL_BATCH_SIZE = 2
EPOCHS = 10
PATIENCE = 3
LR = 2e-5
MAX_LEN = 300
EVAL_MAX_LEN = 512
WEIGHT_DECAY = 0.01
SEEDS = [42, 123, 456]

CLASS_WEIGHTS = torch.tensor([0.2, 5.0, 15.0])

metric = evaluate.load("seqeval")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PREDS_DIR, exist_ok=True)


def fixar_seed(seed):
    """Patch (d): fixa todas as fontes de aleatoriedade sob nosso controle."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def registrar_ambiente():
    import transformers
    info = {
        "gold_hash": GOLD_HASH,
        "gold_file": str(TEST_REAL_FILE),
        "device": str(DEVICE),
        "python": platform.python_version(),
        "plataforma": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "seeds": SEEDS,
        "class_weights": CLASS_WEIGHTS.tolist(),
        "max_len": MAX_LEN,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "models": MODELS,
    }
    with open(OUTPUT_DIR / "ambiente.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return info


def deve_manter(run_name):
    return any(tag in run_name.lower() for tag in MANTER_CHECKPOINT)


def concorre_ao_best(config_tag):
    return str(config_tag).lower().startswith("top")


def salvar_predicoes(run_name, word_ids_col, raw_preds, raw_labels, test_data):
    docs = []
    for i, entry in enumerate(test_data):
        wids = word_ids_col[i]
        pred_seq, label_seq = raw_preds[i], raw_labels[i]

        sub_pred, sub_gold, por_palavra = [], [], {}
        for pos in range(len(wids)):
            if label_seq[pos] == -100:
                continue
            sub_pred.append(int(pred_seq[pos]))
            sub_gold.append(int(label_seq[pos]))
            por_palavra.setdefault(int(wids[pos]), []).append(int(pred_seq[pos]))

        word_pred, word_gold, word_form = [], [], []
        for wid in range(len(entry["tokens"])):
            if wid not in por_palavra:
                continue  # palavra cortada pela truncagem em MAX_LEN
            contagem = Counter(por_palavra[wid])
            topo = max(contagem.values())
            empatados = [k for k, v in contagem.items() if v == topo]
            # voto majoritario, empate resolvido pelo primeiro sub-token
            word_pred.append(empatados[0] if len(empatados) == 1 else por_palavra[wid][0])
            word_gold.append(LABEL2ID[entry["labels"][wid]])
            word_form.append(entry["tokens"][wid].lower())

        docs.append({
            "sub_pred": sub_pred, "sub_gold": sub_gold,
            "word_pred": word_pred, "word_gold": word_gold, "word_form": word_form,
        })

    with open(PREDS_DIR / f"preds_{run_name}.json", "w") as f:
        json.dump(docs, f)

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = class_weights.to(DEVICE)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            loss_fn = torch.nn.CrossEntropyLoss(
                weight=self.class_weights, ignore_index=-100,
            )
        else:
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

        loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

class LiberarMemoria(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        limpar_memoria()

    def on_save(self, args, state, control, **kwargs):
        limpar_memoria()


def limpar_memoria():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_preds = []
    true_labels = []
    for pred_seq, label_seq in zip(predictions, labels):
        preds_filtered = []
        labels_filtered = []
        for p_val, l_val in zip(pred_seq, label_seq):
            if l_val != -100:
                preds_filtered.append(LABEL_LIST[p_val])
                labels_filtered.append(LABEL_LIST[l_val])
        true_preds.append(preds_filtered)
        true_labels.append(labels_filtered)

    results = metric.compute(predictions=true_preds, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def tokenize_and_align(examples, tokenizer, max_len=MAX_LEN):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_len,
    )
    labels = []
    word_ids_col = []  # Patch (b): preservado para a agregacao por palavra
    for i, label_set in enumerate(examples["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        word_ids_col.append([-1 if w is None else w for w in word_ids])
        label_ids = []
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            elif wid < len(label_set):
                label_ids.append(LABEL2ID[label_set[wid]])
            else:
                label_ids.append(-100)
        labels.append(label_ids)
    tokenized["labels"] = labels
    tokenized["word_ids_col"] = word_ids_col
    return tokenized


def carregar_dataset(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def avaliar_no_test_real(model, tokenizer, test_data, run_name=None):
    ds = Dataset.from_list(test_data)
    ds = ds.map(lambda x: tokenize_and_align(x, tokenizer, EVAL_MAX_LEN), batched=True)

    fora = sum(max(0, len(e["tokens"]) - len({w for w in wids if w is not None and w >= 0}))
               for e, wids in zip(test_data, ds["word_ids_col"]))
    if fora:
        print(f"    [!] {fora} palavras fora por truncagem em {EVAL_MAX_LEN} sub-tokens")

    data_collator = DataCollatorForTokenClassification(tokenizer)

    eval_args = TrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        report_to="none",
        fp16=False,
    )

    eval_trainer = Trainer(
        model=model,
        args=eval_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    predictions = eval_trainer.predict(ds)
    raw_preds = np.argmax(predictions.predictions, axis=2)
    raw_labels = predictions.label_ids

    if run_name:
        salvar_predicoes(run_name, ds["word_ids_col"], raw_preds, raw_labels, test_data)

    all_preds = []
    all_labels = []
    for pred_seq, label_seq in zip(raw_preds, raw_labels):
        for p_val, l_val in zip(pred_seq, label_seq):
            if l_val != -100:
                all_preds.append(p_val)
                all_labels.append(l_val)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    report = classification_report(
        all_labels, all_preds,
        labels=[0, 1, 2],
        target_names=LABEL_LIST,
        output_dict=True,
        zero_division=0,
    )

    noise_labels = [1, 2]
    p, r, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds,
        labels=noise_labels,
        average="macro",
        zero_division=0,
    )

    binary_preds = np.array([0 if pred == 0 else 1 for pred in all_preds])
    binary_labels = np.array([0 if lab == 0 else 1 for lab in all_labels])

    binary_report = classification_report(
        binary_labels, binary_preds,
        labels=[0, 1],
        target_names=["CLEAN", "NOISE"],
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])

    metricas = {
        "macro_f1_noise": float(f1),
        "macro_precision_noise": float(p),
        "macro_recall_noise": float(r),
        "clean_f1": float(report["CLEAN"]["f1-score"]),
        "clean_precision": float(report["CLEAN"]["precision"]),
        "clean_recall": float(report["CLEAN"]["recall"]),
        "typo_f1": float(report["TYPO"]["f1-score"]),
        "typo_precision": float(report["TYPO"]["precision"]),
        "typo_recall": float(report["TYPO"]["recall"]),
        "abbrev_f1": float(report["ABBREV"]["f1-score"]),
        "abbrev_precision": float(report["ABBREV"]["precision"]),
        "abbrev_recall": float(report["ABBREV"]["recall"]),
        "overall_accuracy": float(report["accuracy"]),
        "binary_noise_f1": float(binary_report["NOISE"]["f1-score"]),
        "binary_noise_precision": float(binary_report["NOISE"]["precision"]),
        "binary_noise_recall": float(binary_report["NOISE"]["recall"]),
        "binary_accuracy": float(binary_report["accuracy"]),
        "confusion_matrix": cm.tolist(),
    }

    del eval_trainer, predictions, ds
    if Path("./tmp_eval").exists():
        shutil.rmtree("./tmp_eval")
    limpar_memoria()

    return metricas

def train_and_evaluate(ds_path, model_alias, model_ckpt, test_real_data, seed):
    limpar_memoria()

    fixar_seed(seed)

    data = carregar_dataset(ds_path)

    idx = np.random.RandomState(seed).permutation(len(data))
    split = int(len(data) * 0.8)

    ds = DatasetDict({
        "train": Dataset.from_list([data[i] for i in idx[:split]]),
        "val": Dataset.from_list([data[i] for i in idx[split:]]),
    })

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    tokenized_ds = ds.map(lambda x: tokenize_and_align(x, tokenizer), batched=True)

    del data, ds, idx
    limpar_memoria()

    model = AutoModelForTokenClassification.from_pretrained(
        model_ckpt,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(DEVICE)

    config_tag = Path(ds_path).stem.replace("dataset_", "")
    run_name = f"{model_alias}_{config_tag}_seed{seed}"

    args = TrainingArguments(
        output_dir=f"./checkpoints/{run_name}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,
        fp16=False,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = WeightedTrainer(
        class_weights=CLASS_WEIGHTS,
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["val"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE), LiberarMemoria()],
    )

    print(f"\n>>> [{run_name}]")
    print(f"    Treino: {len(tokenized_ds['train'])} | Val: {len(tokenized_ds['val'])} | Test real: {len(test_real_data)}")

    trainer.train()

    try:
        tokenizer.save_pretrained(args.output_dir)
    except Exception as e:
        print(f"    [!] nao foi possivel salvar o tokenizer: {e}")

    val_results = trainer.evaluate()
    print(f"    Val F1 (sintetico): {val_results.get('eval_f1', 0):.4f}")

    del trainer
    limpar_memoria()

    print(f"    Avaliando no test set REAL...")
    model.cpu()
    limpar_memoria()
    test_metricas = avaliar_no_test_real(model, tokenizer, test_real_data, run_name=run_name)

    print(f"    -> ABBREV F1:        {test_metricas['abbrev_f1']:.4f}")
    print(f"    -> TYPO F1:          {test_metricas['typo_f1']:.4f}")
    print(f"    -> Macro F1 (noise): {test_metricas['macro_f1_noise']:.4f}")
    print(f"    -> Binary F1:        {test_metricas['binary_noise_f1']:.4f}")

    resultado = {
        "config": config_tag,
        "model": model_alias,
        "seed": seed,
        "gold_hash": GOLD_HASH,
        "val_f1_sintetico": val_results.get("eval_f1", 0),
        **test_metricas,
    }

    result_file = OUTPUT_DIR / f"res_{run_name}.json"
    with open(result_file, "w") as f:
        json.dump(resultado, f, indent=2)

    del model, tokenizer, tokenized_ds, data_collator, args, val_results
    limpar_memoria()

    return resultado

def main():
    if not TEST_REAL_FILE.exists():
        print("=" * 60)
        print(f"ERRO: {TEST_REAL_FILE} nao encontrado!")
        print("=" * 60)
        return

    print("[1/4] Carregando test set real...")
    test_real_data = carregar_dataset(TEST_REAL_FILE)
    n_tokens = sum(len(e["tokens"]) for e in test_real_data)
    print(f"       {len(test_real_data)} anamneses reais, {n_tokens} tokens")
    print(f"       gold_hash = {GOLD_HASH}")
    registrar_ambiente()

    print("[2/4] Listando datasets de treino...")
    train_datasets = []

    if CLEAN_DATASET.exists():
        train_datasets.append(("CLEAN", str(CLEAN_DATASET)))
        print(f"       [ok] CLEAN: {CLEAN_DATASET}")

    random_files = sorted(DATASET_DIR.glob("dataset_random_rate*.json"))
    if random_files:
        for f in random_files:
            config_name = f.stem.replace("dataset_", "").upper()
            train_datasets.append((config_name, str(f)))
            print(f"       [ok] {config_name}: {f}")
    else:
        print(f"       [!!] RANDOM nao encontrado.")

    for f in sorted(DATASET_DIR.glob("dataset_top*.json")):
        config_name = f.stem.replace("dataset_", "").upper()
        train_datasets.append((config_name, str(f)))
        print(f"       [ok] {config_name}: {f}")

    if REAL_DATASET.exists():
        # Salvaguarda: nenhum documento do REAL pode ser identico a um do teste.
        real_data = carregar_dataset(REAL_DATASET)
        assinaturas_teste = {tuple(e["tokens"]) for e in test_real_data}
        sobrepostos = sum(1 for e in real_data if tuple(e["tokens"]) in assinaturas_teste)
        if sobrepostos:
            raise SystemExit(
                f"[!!] {sobrepostos} anamneses do REAL sao identicas a anamneses do teste. "
                f"O REAL precisa ser disjunto do teste; corrija o arquivo antes de treinar."
            )
        train_datasets.append(("REAL", str(REAL_DATASET)))
        print(f"       [ok] REAL: {REAL_DATASET} ({len(real_data)} anamneses, disjunto do teste)")
    else:
        print(f"       [--] REAL nao encontrado em {REAL_DATASET} (ignorado)")

    total_runs = len(train_datasets) * len(MODELS) * len(SEEDS)
    print(f"\n       Total: {len(train_datasets)} configs x {len(MODELS)} modelos x {len(SEEDS)} seeds = {total_runs} runs")
    print(f"       Tempo estimado: {total_runs * 10 // 60}-{total_runs * 15 // 60} horas")

    print("[3/4] Executando experimentos...")
    todos_resultados = []
    runs_ok = 0
    runs_erro = 0
    runs_pulados = 0

    descartados = 0
    for res_file in sorted(OUTPUT_DIR.glob("res_*.json")):
        try:
            with open(res_file, "r") as f:
                resultado_anterior = json.load(f)
        except Exception:
            continue
        if "model" not in resultado_anterior or "macro_f1_noise" not in resultado_anterior:
            continue
        if resultado_anterior.get("gold_hash") != GOLD_HASH:
            descartados += 1  # Patch (a): calculado sobre outro gold
            continue
        todos_resultados.append(resultado_anterior)

    if todos_resultados:
        print(f"       Carregados {len(todos_resultados)} resultados anteriores")
    if descartados:
        print(f"       [!] {descartados} resultados ignorados: gold diferente do atual")

    melhor_por_modelo = {}

    for r in todos_resultados:
        try:
            if not concorre_ao_best(r["config"]):
                continue 
            alias = r["model"]
            f1 = r["macro_f1_noise"]
            rname = f"{alias}_{r['config']}_seed{r['seed']}"
            if alias not in melhor_por_modelo or f1 > melhor_por_modelo[alias]["f1"]:
                melhor_por_modelo[alias] = {"f1": f1, "run_name": rname}
        except KeyError:
            continue

    for config_name, ds_path in train_datasets:
        for model_alias, model_ckpt in MODELS.items():
            for seed in SEEDS:
                config_tag = Path(ds_path).stem.replace("dataset_", "")
                run_name = f"{model_alias}_{config_tag}_seed{seed}"

                result_file = OUTPUT_DIR / f"res_{run_name}.json"
                if result_file.exists():
                    try:
                        anterior = json.load(open(result_file))
                    except Exception:
                        anterior = {}
                    mesmo_gold = anterior.get("gold_hash") == GOLD_HASH
                    tem_preds = (PREDS_DIR / f"preds_{run_name}.json").exists()
                    if mesmo_gold and (tem_preds or not EXIGIR_PREDICOES):
                        obs = "" if tem_preds else ", sem predicoes"
                        print(f"\n    [PULANDO] {run_name} (ja completado com este gold{obs})")
                        runs_pulados += 1
                        continue
                    motivo = "gold diferente" if not mesmo_gold else "predicoes ausentes"
                    print(f"\n    [REFAZENDO] {run_name} ({motivo})")

                try:
                    resultado = train_and_evaluate(
                        ds_path, model_alias, model_ckpt, test_real_data, seed
                    )
                    todos_resultados.append(resultado)
                    runs_ok += 1

                    f1_atual = resultado["macro_f1_noise"]

                    if not concorre_ao_best(config_tag):
                        ckpt_dir = Path(f"./checkpoints/{run_name}")
                        if ckpt_dir.exists() and not deve_manter(run_name):
                            shutil.rmtree(ckpt_dir)
                    elif model_alias not in melhor_por_modelo or f1_atual > melhor_por_modelo[model_alias]["f1"]:
                        if model_alias in melhor_por_modelo:
                            anterior_run = melhor_por_modelo[model_alias]["run_name"]
                            old_dir = Path(f"./checkpoints/{anterior_run}")
                            if old_dir.exists() and not deve_manter(anterior_run):
                                shutil.rmtree(old_dir)
                        melhor_por_modelo[model_alias] = {"f1": f1_atual, "run_name": run_name}
                        print(f"    ** Novo melhor {model_alias}: {run_name} (F1={f1_atual:.4f})")
                    else:
                        ckpt_dir = Path(f"./checkpoints/{run_name}")
                        if ckpt_dir.exists() and not deve_manter(run_name):
                            shutil.rmtree(ckpt_dir)

                except Exception as e:
                    print(f"    [ERRO] {config_name}_{model_alias}_seed{seed}: {e}")
                    runs_erro += 1
                    limpar_memoria()
                    continue

                with open(OUTPUT_DIR / "metricas_completas.json", "w") as f:
                    json.dump(todos_resultados, f, indent=2, ensure_ascii=False)

    print(f"\n    Runs: {runs_ok} novos + {runs_pulados} anteriores")
    if runs_erro > 0:
        print(f"    Erros: {runs_erro}")

    modelos_dir = Path("modelos_finais")
    modelos_dir.mkdir(exist_ok=True)

    for model_alias, info in melhor_por_modelo.items():
        src = Path(f"./checkpoints/{info['run_name']}")
        dst = modelos_dir / f"{model_alias}_best"
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            if not deve_manter(info["run_name"]):
                shutil.rmtree(src)
            with open(dst / "origem.json", "w") as f:
                json.dump({
                    "run_name": info["run_name"],
                    "macro_f1_noise_teste": info["f1"],
                    "gold_hash": GOLD_HASH,
                    "criterio": "maior F1 macro de ruido no teste real, so entre configuracoes EPNI, "
                                "apenas para inspecao qualitativa",
                }, f, indent=2, ensure_ascii=False)
            print(f"    Modelo salvo: {dst}/ <- {info['run_name']} (F1={info['f1']:.4f})")

    ckpt_root = Path("./checkpoints")
    if ckpt_root.exists():
        for ckpt in sorted(ckpt_root.iterdir()):
            if not ckpt.is_dir() or not deve_manter(ckpt.name):
                continue
            dst = modelos_dir / ckpt.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(ckpt), str(dst))
            print(f"    Checkpoint preservado: {dst}/")
        shutil.rmtree(ckpt_root, ignore_errors=True)

    print("[4/4] Gerando tabelas e graficos...")
    gerar_outputs(todos_resultados)

    print()
    print("=" * 60)
    print(f"[OK] Concluido! {runs_ok} novos + {runs_pulados} anteriores")
    print(f"    Resultados em: {OUTPUT_DIR}/")
    print(f"    Predicoes por run em: {PREDS_DIR}/")
    print(f"    gold_hash = {GOLD_HASH} (registrado em cada res_*.json)")
    print()
    print("    Proximos passos:")
    print("      python reavaliar_word_level.py --preds-dir " + str(PREDS_DIR))
    print("      python medir_faltantes.py split-dev --preds-dir " + str(PREDS_DIR))
    print("=" * 60)

def gerar_outputs(resultados):
    if not resultados:
        print("    [!!] Sem resultados.")
        return

    df = pd.DataFrame(resultados)

    rename = {
        'clean_synthetic_anamneses_gold_standart_tokens': 'CLEAN',
        'random_rate10': 'RANDOM-10',
        'random_rate25': 'RANDOM-25',
        'random_rate50': 'RANDOM-50',
        'top10_rate10': 'TOP10-R10',
        'top10_rate25': 'TOP10-R25',
        'top10_rate50': 'TOP10-R50',
        'top15_rate10': 'TOP15-R10',
        'top15_rate25': 'TOP15-R25',
        'top15_rate50': 'TOP15-R50',
        'top20_rate10': 'TOP20-R10',
        'top20_rate25': 'TOP20-R25',
        'top20_rate50': 'TOP20-R50',
        'real': 'REAL',
    }
    df['config'] = df['config'].map(rename).fillna(df['config'])

    order = ["CLEAN",
             "RANDOM-10", "RANDOM-25", "RANDOM-50",
             "TOP10-R10", "TOP10-R25", "TOP10-R50",
             "TOP15-R10", "TOP15-R25", "TOP15-R50",
             "TOP20-R10", "TOP20-R25", "TOP20-R50",
             "REAL"]

    has_binary = "binary_noise_f1" in df.columns

    agg_kwargs = {
        "macro_f1_mean": ("macro_f1_noise", "mean"),
        "macro_f1_std": ("macro_f1_noise", "std"),
        "abbrev_f1_mean": ("abbrev_f1", "mean"),
        "abbrev_f1_std": ("abbrev_f1", "std"),
        "typo_f1_mean": ("typo_f1", "mean"),
        "typo_f1_std": ("typo_f1", "std"),
    }
    if has_binary:
        agg_kwargs["binary_f1_mean"] = ("binary_noise_f1", "mean")
        agg_kwargs["binary_f1_std"] = ("binary_noise_f1", "std")

    tabela = df.groupby(["config", "model"]).agg(**agg_kwargs).round(4)

    tabela.to_csv(OUTPUT_DIR / "tabela_principal.csv")
    print(f"    Tabela salva: tabela_principal.csv")

    latex_rows = []
    for (config, model), row in tabela.iterrows():
        line = (
            f"    {config} & {model} & "
            f"{row['abbrev_f1_mean']:.4f}$\\pm${row['abbrev_f1_std']:.4f} & "
            f"{row['typo_f1_mean']:.4f}$\\pm${row['typo_f1_std']:.4f} & "
            f"{row['macro_f1_mean']:.4f}$\\pm${row['macro_f1_std']:.4f}"
        )
        if has_binary:
            line += f" & {row['binary_f1_mean']:.4f}$\\pm${row['binary_f1_std']:.4f}"
        line += " \\\\"
        latex_rows.append(line)

    header = "Config & Model & ABBREV F1 & TYPO F1 & Macro F1"
    if has_binary:
        header += " & Binary F1"
    header += " \\\\"

    with open(OUTPUT_DIR / "tabela_latex.txt", "w") as f:
        f.write(header + "\n")
        f.write("\\hline\n")
        f.write("\n".join(latex_rows))
    print(f"    Tabela LaTeX salva: tabela_latex.txt")

    try:
        pivot = df.groupby(["config", "model"])["abbrev_f1"].mean().reset_index()
        pivot_table = pivot.pivot(index="config", columns="model", values="abbrev_f1")
        existing_order = [o for o in order if o in pivot_table.index]
        if existing_order:
            pivot_table = pivot_table.reindex(existing_order)

        fig, ax = plt.subplots(figsize=(7, 7))
        sns.heatmap(
            pivot_table, annot=True, fmt=".3f", cmap="YlOrRd",
            ax=ax, vmin=0, vmax=0.85, linewidths=0.5, linecolor='white',
            annot_kws={"size": 10},
        )
        ax.set_title("ABBREV F1 on Real Clinical Data", fontsize=13, pad=12)
        ax.set_ylabel("Training Configuration", fontsize=11)
        ax.set_xlabel("Model", fontsize=11)
        ax.tick_params(axis='y', labelsize=9)
        ax.tick_params(axis='x', labelsize=10)
        ax.axhline(y=1, color='black', linewidth=2)
        ax.axhline(y=4, color='black', linewidth=2)
        ax.axhline(y=7, color='black', linewidth=1.5)
        ax.axhline(y=10, color='black', linewidth=1.5)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "grafico_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    Heatmap salvo")
    except Exception as e:
        print(f"    [!!] Erro heatmap: {e}")

    try:
        df_no_clean = df[df['config'] != 'CLEAN']
        order_barras = [o for o in order if o != "CLEAN"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        for idx, model_name in enumerate(MODELS.keys()):
            model_df = df_no_clean[df_no_clean["model"] == model_name]
            summary = model_df.groupby("config").agg(
                f1_mean=("abbrev_f1", "mean"),
                f1_std=("abbrev_f1", "std"),
            )
            existing_barras = [o for o in order_barras if o in summary.index]
            summary = summary.reindex(existing_barras)
            colors = ["#f0ad4e" if "RANDOM" in c else "#5cb85c" for c in summary.index]

            axes[idx].barh(
                range(len(summary)), summary["f1_mean"], xerr=summary["f1_std"],
                capsize=3, color=colors, edgecolor='white', linewidth=0.5, height=0.7,
            )
            axes[idx].set_yticks(range(len(summary)))
            axes[idx].set_yticklabels(summary.index, fontsize=8)
            axes[idx].set_xlabel("ABBREV F1", fontsize=10)
            axes[idx].set_title(f"{model_name}", fontsize=12)
            axes[idx].set_xlim(0, 0.90)
            axes[idx].invert_yaxis()
            axes[idx].grid(axis='x', alpha=0.3)

        legend_elements = [
            Patch(facecolor='#f0ad4e', label='RANDOM'),
            Patch(facecolor='#5cb85c', label='EPNI'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10,
                   bbox_to_anchor=(0.5, -0.02))
        plt.suptitle("Abbreviation Detection Performance on Real Clinical Data", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "grafico_barras.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    Barras salvo")
    except Exception as e:
        print(f"    [!!] Erro barras: {e}")

    try:
        best_results = [r for r in resultados
                        if "confusion_matrix" in r and concorre_ao_best(r.get("config", ""))]
        if best_results:
            best = max(best_results, key=lambda x: x.get("abbrev_f1", 0))
            cm = np.array(best["confusion_matrix"])

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_LIST, yticklabels=LABEL_LIST,
                ax=ax, linewidths=0.5, linecolor='white',
            )
            ax.set_xlabel("Predicted", fontsize=11)
            ax.set_ylabel("True", fontsize=11)
            ax.set_title(
                f"Confusion Matrix - {best['model']} ({best['config']})",
                fontsize=12, pad=10,
            )
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "confusion_matrix_best.png", dpi=300, bbox_inches="tight")
            plt.close()
            print(f"    Confusion matrix salva")
    except Exception as e:
        print(f"    [!!] Erro confusion matrix: {e}")

    print()
    print("=" * 70)
    print("RESUMO DOS RESULTADOS")
    print("=" * 70)
    print(tabela.to_string())
    print()

    for model_name in MODELS.keys():
        model_df = df[(df["model"] == model_name) & df["config"].astype(str).str.startswith("TOP")]
        if not model_df.empty:
            best = model_df.loc[model_df["abbrev_f1"].idxmax()]
            print(f"  Melhor EPNI para {model_name}:")
            print(f"    Config:    {best['config']}")
            print(f"    ABBREV F1: {best['abbrev_f1']:.4f}")
            print(f"    TYPO F1:   {best['typo_f1']:.4f}")
            print(f"    Macro F1:  {best['macro_f1_noise']:.4f}")
            if "binary_noise_f1" in best:
                print(f"    Binary F1: {best['binary_noise_f1']:.4f}")
            print()


if __name__ == "__main__":
    main()