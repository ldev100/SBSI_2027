import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

LABELS = ["CLEAN", "TYPO", "ABBREV"]
L2I = {l: i for i, l in enumerate(LABELS)}
MAX_LEN = 512  # 300 reproduz o corte do script antigo


# ---------------------------------------------------------------- inferencia

def prever(checkpoint, test_data, agregacao):
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    origem = checkpoint
    if not (Path(checkpoint) / "config.json").exists():
        subs = [d for d in Path(checkpoint).glob("checkpoint-*")
                if (d / "config.json").exists()]
        if not subs:
            raise SystemExit(f"[!] Nao achei config.json em {checkpoint}")
        checkpoint = str(max(subs, key=lambda d: int(d.name.rsplit("-", 1)[-1])))
        print(f"    checkpoint resolvido para {checkpoint}")

    model = AutoModelForTokenClassification.from_pretrained(checkpoint)
    model.eval()
    hub = {"bertimbau": "neuralmind/bert-base-portuguese-cased",
           "biobertpt": "pucpr/biobertpt-all"}
    alvos = [checkpoint] + [v for k, v in hub.items() if k in str(origem).lower()]
    tok = None
    for alvo in alvos:
        try:
            cand = AutoTokenizer.from_pretrained(alvo)
        except Exception:
            continue
        if len(cand) == model.config.vocab_size:
            tok = cand
            if alvo != checkpoint:
                print(f"    tokenizer carregado de '{alvo}'")
            break
    if tok is None:
        raise SystemExit(f"[!] Nenhum tokenizer compativel para {checkpoint}")

    docs = []
    truncados = 0

    with torch.no_grad():
        for entry in test_data:
            enc = tok(
                entry["tokens"],
                is_split_into_words=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            word_ids = enc.word_ids(batch_index=0)
            logits = model(**enc).logits[0]
            preds = logits.argmax(-1).tolist()

            sub_pred, sub_gold = [], []
            por_palavra = {}
            for pos, wid in enumerate(word_ids):
                if wid is None or wid >= len(entry["labels"]):
                    continue
                sub_pred.append(preds[pos])
                sub_gold.append(L2I[entry["labels"][wid]])
                por_palavra.setdefault(wid, []).append(preds[pos])

            n_palavras = len(entry["tokens"])
            if len(por_palavra) < n_palavras:
                truncados += n_palavras - len(por_palavra)

            word_pred, word_gold, word_form = [], [], []
            for wid in range(n_palavras):
                if wid not in por_palavra:
                    continue  # palavra cortada pela truncagem
                word_pred.append(agregar(por_palavra[wid], agregacao))
                word_gold.append(L2I[entry["labels"][wid]])
                word_form.append(entry["tokens"][wid].lower())

            docs.append({
                "sub_pred": sub_pred,
                "sub_gold": sub_gold,
                "word_pred": word_pred,
                "word_gold": word_gold,
                "word_form": word_form,
            })

    if truncados:
        print(f"    [!] {truncados} palavras ficaram fora por truncagem em {MAX_LEN} sub-tokens")
    return docs


def agregar(preds_da_palavra, modo):
    if modo == "first":
        return preds_da_palavra[0]
    if modo == "any":  # qualquer sub-token ruidoso torna a palavra ruidosa
        for p in preds_da_palavra:
            if p != L2I["CLEAN"]:
                return p
        return L2I["CLEAN"]
    # majority: voto majoritario, empate resolvido pelo primeiro sub-token
    cont = Counter(preds_da_palavra)
    topo = max(cont.values())
    empatados = [p for p, c in cont.items() if c == topo]
    if len(empatados) == 1:
        return empatados[0]
    return preds_da_palavra[0]


# ------------------------------------------------------------------ metricas

def contagens(gold, pred, classe):
    g = np.asarray(gold)
    p = np.asarray(pred)
    tp = int(np.sum((g == classe) & (p == classe)))
    fp = int(np.sum((g != classe) & (p == classe)))
    fn = int(np.sum((g == classe) & (p != classe)))
    return tp, fp, fn


def prf(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def metricas_por_classe(gold, pred, nivel):
    out = {}
    for nome in ["ABBREV", "TYPO"]:
        p, r, f = prf(*contagens(gold, pred, L2I[nome]))
        out[f"{nome.lower()}_{nivel}"] = {"P": round(p, 4), "R": round(r, 4), "F1": round(f, 4)}
    bg = [0 if x == 0 else 1 for x in gold]
    bp = [0 if x == 0 else 1 for x in pred]
    p, r, f = prf(*contagens(bg, bp, 1))
    out[f"binary_{nivel}"] = {"P": round(p, 4), "R": round(r, 4), "F1": round(f, 4)}
    return out


def matriz_confusao(gold, pred):
    m = np.zeros((3, 3), dtype=int)
    for g, p in zip(gold, pred):
        m[g][p] += 1
    return m


# ---------------------------------------------------------------- dicionarios

def carregar_formas(filepath):
   with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    formas = set()
    if isinstance(data, dict) and "tokens" in data:
        formas.update(t.lower() for t in data["tokens"])
    elif isinstance(data, list):
        for e in data:
            if isinstance(e, dict):
                t = e.get("token", e.get("word", ""))
                if t:
                    formas.add(t.lower())
            elif isinstance(e, str):
                formas.add(e.lower())
    elif isinstance(data, dict):
        formas.update(k.lower() for k in data if k not in ("total", "tokens", "config"))
    return formas


def carregar_top20(filepath):
    with open(filepath, encoding="utf-8") as f:
        m = json.load(f)
    return (set(k.lower() for k in m.get("abbreviations", {})),
            set(k.lower() for k in m.get("typos", {})))


def predizer_dicionario(docs, abbrev_set, typo_set):
    saida = []
    for d in docs:
        preds = []
        for forma in d["word_form"]:
            if forma in abbrev_set:
                preds.append(L2I["ABBREV"])
            elif forma in typo_set:
                preds.append(L2I["TYPO"])
            else:
                preds.append(L2I["CLEAN"])
        saida.append(preds)
    return saida


# ----------------------------------------------------------------- bootstrap

def bootstrap_pareado(docs, pred_a, pred_b, classe=L2I["ABBREV"], n=10000, seed=0):
    ca, cb = [], []
    for d, pa, pb in zip(docs, pred_a, pred_b):
        ca.append(contagens(d["word_gold"], pa, classe))
        cb.append(contagens(d["word_gold"], pb, classe))
    ca, cb = np.array(ca), np.array(cb)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(docs), size=(n, len(docs)))
    difs = np.empty(n)
    for i in range(n):
        sa = ca[idx[i]].sum(axis=0)
        sb = cb[idx[i]].sum(axis=0)
        difs[i] = prf(*sa)[2] - prf(*sb)[2]

    ponto = prf(*ca.sum(axis=0))[2] - prf(*cb.sum(axis=0))[2]
    lo, hi = np.percentile(difs, [2.5, 97.5])
    p = 2 * min((difs <= 0).mean(), (difs >= 0).mean())
    return {
        "diferenca": round(float(ponto), 4),
        "IC95": [round(float(lo), 4), round(float(hi), 4)],
        "p": max(float(p), 1.0 / n),
    }


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", default=[])
    ap.add_argument("--test", default="test_set_real.json")
    ap.add_argument("--top20", default="template_mapeamento_top20.json")
    ap.add_argument("--taxonomia-abbrev", default="potential_abbreviations.json")
    ap.add_argument("--taxonomia-typos", default="potential_typos.json")
    ap.add_argument("--agregacao", choices=["majority", "first", "any"], default="majority")
    ap.add_argument("--preds", default="predicoes_test_real.json")
    ap.add_argument("--preds-dir", default=None,
                    help="pasta com os preds_*.json gravados pelo fine_tuning_v2.py")
    ap.add_argument("--filtro", default="top20_rate50",
                    help="substring do nome do run a analisar dentro de --preds-dir")
    ap.add_argument("--out", default="resultados_word_level.json")
    args = ap.parse_args()

    with open(args.test, encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"[1] Test set: {len(test_data)} anamneses, "
          f"{sum(len(e['tokens']) for e in test_data)} tokens")

    if args.preds_dir:
        import glob as _glob
        arquivos = sorted(f for f in _glob.glob(str(Path(args.preds_dir) / "preds_*.json"))
                          if args.filtro in Path(f).stem)
        if not arquivos:
            print(f"[!] Nenhum preds_*.json com '{args.filtro}' em {args.preds_dir}")
            return
        por_ckpt = {Path(f).stem.replace("preds_", ""): json.load(open(f)) for f in arquivos}
        print(f"[2] Carregadas predicoes de {len(por_ckpt)} runs: {list(por_ckpt)}")
        preds_path = None
    else:
        preds_path = Path(args.preds)
    if preds_path is not None and preds_path.exists():
        print(f"[2] Reusando predicoes de {preds_path}")
        with open(preds_path, encoding="utf-8") as f:
            por_ckpt = json.load(f)
    elif preds_path is not None:
        por_ckpt = {}
        for ckpt in args.checkpoints:
            print(f"[2] Inferencia: {ckpt}")
            por_ckpt[ckpt] = prever(ckpt, test_data, args.agregacao)
        with open(preds_path, "w") as f:
            json.dump(por_ckpt, f)
        print(f"    Predicoes salvas em {preds_path}")

    top20_abbrev, top20_typo = carregar_top20(args.top20)
    full_abbrev = carregar_formas(args.taxonomia_abbrev)
    full_typo = carregar_formas(args.taxonomia_typos)
    print(f"[3] Top-20: {len(top20_abbrev)} abrev / {len(top20_typo)} typos | "
          f"Taxonomia: {len(full_abbrev)} / {len(full_typo)}")

    relatorio = {"agregacao": args.agregacao, "por_seed": {}, "dicionarios": {}, "bootstrap": {}}

    ref = next(iter(por_ckpt.values()))
    dict_top20 = predizer_dicionario(ref, top20_abbrev, top20_typo)
    dict_full = predizer_dicionario(ref, full_abbrev, full_typo)

    gold_w = [g for d in ref for g in d["word_gold"]]
    for nome, dp in [("DICT-Top20", dict_top20), ("DICT-Full", dict_full)]:
        flat = [p for doc in dp for p in doc]
        relatorio["dicionarios"][nome] = metricas_por_classe(gold_w, flat, "word")

    for ckpt, docs in por_ckpt.items():
        gold_s = [g for d in docs for g in d["sub_gold"]]
        pred_s = [p for d in docs for p in d["sub_pred"]]
        gold_w = [g for d in docs for g in d["word_gold"]]
        pred_w = [p for d in docs for p in d["word_pred"]]

        r = {}
        r.update(metricas_por_classe(gold_s, pred_s, "subtoken"))
        r.update(metricas_por_classe(gold_w, pred_w, "word"))
        r["matriz_confusao_subtoken"] = matriz_confusao(gold_s, pred_s).tolist()

        for particao, pertence in [("seen", True), ("unseen", False)]:
            tp = fn = 0
            for d in docs:
                for forma, g, p in zip(d["word_form"], d["word_gold"], d["word_pred"]):
                    if g != L2I["ABBREV"]:
                        continue
                    if (forma in top20_abbrev) is not pertence:
                        continue
                    tp += int(p == L2I["ABBREV"])
                    fn += int(p != L2I["ABBREV"])
            r[f"recall_{particao}_word"] = round(tp / (tp + fn), 4) if tp + fn else None
            r[f"n_{particao}"] = tp + fn

        relatorio["por_seed"][ckpt] = r
        print(f"    {Path(ckpt).name}: ABBREV word F1={r['abbrev_word']['F1']:.4f} "
              f"| sub-token F1={r['abbrev_subtoken']['F1']:.4f}")

        relatorio["bootstrap"][ckpt] = {
            "vs_DICT-Full": bootstrap_pareado(docs, [d["word_pred"] for d in docs], dict_full),
            "vs_DICT-Top20": bootstrap_pareado(docs, [d["word_pred"] for d in docs], dict_top20),
        }

    f1s = [r["abbrev_word"]["F1"] for r in relatorio["por_seed"].values()]
    relatorio["resumo"] = {
        "abbrev_word_F1_media": round(float(np.mean(f1s)), 4),
        "abbrev_word_F1_std": round(float(np.std(f1s, ddof=1)), 4) if len(f1s) > 1 else 0.0,
        "diferencas_vs_DICT_Full": [
            relatorio["bootstrap"][c]["vs_DICT-Full"]["diferenca"] for c in relatorio["bootstrap"]
        ],
    }

    with open(args.out, "w") as f:
        json.dump(relatorio, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 66)
    print("FRASE DA SECAO 5.4 (confira antes de colar)")
    print("=" * 66)
    d = sorted(relatorio["resumo"]["diferencas_vs_DICT_Full"])
    pior = min(relatorio["bootstrap"].values(), key=lambda b: b["vs_DICT-Full"]["IC95"][0])["vs_DICT-Full"]
    print(f"F1 medio do modelo (palavra): {relatorio['resumo']['abbrev_word_F1_media']:.3f} "
          f"+/- {relatorio['resumo']['abbrev_word_F1_std']:.3f}")
    print(f"DICT-Full (palavra):          {relatorio['dicionarios']['DICT-Full']['abbrev_word']['F1']:.3f}")
    print(f"Diferencas por semente:       {', '.join(f'{x:+.3f}' for x in d)}")
    print(f"Media das diferencas:         {np.mean(d):+.3f}")
    print(f"Pior IC95:                    [{pior['IC95'][0]:.3f}, {pior['IC95'][1]:.3f}], p = {pior['p']:.4f}")
    print()
    print(f"Relatorio completo em {args.out}")


if __name__ == "__main__":
    main()