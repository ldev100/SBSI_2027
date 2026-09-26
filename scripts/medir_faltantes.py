import argparse
import glob
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np 

LABELS = ["CLEAN", "TYPO", "ABBREV"]
L2I = {l: i for i, l in enumerate(LABELS)}
CLEAN, TYPO, ABBREV = L2I["CLEAN"], L2I["TYPO"], L2I["ABBREV"]

TOKEN_RE = re.compile(r"\b[\w\-]+\b|[^\w\s]", re.UNICODE)  # mesmo de injetar_ruido_ngram.py


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _exigir(filepath, dica=""):
    if not Path(filepath).exists():
        raise SystemExit(f"[!] Arquivo nao encontrado: {filepath}\n    {dica}".rstrip())
    return filepath


def carregar_formas(filepath):
    _exigir(filepath, "Aponte para o arquivo da taxonomia da Fase 1 "
                      "(--taxonomia-abbrev / --taxonomia-typos).")
    data = json.load(open(filepath, encoding="utf-8"))
    formas = set()
    if isinstance(data, dict) and "tokens" in data:
        formas.update(str(t).lower() for t in data["tokens"])
    elif isinstance(data, list):
        for e in data:
            if isinstance(e, dict):
                t = e.get("token", e.get("word", e.get("forma", "")))
                if t:
                    formas.add(str(t).lower())
            else:
                formas.add(str(e).lower())
    elif isinstance(data, dict):
        formas.update(str(k).lower() for k in data if k not in ("total", "tokens", "config"))
    return formas


def carregar_top20(filepath):
    _exigir(filepath, "Aponte para o template_mapeamento_top20.json (--top20).")
    m = json.load(open(filepath, encoding="utf-8"))
    return ({k.lower() for k in m.get("abbreviations", {})},
            {k.lower() for k in m.get("typos", {})})


# ------------------------------------------------------------------ 1. densidade

def cmd_densidade(args):
    arquivos = sorted(glob.glob(str(Path(args.datasets) / "*.json")))
    if args.clean and Path(args.clean).exists():
        arquivos = [args.clean] + arquivos
    print(f"{'split':26s} {'docs':>6s} {'tokens':>9s} {'ABBREV':>8s} {'TYPO':>7s} {'ruido %':>8s}")
    linhas = []
    for f in arquivos:
        data = json.load(open(f, encoding="utf-8"))
        c = Counter(l for e in data for l in e["labels"])
        tot = sum(c.values())
        pct = 100 * (c["ABBREV"] + c["TYPO"]) / tot
        nome = Path(f).stem.replace("dataset_", "")
        print(f"{nome:26s} {len(data):6d} {tot:9d} {c['ABBREV']:8d} {c['TYPO']:7d} {pct:8.2f}")
        linhas.append({"split": nome, "tokens": tot, "abbrev": c["ABBREV"],
                       "typo": c["TYPO"], "densidade_pct": round(pct, 2)})
    json.dump(linhas, open("densidade_por_split.json", "w"), indent=2)
    print("\nCompare a coluna 'ruido %' com os 13,1% do teste antes de reescrever a 4.3.2 e a 5.2.")


# ----------------------------------------------------------------- 2. lowercase

HUB_FALLBACK = {
    "bertimbau": "neuralmind/bert-base-portuguese-cased",
    "biobertpt": "pucpr/biobertpt-all",
}


def _resolver_ckpt(ckpt):
    """Aceita tanto a pasta do Trainer quanto o checkpoint-N de dentro dela."""
    p = Path(ckpt)
    if (p / "config.json").exists():
        return str(p)
    subs = [d for d in p.glob("checkpoint-*") if (d / "config.json").exists()]
    if subs:
        escolhido = max(subs, key=lambda d: int(d.name.rsplit("-", 1)[-1]))
        print(f"    checkpoint resolvido para {escolhido}")
        return str(escolhido)
    raise SystemExit(
        f"[!] Nao achei config.json em {ckpt} nem em subpastas checkpoint-*.\n"
        f"    Confira o caminho com: ls {ckpt}"
    )


def _carregar_tokenizer(ckpt, override=None, vocab_size=None, nome_original=None):
    from transformers import AutoTokenizer
    tentativas = []
    if override:
        tentativas.append(override)
    tentativas.append(str(ckpt))
    nome = str(nome_original or ckpt).lower()
    for chave, hub in HUB_FALLBACK.items():
        if chave in nome and hub not in tentativas:
            tentativas.append(hub)

    for alvo in tentativas:
        try:
            tok = AutoTokenizer.from_pretrained(alvo)
        except Exception as e:
            print(f"    [!] tokenizer de '{alvo}': {type(e).__name__}")
            continue
        if vocab_size and len(tok) != vocab_size:
            print(f"    [!] '{alvo}' tem {len(tok)} tokens, o modelo espera "
                  f"{vocab_size}. Descartado.")
            continue
        if alvo != str(ckpt):
            print(f"    tokenizer carregado de '{alvo}' (o checkpoint nao tem)")
        return tok

    raise SystemExit(
        "[!] Nenhum tokenizer compativel. Passe --tokenizer com o id do Hub "
        "usado no treino, por exemplo:\n"
        "    --tokenizer neuralmind/bert-base-portuguese-cased"
    )


def _inferir_docs(ckpt, test_data, lower=False, tokenizer_id=None, max_len=512):
    import torch
    from transformers import AutoModelForTokenClassification
    ckpt_original, ckpt = ckpt, _resolver_ckpt(ckpt)
    model = AutoModelForTokenClassification.from_pretrained(ckpt)
    model.eval()
    tok = _carregar_tokenizer(ckpt, tokenizer_id, model.config.vocab_size,
                              nome_original=ckpt_original)
    docs, truncados = [], 0
    with torch.no_grad():
        for e in test_data:
            toks = [t.lower() for t in e["tokens"]] if lower else e["tokens"]
            enc = tok(toks, is_split_into_words=True, truncation=True,
                      max_length=max_len, return_tensors="pt")
            wids = enc.word_ids(batch_index=0)
            out = model(**enc).logits[0].argmax(-1).tolist()
            sub_pred, sub_gold, por_palavra = [], [], defaultdict(list)
            for pos, w in enumerate(wids):
                if w is None or w >= len(e["labels"]):
                    continue
                sub_pred.append(int(out[pos]))
                sub_gold.append(L2I[e["labels"][w]])
                por_palavra[w].append(int(out[pos]))
            if len(por_palavra) < len(e["tokens"]):
                truncados += len(e["tokens"]) - len(por_palavra)
            word_pred, word_gold, word_form = [], [], []
            for w in range(len(e["tokens"])):
                if w not in por_palavra:
                    continue
                ps = por_palavra[w]
                c = Counter(ps); topo = max(c.values())
                emp = [k for k, v in c.items() if v == topo]
                word_pred.append(emp[0] if len(emp) == 1 else ps[0])
                word_gold.append(L2I[e["labels"][w]])
                word_form.append(e["tokens"][w].lower())
            docs.append({"sub_pred": sub_pred, "sub_gold": sub_gold,
                         "word_pred": word_pred, "word_gold": word_gold,
                         "word_form": word_form})
    if truncados:
        print(f"    [!] {truncados} palavras fora por truncagem em {max_len} sub-tokens")
    else:
        print(f"    nenhuma palavra perdida por truncagem (max_len={max_len})")
    return docs


def _achatar(docs):
    return ([g for d in docs for g in d["word_gold"]],
            [p for d in docs for p in d["word_pred"]])


def cmd_gerar_preds(args):
    test_data = json.load(open(args.test, encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = _inferir_docs(args.ckpt, test_data, tokenizer_id=args.tokenizer,
                         max_len=args.max_len)
    destino = out_dir / f"preds_{args.nome}.json"
    json.dump(docs, open(destino, "w"))
    gold, pred = _achatar(docs)
    g, p = np.array(gold), np.array(pred)
    P, R, F = prf(int(((g == ABBREV) & (p == ABBREV)).sum()),
                  int(((g != ABBREV) & (p == ABBREV)).sum()),
                  int(((g == ABBREV) & (p != ABBREV)).sum()))
    print(f"    {len(docs)} documentos -> {destino}")
    print(f"    ABBREV por palavra: P={P:.3f} R={R:.3f} F1={F:.3f}")


def cmd_lowercase(args):
    test_data = json.load(open(args.test, encoding="utf-8"))
    linhas = []
    for ckpt in args.ckpt:
        print(f"\n{ckpt}")
        res = {"ckpt": ckpt}
        for lower in [False, True]:
            gold, pred = _achatar(
                _inferir_docs(ckpt, test_data, lower, tokenizer_id=args.tokenizer,
                              max_len=args.max_len))
            g, p = np.array(gold), np.array(pred)
            tp = int(((g == ABBREV) & (p == ABBREV)).sum())
            fp = int(((g != ABBREV) & (p == ABBREV)).sum())
            fn = int(((g == ABBREV) & (p != ABBREV)).sum())
            P, R, F = prf(tp, fp, fn)
            res["minusculas" if lower else "original"] = {"P": P, "R": R, "F1": F}
            print(f"  {'minusculas' if lower else 'original  '}  "
                  f"ABBREV  P={P:.3f}  R={R:.3f}  F1={F:.3f}")
        linhas.append(res)

    if len(linhas) > 1:
        print(f"\nResumo de {len(linhas)} checkpoints (media e faixa)")
        for chave in ("original", "minusculas"):
            for m in ("P", "R", "F1"):
                v = [l[chave][m] for l in linhas]
                print(f"  {chave:10s} {m:2s}  media {np.mean(v):.3f}   faixa {min(v):.3f} a {max(v):.3f}")
    json.dump(linhas, open(args.out, "w"), indent=2)
    print(f"\nSalvo em {args.out}. A queda de recall entre original e minusculas "
          f"e o numero da Secao 5.7.")


# ------------------------------------------------- 3. particoes de conhecimento

def cmd_particoes(args):
    top20_abbrev, _ = carregar_top20(args.top20)
    taxonomia = carregar_formas(args.taxonomia_abbrev)
    print(f"Top-20: {len(top20_abbrev)} formas | taxonomia completa: {len(taxonomia)} formas\n")

    def particao(forma):
        if forma in top20_abbrev:
            return "A_top20"
        if forma in taxonomia:
            return "B_taxonomia"
        return "C_fora"

    arquivos = sorted(glob.glob(args.preds))
    if not arquivos:
        print(f"[!] Nenhum arquivo em {args.preds}")
        pasta = Path(args.preds).parent
        existentes = sorted(glob.glob(str(pasta / "preds_*.json"))) if pasta.exists() else []
        if existentes:
            print("    Disponiveis nessa pasta:")
            for f in existentes:
                print(f"      {Path(f).name}")
            print("    Ajuste o --preds.")
        else:
            print("    Nenhum preds_*.json existe ainda. Gere um a partir de um "
                  "checkpoint que voce ja tem:")
            print("      python medir_faltantes.py gerar-preds \\")
            print("          --ckpt modelos_finais/BERTimbau_best \\")
            print("          --nome BERTimbau_top20_rate50_ckptantigo")
        return

    por_particao = defaultdict(list)
    prec_fora_top20, prec_fora_tax = [], []
    n_tokens = {}

    for arq in arquivos:
        docs = json.load(open(arq))
        acertos, totais = Counter(), Counter()
        tp_ft = fp_ft = tp_fx = fp_fx = 0
        for d in docs:
            for forma, g, p in zip(d["word_form"], d["word_gold"], d["word_pred"]):
                if g == ABBREV:
                    part = particao(forma)
                    totais[part] += 1
                    acertos[part] += int(p == ABBREV)
                if p == ABBREV and forma not in top20_abbrev:
                    tp_ft += int(g == ABBREV); fp_ft += int(g != ABBREV)
                if p == ABBREV and forma not in taxonomia:
                    tp_fx += int(g == ABBREV); fp_fx += int(g != ABBREV)
        for part in ["A_top20", "B_taxonomia", "C_fora"]:
            if totais[part]:
                por_particao[part].append(acertos[part] / totais[part])
                n_tokens[part] = totais[part]
        prec_fora_top20.append(tp_ft / (tp_ft + fp_ft) if tp_ft + fp_ft else 0.0)
        prec_fora_tax.append(tp_fx / (tp_fx + fp_fx) if tp_fx + fp_fx else 0.0)
        print(f"  lido: {Path(arq).stem}")

    total = sum(n_tokens.values())
    dicts = {"A_top20": (1.0, 1.0), "B_taxonomia": (0.0, 1.0), "C_fora": (0.0, 0.0)}
    rotulo = {"A_top20": "A no Top-20", "B_taxonomia": "B na taxonomia",
              "C_fora": "C fora de tudo"}
    print(f"\n{'particao':16s} {'# tokens':>9s} {'% total':>8s} "
          f"{'recall modelo':>17s} {'DICT-Top20':>11s} {'DICT-Full':>10s}")
    linhas = {}
    for part in ["A_top20", "B_taxonomia", "C_fora"]:
        if part not in n_tokens:
            continue
        v = por_particao[part]
        media = float(np.mean(v))
        desvio = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        d20, dfull = dicts[part]
        print(f"{rotulo[part]:16s} {n_tokens[part]:9d} {100*n_tokens[part]/total:7.1f}% "
              f"{media:11.3f} ±{desvio:.3f} {d20:11.3f} {dfull:10.3f}")
        linhas[part] = {"n": n_tokens[part], "recall_modelo": round(media, 4),
                        "std": round(desvio, 4), "dict_top20": d20, "dict_full": dfull}

    print(f"\nPrecisao do modelo restrita as formas fora do Top-20:    "
          f"{np.mean(prec_fora_top20):.3f}")
    print(f"Precisao do modelo restrita as formas fora da taxonomia: "
          f"{np.mean(prec_fora_tax):.3f}")
    print("Compare com os 0.42 do regex de caixa alta, nao com a precisao geral de 0.897.")
    json.dump({"particoes": linhas,
               "precisao_fora_top20": round(float(np.mean(prec_fora_top20)), 4),
               "precisao_fora_taxonomia": round(float(np.mean(prec_fora_tax)), 4)},
              open("particoes_vocabulario.json", "w"), indent=2, ensure_ascii=False)
    print("\nSalvo em particoes_vocabulario.json")


# --------------------------------------------- 4. formas exclusivas do teste

def _texto_dos_registros(path, campo=None):
    if not Path(path).exists():
        raise SystemExit(
            f"[!] Arquivo nao encontrado: {path}\n"
            f"    --corpus precisa apontar para o arquivo real das 8.411 anamneses\n"
            f"    (o que alimentou a Fase 1). Aceita .json, .csv ou .txt.\n"
            f"    Ex.: --corpus ../fase1/anamneses.csv --campo-texto texto"
        )
    if path.endswith(".csv") or path.endswith(".tsv"):
        import csv as _csv
        sep = "\t" if path.endswith(".tsv") else ","
        with open(path, encoding="utf-8", errors="replace", newline="") as fh:
            linhas = list(_csv.DictReader(fh, delimiter=sep))
        if not linhas:
            return []
        colunas = list(linhas[0].keys())
        if campo and campo not in colunas:
            raise SystemExit(f"[!] Coluna '{campo}' nao existe. Colunas: {colunas}")
        if not campo:
            campo = max(colunas, key=lambda c: np.mean(
                [len(str(l.get(c) or "")) for l in linhas[:200]]))
            print(f"    coluna de texto inferida: '{campo}'")
        return [str(l.get(campo) or "") for l in linhas]
    if path.endswith(".json"):
        data = json.load(open(path, encoding="utf-8"))
        if isinstance(data, dict):
            data = list(data.values())
        textos = []
        for e in data:
            if isinstance(e, str):
                textos.append(e)
            elif isinstance(e, dict):
                if campo and campo in e:
                    textos.append(str(e[campo]))
                elif isinstance(e.get("tokens"), list):
                    textos.append(" ".join(map(str, e["tokens"])))
                else:
                    cand = [v for v in e.values() if isinstance(v, str) and len(v) > 40]
                    textos.append(max(cand, key=len) if cand else "")
        return textos
    return list(open(path, encoding="utf-8", errors="replace"))


def cmd_formas_exclusivas(args):
    taxonomia = carregar_formas(args.taxonomia_abbrev) | carregar_formas(args.taxonomia_typos)
    print(f"Taxonomia: {len(taxonomia)} formas")

    test_data = json.load(open(args.test, encoding="utf-8"))
    no_teste = Counter(t.lower() for e in test_data for t in e["tokens"])

    textos = _texto_dos_registros(args.corpus, args.campo_texto)
    print(f"Corpus-fonte: {len(textos)} registros")
    no_corpus = Counter()
    for texto in textos:
        no_corpus.update(t.lower() for t in TOKEN_RE.findall(texto))

    exclusivas, sem_ocorrencia = [], 0
    for forma in sorted(taxonomia):
        c_corpus = no_corpus.get(forma, 0)
        c_teste = no_teste.get(forma, 0)
        fora_do_teste = c_corpus if args.corpus_sem_teste else c_corpus - c_teste
        if c_teste == 0:
            continue  # nao afeta o dicionario neste teste
        if fora_do_teste <= 0:
            exclusivas.append(forma)
        if c_corpus == 0:
            sem_ocorrencia += 1

    json.dump(exclusivas, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"\n{len(exclusivas)} formas da taxonomia so ocorrem nos registros do teste.")
    if exclusivas:
        print("Exemplos:", ", ".join(exclusivas[:15]))
    if sem_ocorrencia:
        print(f"[!] {sem_ocorrencia} formas nao foram encontradas no corpus com esta "
              f"tokenizacao. Confira --campo-texto antes de confiar no resultado.")
    print(f"\nSalvo em {args.out}. Agora rode:")
    print(f"  python dicionario_baseline.py --modo ruidosa --excluir-formas {args.out}")
    if not args.corpus_sem_teste:
        print("\n(assumindo que o corpus-fonte inclui os 500 registros do teste; "
              "se nao incluir, repita com --corpus-sem-teste)")



# ------------------------------------------------------------ top-sem-teste

def _ranking(contagem):
    return sorted(contagem, key=lambda f: (-contagem[f], f))


def _truncamentos(formas, vocab_corpus):
    por_prefixo = defaultdict(list)
    for t in vocab_corpus:
        if len(t) > 3 and any(ord(c) > 127 for c in t):
            por_prefixo[_dobrar(t)[:4]].append(t)
    dobradas = {_dobrar(t) for t in vocab_corpus}
    marcadas = set()
    for f in formas:
        if len(f) < 4 or any(ord(c) > 127 for c in f):
            continue
       if any(f + fim in dobradas for fim in ("ao", "aos", "oes")):
            marcadas.add(f)
            continue
        for t in por_prefixo.get(f[:4], ()):
            if len(t) > len(f) and _dobrar(t).startswith(f) and ord(t[len(f)]) > 127:
                marcadas.add(f)
                break
    return marcadas


def cmd_top_sem_teste(args):
    top_ab, top_ty = carregar_top20(args.top20)
    textos = _texto_dos_registros(args.corpus, args.campo_texto)
    print(f"Corpus-fonte: {len(textos)} registros")
    no_corpus = Counter(t.lower() for texto in textos for t in TOKEN_RE.findall(texto))
    test_data = json.load(open(_exigir(args.test), encoding="utf-8"))
    no_teste = Counter(t.lower() for e in test_data for t in e["tokens"])
    ignorar = {f.strip().lower() for f in (args.ignorar or "").split(",") if f.strip()}

    resumo = {}
    for nome, arq, injetado in (("abreviacoes", args.taxonomia_abbrev, top_ab),
                                ("typos", args.taxonomia_typos, top_ty)):
        formas = carregar_formas(arq) | injetado
        fora = set(ignorar)
        if args.sem_truncamentos:
            trunc = _truncamentos(formas, no_corpus)
            if trunc & injetado:
                print(f"  [!] formas injetadas que parecem truncamento e foram mantidas  "
                      f"{', '.join(sorted(trunc & injetado))}")
            fora |= trunc - injetado
        formas = formas - (fora - injetado)
        total = {f: no_corpus.get(f, 0) for f in formas}
        sem = {f: total[f] if args.corpus_sem_teste else max(total[f] - no_teste.get(f, 0), 0)
               for f in formas}
        n = len(injetado)
        r_total, r_sem = _ranking(total), _ranking(sem)
        print(f"\n{nome.upper()}  ({len(formas)} formas no ranking, Top-{n} injetado)")
        if fora - injetado:
            maiores = sorted(fora - injetado, key=lambda f: -no_corpus.get(f, 0))[:12]
            print(f"  fora do ranking ({len(fora - injetado)})  "
                  f"{', '.join(f'{f} ({no_corpus.get(f, 0)})' for f in maiores)}"
                  f"{' ...' if len(fora - injetado) > 12 else ''}")
        print(f"  ranking no corpus inteiro reproduz o Top-{n} injetado  "
              f"{len(set(r_total[:n]) & injetado)} de {n} formas")
        print(f"  ranking sem os registros do teste                      "
              f"{len(set(r_sem[:n]) & injetado)} de {n} formas")
        for k in (10, 15, 20):
            if k <= n:
                a, b = set(r_total[:k]), set(r_sem[:k])
                if a == b:
                    print(f"    Top-{k:<2d} com e sem o teste  igual")
                else:
                    print(f"    Top-{k:<2d} com e sem o teste  DIFERENTE  "
                          f"sai {', '.join(sorted(a - b))} | entra {', '.join(sorted(b - a))}")
        saem = sorted(injetado - set(r_sem[:n]), key=lambda f: -sem[f])
        entram = [f for f in r_sem[:n] if f not in injetado]
        if saem or entram:
            print(f"  diferencas para o injetado, sem o teste  saem {', '.join(f'{f} ({sem[f]})' for f in saem) or '-'}"
                  f" | entram {', '.join(f'{f} ({sem[f]})' for f in entram) or '-'}")
        outras = [f for f in formas if f not in injetado]
        if outras:
            for rotulo, cont in (("corpus inteiro", total), ("sem o teste", sem)):
                menor = min(injetado, key=lambda f: cont[f])
                maior = max(outras, key=lambda f: cont[f])
                situacao = ("o injetado inteiro fica acima de todas as outras formas"
                            if cont[menor] > cont[maior] else
                            "empate na fronteira" if cont[menor] == cont[maior] else
                            "alguma forma de fora supera uma injetada")
                print(f"  margem, {rotulo:14s}  menor injetada {menor} ({cont[menor]}) | "
                      f"maior de fora {maior} ({cont[maior]})  ->  {situacao}")
        # Uma forma de fora que estava abaixo (ou empatada) de uma injetada no corpus
        # inteiro e passa a ficar acima dela sem o teste. So essas trocas podem
        # mudar o vocabulario por causa do teste, qualquer que seja o criterio usado
        # para excluir formas (truncamentos, usos que nao sao ruido etc.).
        trocas = sorted({(o, i) for i in injetado for o in outras
                         if total[o] <= total[i] and sem[o] > sem[i]},
                        key=lambda p: (-sem[p[0]], p[0]))
        print(f"  formas de fora que passam uma injetada so por tirar o teste  {len(trocas)}")
        for o, i in trocas[:10]:
            print(f"    {o} ({total[o]} -> {sem[o]}) passa {i} ({total[i]} -> {sem[i]})")
        zeradas = sorted(f for f in injetado if total[f] == 0)
        if zeradas:
            print(f"  [!] formas injetadas sem ocorrencia com esta tokenizacao  {', '.join(zeradas)}")
        resumo[nome] = {
            "injetado": sorted(injetado),
            "fora_do_ranking": sorted(fora - injetado),
            "top_corpus_inteiro": r_total[:n],
            "top_sem_teste": r_sem[:n],
            "iguais_ao_injetado_sem_teste": len(set(r_sem[:n]) & injetado),
            "trocas_causadas_pelo_teste": [[o, i] for o, i in trocas],
            "contagens_sem_teste": {f: sem[f] for f in r_sem[:n + 10]},
        }
    json.dump(resumo, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"\nSalvo em {args.out}.")
    if not args.corpus_sem_teste:
        print("(assumindo que o corpus-fonte inclui os 500 registros do teste; "
              "se nao incluir, repita com --corpus-sem-teste)")


# ------------------------------------------------------------------- termos

def _dobrar(texto):
    texto = unicodedata.normalize("NFKD", str(texto).lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


def _carregar_glossario(path):
    _exigir(path, "Aponte para o glossario de termos medicos do trabalho anterior (--glossario).")
    if path.endswith(".json"):
        data = json.load(open(path, encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("termos", data.get("terms", list(data.keys())))
        saida = []
        for e in data:
            if isinstance(e, dict):
                e = e.get("termo", e.get("term", e.get("token", e.get("word", ""))))
            if str(e).strip():
                saida.append(str(e).strip())
        return saida
    if path.endswith(".csv") or path.endswith(".tsv"):
        import csv as _csv
        sep = "\t" if path.endswith(".tsv") else ","
        with open(path, encoding="utf-8", errors="replace", newline="") as fh:
            linhas = list(_csv.reader(fh, delimiter=sep))
        if linhas and linhas[0] and linhas[0][0].strip().lower() in ("termo", "term", "palavra", "word"):
            linhas = linhas[1:]
        return [l[0].strip() for l in linhas if l and l[0].strip()]
    return [l.strip() for l in open(path, encoding="utf-8", errors="replace")
            if l.strip() and not l.lstrip().startswith("#")]


def cmd_termos(args):
    brutos = _carregar_glossario(args.glossario)
    termos = {}
    for t in brutos:
        toks = tuple(_dobrar(x) for x in TOKEN_RE.findall(t))
        if toks:
            termos[" ".join(toks)] = toks
    max_n = max(len(t) for t in termos.values())

    def contar(textos):
        cont = Counter()
        for texto in textos:
            toks = [_dobrar(x) for x in TOKEN_RE.findall(texto)]
            for n in range(1, max_n + 1):
                for i in range(len(toks) - n + 1):
                    g = " ".join(toks[i:i + n])
                    if g in termos:
                        cont[g] += 1
        return cont

    fonte = _texto_dos_registros(args.corpus, args.campo_texto)
    sint_dados = json.load(open(_exigir(args.sintetico), encoding="utf-8"))
    sint = [e.get("texto_limpo") or " ".join(map(str, e.get("tokens", []))) for e in sint_dados]
    c_fonte, c_sint = contar(fonte), contar(sint)

    presentes = {t for t, n in c_fonte.items() if n > 0}
    cobertos = {t for t in presentes if c_sint.get(t, 0) > 0}
    occ_total = sum(c_fonte[t] for t in presentes)
    occ_cob = sum(c_fonte[t] for t in cobertos)
    print(f"Glossario: {len(brutos)} entradas no arquivo, {len(termos)} termos distintos "
          f"sem caixa e sem acento (o artigo de 2026 cita 2.582)")
    print(f"Corpus-fonte: {len(fonte)} registros | sinteticas: {len(sint)} anamneses")
    print(f"  termos do glossario que ocorrem no corpus-fonte   {len(presentes)}   (texto atual 2.507)")
    print(f"  desses, presentes nas sinteticas                  {len(cobertos)}   "
          f"= {100 * len(cobertos) / max(len(presentes), 1):.1f}% dos presentes   (texto atual 34,7%)")
    print(f"                                                         "
          f"= {100 * len(cobertos) / max(len(termos), 1):.1f}% do glossario inteiro")
    print(f"  ocorrencias no corpus-fonte desses termos         "
          f"{100 * occ_cob / max(occ_total, 1):.1f}% das ocorrencias de termos medicos   (texto atual 89,6%)")
    json.dump({"entradas": len(brutos), "termos": len(termos), "presentes_no_corpus": len(presentes),
               "cobertos_pelas_sinteticas": len(cobertos),
               "pct_cobertos_dos_presentes": 100 * len(cobertos) / max(len(presentes), 1),
               "pct_cobertos_do_glossario": 100 * len(cobertos) / max(len(termos), 1),
               "pct_ocorrencias_cobertas": 100 * occ_cob / max(occ_total, 1)},
              open(args.out, "w"), indent=2)
    print(f"\nSalvo em {args.out}.")


# ------------------------------------------------------------ reetiquetagem

def _formas_brutas(filepath):
    data = json.load(open(_exigir(filepath), encoding="utf-8"))
    if isinstance(data, dict) and "tokens" in data:
        return {str(t) for t in data["tokens"]}
    if isinstance(data, list):
        return {str(e.get("token", e.get("word", e.get("forma", "")))) if isinstance(e, dict) else str(e)
                for e in data} - {""}
    if isinstance(data, dict):
        return {str(k) for k in data if k not in ("total", "tokens", "config")}
    return set()


def _eh_gold(data):
    return (isinstance(data, list) and data and isinstance(data[0], dict)
            and "tokens" in data[0] and "labels" in data[0])


def cmd_reetiquetagem(args):
    import hashlib

    if args.procurar:
        print(f"Procurando arquivos de gold em {args.procurar}")
        print("(o gold antes da troca, se ela foi a unica mudanca, tem 4.368 ABBREV e 408 TYPO)\n")
        for arq in sorted(Path(args.procurar).rglob("*.json")):
            try:
                if arq.stat().st_size > 200 * 1024 * 1024:
                    continue
                data = json.load(open(arq, encoding="utf-8"))
            except Exception:
                continue
            if not _eh_gold(data) or len(data) != 500:
                continue
            c = Counter(l for e in data for l in e["labels"])
            h = hashlib.md5(arq.read_bytes()).hexdigest()[:10]
            marca = "   <- bate com o gold antes da troca" if (c["ABBREV"], c["TYPO"]) == (4368, 408) else ""
            print(f"  {str(arq):70s} hash {h}  ABBREV {c['ABBREV']:5d}  TYPO {c['TYPO']:4d}{marca}")
        return

    atual = json.load(open(_exigir(args.test), encoding="utf-8"))
    tax_ab, tax_ty = _formas_brutas(args.taxonomia_abbrev), _formas_brutas(args.taxonomia_typos)
    ab_min, ty_min = {f.lower() for f in tax_ab}, {f.lower() for f in tax_ty}

    def lista(t):
        f = t.lower()
        return ("A" if f in ab_min else "") + ("T" if f in ty_min else "")

    def exata(t):
        return t in tax_ab or t in tax_ty

    rot_por_forma = defaultdict(Counter)
    for e in atual:
        for t, l in zip(e["tokens"], e["labels"]):
            rot_por_forma[t.lower()][l] += 1
    c = Counter(l for e in atual for l in e["labels"])
    print(f"Gold atual {args.test}: ABBREV {c['ABBREV']}, TYPO {c['TYPO']}, CLEAN {c['CLEAN']}")
    print(f"Se a troca foi a unica mudanca, o gold antes dela tinha ABBREV {c['ABBREV'] - 49} "
          f"e TYPO {c['TYPO'] - 74}.")

    if not args.antigo:
        limpos = [(t, lista(t)) for e in atual for t, l in zip(e["tokens"], e["labels"])
                  if l == "CLEAN" and lista(t)]
        print(f"\nTokens CLEAN com forma listada na taxonomia (sem caixa)  {len(limpos)}")
        print(f"  com a forma exata, respeitando a caixa                 {sum(exata(t) for t, _ in limpos)}")
        print(f"  so batem depois de passar para minusculas              {sum(not exata(t) for t, _ in limpos)}")
        print("  formas mais comuns (lista A = abreviacoes, T = typos) e os rotulos da mesma forma no gold")
        for f, n in Counter(t.lower() for t, _ in limpos).most_common(args.top):
            r = rot_por_forma[f]
            print(f"    {f:18s} {n:4d}  lista {lista(f):2s}  no gold  CLEAN {r['CLEAN']:4d}  "
                  f"ABBREV {r['ABBREV']:4d}  TYPO {r['TYPO']:4d}")
        print("\nSem o gold antigo nao da para saber quais foram os 123. Rode com --procurar para "
              "achar uma versao antiga e depois com --antigo.")
        return

    antigo = json.load(open(_exigir(args.antigo), encoding="utf-8"))
    if len(antigo) != len(atual):
        print(f"[!] o gold antigo tem {len(antigo)} anamneses e o atual {len(atual)}; "
              f"comparando so as de mesmo indice e mesmos tokens")
    trocas, desalinhadas = [], 0
    for i, (ea, en) in enumerate(zip(antigo, atual)):
        if ea["tokens"] != en["tokens"]:
            desalinhadas += 1
            continue
        for j, (t, la, ln) in enumerate(zip(en["tokens"], ea["labels"], en["labels"])):
            if la != ln:
                trocas.append((i, j, t, la, ln))
    print(f"\nComparando com {args.antigo}")
    if desalinhadas:
        print(f"  [!] {desalinhadas} anamneses com tokens diferentes ficaram de fora da comparacao")
    print(f"  tokens com rotulo diferente  {len(trocas)}")
    for (la, ln), n in Counter((la, ln) for _, _, _, la, ln in trocas).most_common():
        print(f"    {la:6s} -> {ln:6s}  {n}")
    print(f"  forma exata na taxonomia     {sum(exata(t) for _, _, t, _, _ in trocas)} de {len(trocas)}")
    print(f"  anamneses envolvidas         {len({i for i, *_ in trocas})}")
    print("  formas trocadas, a lista em que aparecem e quantas ocorrencias da mesma forma "
          "ficaram CLEAN no gold atual")
    for f, n in Counter(t.lower() for _, _, t, _, _ in trocas).most_common(args.top):
        print(f"    {f:18s} {n:4d}  lista {lista(f):2s}  ainda CLEAN no gold atual  "
              f"{rot_por_forma[f]['CLEAN']}")
    json.dump([{"anamnese": i, "posicao": j, "forma": t, "antes": la, "depois": ln}
               for i, j, t, la, ln in trocas], open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"\nSalvo em {args.out}.")


# ------------------------------------------------------------------- regex

MAIUSC = "A-ZÁÉÍÓÚÂÊÔÃÕÇÀÜ"
REGRAS_REGEX = {
   "estrita": re.compile(rf"^[{MAIUSC}]{{2,}}$"),
    "ampla": re.compile(rf"^(?=(?:[^{MAIUSC}]*[{MAIUSC}]){{2}})[{MAIUSC}0-9/\-\.]+$"),
}


def cmd_regex(args):
    test_data = json.load(open(_exigir(args.test), encoding="utf-8"))
    top20_abbrev, _ = carregar_top20(args.top20)

    for nome, rx in REGRAS_REGEX.items():
        tp = fp = fn = 0
        tpo = fpo = fno = 0
        for e in test_data:
            for tok, lab in zip(e["tokens"], e["labels"]):
                pred = bool(rx.match(tok))
                gold = lab == "ABBREV"
                tp += pred and gold; fp += pred and not gold; fn += gold and not pred
                if tok.lower() not in top20_abbrev:
                    tpo += pred and gold; fpo += pred and not gold; fno += gold and not pred
        P, R, F = prf(tp, fp, fn)
        Po, Ro, Fo = prf(tpo, fpo, fno)
        print(f"regra {nome}:")
        print(f"   teste inteiro        P={P:.3f}  R={R:.3f}  F1={F:.3f}")
        print(f"   fora do Top-20       P={Po:.3f}  R={Ro:.3f}  F1={Fo:.3f}")
    print("\nNa mesma base, o modelo tem precisao e recall impressos pelo 'particoes'.")

# ---------------------------------------------------------------- 5. split dev

def cmd_split_dev(args):
    arquivos = sorted(glob.glob(str(Path(args.preds_dir) / "preds_*.json")))
    if not arquivos:
        print("Nenhum preds_*.json encontrado. Rode o grid com o fine_tuning_v2.py.")
        return

    n_docs = len(json.load(open(arquivos[0])))
    rng = np.random.default_rng(args.seed)
    dev_idx = set(rng.choice(n_docs, size=int(n_docs * args.frac), replace=False).tolist())
    print(f"dev: {len(dev_idx)} anamneses | teste: {n_docs - len(dev_idx)} | seed {args.seed}")

    def f1_abbrev(gold, pred):
        g, p = np.array(gold), np.array(pred)
        return prf(int(((g == ABBREV) & (p == ABBREV)).sum()),
                   int(((g != ABBREV) & (p == ABBREV)).sum()),
                   int(((g == ABBREV) & (p != ABBREV)).sum()))[2]

    por_config = defaultdict(list)
    for arq in arquivos:
        nome = Path(arq).stem.replace("preds_", "")
        partes = nome.split("_")
        modelo, config = partes[0], "_".join(partes[1:-1])
        docs = json.load(open(arq))
        res = {}
        for parte, escolha in [("dev", True), ("test", False)]:
            g, p = [], []
            for i, d in enumerate(docs):
                if (i in dev_idx) is not escolha:
                    continue
                g += d["word_gold"]; p += d["word_pred"]
            res[parte] = f1_abbrev(g, p)
        por_config[(modelo, config)].append(res)

    print(f"\n{'modelo':12s} {'config':16s} {'F1 dev':>8s} {'F1 teste':>10s}")
    medias = {}
    for (modelo, config), runs in sorted(por_config.items()):
        dev = float(np.mean([r["dev"] for r in runs]))
        test = float(np.mean([r["test"] for r in runs]))
        medias[(modelo, config)] = (dev, test)
        print(f"{modelo:12s} {config:16s} {dev:8.4f} {test:10.4f}")

    print()
    for modelo in sorted({m for m, _ in medias}):
        # So as configuracoes EPNI concorrem. CLEAN e RANDOM sao controles, e o
        # REAL e a referencia superior treinada em dado real, nao um candidato.
        cand = {c: v for (m, c), v in medias.items() if m == modelo and c.startswith("top")}
        if not cand:
            continue
        escolhida = max(cand, key=lambda c: cand[c][0])
        melhor = max(cand, key=lambda c: cand[c][1])
        print(f"{modelo}: escolhida no dev -> {escolhida} "
              f"(dev {cand[escolhida][0]:.4f}, teste {cand[escolhida][1]:.4f})")
        print(f"{modelo}: maxima no teste  -> {melhor} ({cand[melhor][1]:.4f})")
        if (modelo, "real") in medias:
            dev, test = medias[(modelo, "real")]
            print(f"{modelo}: referencia REAL  -> dev {dev:.4f}, teste {test:.4f}")
    json.dump({f"{m}|{c}": v for (m, c), v in medias.items()},
              open("selecao_dev_teste.json", "w"), indent=2)
    print("\nReporte a linha 'escolhida no dev'.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("densidade")
    a.add_argument("--datasets", default="datasets_experimento_ruido")
    a.add_argument("--clean", default="clean_synthetic_anamneses_gold_standart_tokens.json")
    a.set_defaults(func=cmd_densidade)

    b = sub.add_parser("lowercase")
    b.add_argument("--ckpt", required=True, nargs="+",
                   help="um ou mais checkpoints, por exemplo as tres seeds de uma configuracao")
    b.add_argument("--out", default="lowercase_resultados.json")
    b.add_argument("--test", default="test_set_real.json")
    b.add_argument("--tokenizer", default=None,
                   help="id do Hub, se o checkpoint nao tiver os arquivos do tokenizer")
    b.add_argument("--max-len", type=int, default=512,
                   help="512 avalia o documento inteiro; 300 reproduz o corte antigo")
    b.set_defaults(func=cmd_lowercase)

    c = sub.add_parser("particoes", aliases=["fora-do-top20"])
    c.add_argument("--preds", default="resultados_experimento/predicoes/preds_*top20_rate50*.json")
    c.add_argument("--top20", default="template_mapeamento_top20.json")
    c.add_argument("--taxonomia-abbrev", default="potential_abbreviations.json")
    c.set_defaults(func=cmd_particoes)

    d = sub.add_parser("formas-exclusivas")
    d.add_argument("--corpus", required=True, help="arquivo do corpus-fonte (8.411 anamneses)")
    d.add_argument("--campo-texto", default=None, help="nome do campo de texto, se for JSON")
    d.add_argument("--test", default="test_set_real.json")
    d.add_argument("--taxonomia-abbrev", default="potential_abbreviations.json")
    d.add_argument("--taxonomia-typos", default="potential_typos.json")
    d.add_argument("--corpus-sem-teste", action="store_true",
                   help="use se o corpus-fonte NAO contiver os 500 registros do teste")
    d.add_argument("--out", default="formas_exclusivas_do_teste.json")
    d.set_defaults(func=cmd_formas_exclusivas)

    g = sub.add_parser("gerar-preds")
    g.add_argument("--ckpt", required=True)
    g.add_argument("--nome", required=True,
                   help="ex.: BERTimbau_top20_rate50_seed42 (vira preds_<nome>.json)")
    g.add_argument("--test", default="test_set_real.json")
    g.add_argument("--tokenizer", default=None,
                   help="id do Hub, se o checkpoint nao tiver os arquivos do tokenizer")
    g.add_argument("--max-len", type=int, default=512,
                   help="512 avalia o documento inteiro; 300 reproduz o corte antigo")
    g.add_argument("--out-dir", default="resultados_experimento/predicoes")
    g.set_defaults(func=cmd_gerar_preds)

    r = sub.add_parser("regex")
    r.add_argument("--test", default="test_set_real.json")
    r.add_argument("--top20", default="template_mapeamento_top20.json")
    r.set_defaults(func=cmd_regex)

    t = sub.add_parser("top-sem-teste")
    t.add_argument("--corpus", required=True, help="arquivo do corpus-fonte (8.411 anamneses)")
    t.add_argument("--campo-texto", default=None, help="nome do campo de texto, se for JSON ou CSV")
    t.add_argument("--test", default="test_set_real.json")
    t.add_argument("--top20", default="template_mapeamento_top20.json")
    t.add_argument("--taxonomia-abbrev", default="potential_abbreviations.json")
    t.add_argument("--taxonomia-typos", default="potential_typos.json")
    t.add_argument("--corpus-sem-teste", action="store_true",
                   help="use se o arquivo do corpus NAO incluir os 500 registros do teste")
    t.add_argument("--sem-truncamentos", action="store_true",
                   help="tira do ranking as palavras cortadas por codificacao, como 'alterac'")
    t.add_argument("--ignorar", default=None,
                   help="formas a tirar do ranking, separadas por virgula")
    t.add_argument("--out", default="top_sem_teste.json")
    t.set_defaults(func=cmd_top_sem_teste)

    m = sub.add_parser("termos")
    m.add_argument("--corpus", required=True, help="arquivo do corpus-fonte (8.411 anamneses)")
    m.add_argument("--campo-texto", default=None, help="nome do campo de texto, se for JSON ou CSV")
    m.add_argument("--glossario", required=True,
                   help="glossario de termos medicos do trabalho anterior (.txt, .json ou .csv)")
    m.add_argument("--sintetico", default="clean_synthetic_anamneses_gold_standart_tokens.json")
    m.add_argument("--out", default="termos_medicos.json")
    m.set_defaults(func=cmd_termos)

    q = sub.add_parser("reetiquetagem")
    q.add_argument("--test", default="test_set_real.json")
    q.add_argument("--antigo", default=None, help="versao antiga do gold, antes da troca")
    q.add_argument("--procurar", default=None, help="pasta onde procurar versoes antigas do gold")
    q.add_argument("--taxonomia-abbrev", default="potential_abbreviations.json")
    q.add_argument("--taxonomia-typos", default="potential_typos.json")
    q.add_argument("--top", type=int, default=25)
    q.add_argument("--out", default="reetiquetagem.json")
    q.set_defaults(func=cmd_reetiquetagem)

    e = sub.add_parser("split-dev")
    e.add_argument("--preds-dir", default="resultados_experimento/predicoes")
    e.add_argument("--frac", type=float, default=0.3)
    e.add_argument("--seed", type=int, default=20260922)
    e.set_defaults(func=cmd_split_dev)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()