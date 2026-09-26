import json
import random
import re
from pathlib import Path
from copy import deepcopy

INPUT_FILE = "clean_synthetic_anamneses_gold_standart.json"
OUTPUT_DIR = Path("datasets_experimento_ruido_1196")

MAPPING_FILES = {
    "top10": "template_mapeamento_top10.json",
    "top15": "template_mapeamento_top15.json",
    "top20": "template_mapeamento_top20.json",
}

RATES = {
    "rate10": 0.10,
    "rate25": 0.25,
    "rate50": 0.50,
}

SEED = 42

def carregar_mapeamento(filepath):
    """
    Carrega o mapeamento e organiza por número de tokens (n-grama).
    Retorna dois dicts:
      - abbrev_map: {forma_limpa_lower: (forma_ruidosa, n_tokens)}
      - typo_map: {forma_limpa_lower: (forma_ruidosa, n_tokens)}
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    abbrev_map = {}
    for abrev, info in data.get("abbreviations", {}).items():
        significado = info.get("significado", "").strip()
        if significado:
            chave = significado.lower()
            n_tokens = len(chave.split())
            abbrev_map[chave] = {
                "noisy": abrev,
                "n_tokens": n_tokens,
                "label": "ABBREV",
            }

    typo_map = {}
    for typo, info in data.get("typos", {}).items():
        significado = info.get("significado", "").strip()
        if significado:
            chave = significado.lower()
            n_tokens = len(chave.split())
            typo_map[chave] = {
                "noisy": typo,
                "n_tokens": n_tokens,
                "label": "TYPO",
            }

    combined = {}
    combined.update(abbrev_map)
    combined.update(typo_map)

    by_size = {4: {}, 3: {}, 2: {}, 1: {}}
    for chave, info in combined.items():
        n = info["n_tokens"]
        if n in by_size:
            by_size[n][chave] = info
        else:
            by_size[1][chave] = info 

    print(f"  Mapeamento: {len(combined)} termos "
          f"(3-gram: {len(by_size[3])}, 2-gram: {len(by_size[2])}, 1-gram: {len(by_size[1])})")

    return by_size


def injetar_ruido_ngram(texto, mapeamento_by_size, taxa):
    tokens_originais = re.findall(r'\b[\w\-]+\b|[^\w\s]', texto)
    tokens_lower = [t.lower() for t in tokens_originais]

    n_tokens = len(tokens_originais)
    tokens_resultado = []
    labels = []

    i = 0
    while i < n_tokens:
        matched = False

        for n in [4, 3, 2, 1]:
            if i + n > n_tokens:
                continue

            ngram = " ".join(tokens_lower[i:i + n])

            if ngram in mapeamento_by_size.get(n, {}):
                info = mapeamento_by_size[n][ngram]

                if random.random() < taxa:
                    tokens_resultado.append(info["noisy"])
                    labels.append(info["label"])
                else:
                    for j in range(n):
                        tokens_resultado.append(tokens_originais[i + j])
                        labels.append("CLEAN")

                i += n
                matched = True
                break

        if not matched:
            tokens_resultado.append(tokens_originais[i])
            labels.append("CLEAN")
            i += 1

    texto_processado = " ".join(tokens_resultado)
    return tokens_resultado, labels, texto_processado


def processar_config(anamneses, mapeamento_by_size, taxa, config_name):
    """Processa todas as anamneses para uma configuração."""
    dataset = []
    total_tokens = 0
    total_abbrev = 0
    total_typo = 0

    for entry in anamneses:
        texto = entry["texto"]
        tokens, labels_list, texto_processado = injetar_ruido_ngram(
            texto, mapeamento_by_size, taxa
        )

        n_abbrev = labels_list.count("ABBREV")
        n_typo = labels_list.count("TYPO")

        dataset.append({
            "id": entry.get("id", len(dataset) + 1),
            "texto_limpo": texto,
            "texto_processado": texto_processado,
            "tokens": tokens,
            "labels": labels_list,
            "config": config_name,
        })

        total_tokens += len(labels_list)
        total_abbrev += n_abbrev
        total_typo += n_typo

    total_noise = total_abbrev + total_typo
    taxa_real = total_noise / max(total_tokens, 1)

    return dataset, {
        "total_tokens": total_tokens,
        "total_abbrev": total_abbrev,
        "total_typo": total_typo,
        "total_noise": total_noise,
        "taxa_real": taxa_real,
    }


def main():
    random.seed(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("[1/3] Carregando anamneses limpas...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        anamneses = json.load(f)
    print(f"       {len(anamneses)} anamneses")

    print("\n[2/3] Gerando dataset CLEAN...")
    clean_dataset = []
    for entry in anamneses:
        texto = entry["texto"]
        tokens = re.findall(r'\b[\w\-]+\b|[^\w\s]', texto)
        clean_dataset.append({
            "id": entry.get("id", len(clean_dataset) + 1),
            "texto_limpo": texto,
            "texto_processado": texto,
            "tokens": tokens,
            "labels": ["CLEAN"] * len(tokens),
            "config": "CLEAN",
        }) 

    clean_path = "clean_synthetic_anamneses_gold_standart_tokens.json"
    with open(clean_path, "w", encoding="utf-8") as f:
        json.dump(clean_dataset, f, ensure_ascii=False, indent=2)
    print(f"       Salvo: {clean_path}")

    print("\n[3/3] Gerando datasets EPNI...")
    for top_name, mapping_file in MAPPING_FILES.items():
        print(f"\n  === {top_name.upper()} ===")

        if not Path(mapping_file).exists():
            print(f"  [!] Arquivo não encontrado: {mapping_file}")
            continue

        mapeamento = carregar_mapeamento(mapping_file)

        for rate_name, rate_value in RATES.items():
            config_name = f"{top_name}_{rate_name}"
            print(f"\n  Config: {config_name} (rate={rate_value*100:.0f}%)")

            dataset, stats = processar_config(
                anamneses, mapeamento, rate_value, config_name
            )

            filename = f"dataset_{config_name}.json"
            filepath = OUTPUT_DIR / filename
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)

            print(f"    Tokens: {stats['total_tokens']}")
            print(f"    ABBREV: {stats['total_abbrev']}")
            print(f"    TYPO:   {stats['total_typo']}")
            print(f"    Noise:  {stats['total_noise']} ({stats['taxa_real']*100:.2f}%)")
            print(f"    Salvo:  {filepath}")

            exemplo = dataset[0]
            ruidos = [(t, l) for t, l in zip(exemplo["tokens"], exemplo["labels"]) if l != "CLEAN"]
            if ruidos:
                print(f"    Exemplo (ID {exemplo['id']}):")
                for t, l in ruidos[:5]:
                    print(f"      '{t}' -> {l}")

    print("\n" + "=" * 60)
    print("VERIFICAÇÃO DE EXPRESSÕES COMPOSTAS (top20)")
    print("=" * 60)

    if Path(MAPPING_FILES["top20"]).exists():
        mapeamento = carregar_mapeamento(MAPPING_FILES["top20"])

        expressoes_2gram = list(mapeamento.get(2, {}).keys())
        expressoes_3gram = list(mapeamento.get(3, {}).keys())

        print(f"\n  Expressões de 2 tokens:")
        for expr in expressoes_2gram:
            count = sum(1 for a in anamneses if expr in a["texto"].lower())
            print(f"    '{expr}': presente em {count}/{len(anamneses)} anamneses ({100*count/len(anamneses):.1f}%)")

        print(f"\n  Expressões de 3 tokens:")
        for expr in expressoes_3gram:
            count = sum(1 for a in anamneses if expr in a["texto"].lower())
            print(f"    '{expr}': presente em {count}/{len(anamneses)} anamneses ({100*count/len(anamneses):.1f}%)")

    print("\n[OK] Concluído!")


if __name__ == "__main__":
    main()