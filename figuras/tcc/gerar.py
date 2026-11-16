"""Figuras e numeros do capitulo de resultados do TCC.

Usa so a execucao oficial `final-pareado`, sem o overlay de custo, para que
o resultado corresponda ao metodo descrito no capitulo anterior.
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, "C:/Users/luiz_/OneDrive/Codigos/exp-mlquantify")
import numpy as np
import pandas as pd

import painel_bibliotecas as pb

RAIZ = Path("C:/Users/luiz_/exp-mlquantify-run/results")
SAIDA = Path("C:/Users/luiz_/OneDrive/Codigos/exp-mlquantify/figuras/tcc")
SAIDA.mkdir(parents=True, exist_ok=True)

#: Dez conjuntos entraram na colecao depois dos criterios de selecao
#: descritos no capitulo de metodo e ficam de fora aqui, o que devolve os
#: 132 do texto, com 108 binarios e 24 multiclasse. Sao todos multiclasse.
FORA = [l.strip() for l in io.open(
    "C:/Users/luiz_/OneDrive/Codigos/exp-mlquantify/descricao/"
    "conjuntos_fora_do_criterio.txt", encoding="utf-8") if l.strip()]

ok = pb.carregar(RAIZ / "final-pareado")
ok = ok[~ok["dataset_id"].isin(FORA)]
pares = pb.parear(ok)

# tempo soma entre as fases, memoria e o pico, que e o maior dos dois
for lado in ("minha", "dela"):
    pares[f"tempo_total_s_{lado}"] = (pares[f"tempo_treino_s_{lado}"]
                                      + pares[f"tempo_estimacao_total_s_{lado}"])
    pares[f"memoria_pico_mb_{lado}"] = np.maximum(
        pares[f"memoria_treino_mb_{lado}"],
        pares[f"memoria_estimacao_mb_{lado}"])

print("=" * 70)
print(f"conjuntos na execucao: {ok['dataset_id'].nunique()}")
print(f"pares metodo x biblioteca: "
      f"{pares.groupby(['metodo', 'biblioteca']).ngroups}")
print(f"metodos: {pares['metodo'].nunique()}")

# ---------------------------------------------------------------- Q1
teste = pb.testar(pares, erro="mae", margem=pb.MARGEM)
pb.figura_equivalencia(teste, pb.MARGEM, "mae", SAIDA / "q1-equivalencia.png")
print("\nQ1, AE")
print(teste["veredito"].value_counts().to_string())
print(f"  IC na faixa: {int(teste['ic_na_faixa'].sum())} de {len(teste)}")
print(f"  n por par: {teste['n_conjuntos'].min()} a "
      f"{teste['n_conjuntos'].max()}")
print(f"  |HL| maximo: {teste['hl'].abs().max():.4f}")
print("  os cinco maiores |HL|:")
print(teste.head(5)[["metodo", "biblioteca", "n_conjuntos", "hl", "ic_baixo",
                     "ic_alto", "veredito"]].to_string(index=False))
fora = teste[~teste["ic_na_faixa"].astype(bool)]
print("  fora da faixa:")
print(fora[["metodo", "biblioteca", "hl", "ic_baixo", "ic_alto",
            "p_equivalencia_holm"]].to_string(index=False))

rae = pb.testar(pares, erro="mrae", margem=pb.MARGEM)
print("\nQ1, RAE (sem veredito, margem nao definida)")
print(f"  |HL| mediano {rae['hl'].abs().median():.4f}, "
      f"maximo {rae['hl'].abs().max():.4f}")
print(f"  pares com IC contendo zero: "
      f"{int(((rae['ic_baixo'] < 0) & (rae['ic_alto'] > 0)).sum())} "
      f"de {len(rae)}")
print(rae.head(4)[["metodo", "biblioteca", "hl", "ic_baixo", "ic_alto"]]
      .to_string(index=False))

# ---------------------------------------------------------------- Q2
medidas = [
    ("tempo_total_s", "Tempo total", "q2-tempo"),
    ("memoria_pico_mb", "Memória", "q2-memoria"),
]
for coluna, titulo, nome in medidas:
    tabela = pb.testar_custo(pares, coluna)
    pb.figura_custo(tabela, titulo, SAIDA / f"{nome}.png")
    tabela.to_csv(SAIDA / f"{nome}.csv", index=False)
    print(f"\nQ2, {titulo}")
    print(f"  mais barata com IC inteiro abaixo de 1: "
          f"{int(tabela['ganha'].sum())} de {len(tabela)}")
    print(f"  mais cara com IC inteiro acima de 1: "
          f"{int(tabela['perde'].sum())} de {len(tabela)}")
    print(f"  razao mediana entre pares: {tabela['razao'].median():.3f}")
    print("  os tres extremos de cada lado:")
    print(tabela.head(3)[["metodo", "biblioteca", "razao", "ic_baixo",
                          "ic_alto"]].to_string(index=False))
    print(tabela.tail(3)[["metodo", "biblioteca", "razao", "ic_baixo",
                          "ic_alto"]].to_string(index=False))

print("\nQ2, razao mediana por fase e biblioteca")
linhas = []
for coluna, rotulo in (("tempo_treino_s", "tempo de treinamento"),
                       ("tempo_estimacao_total_s", "tempo de estimacao"),
                       ("memoria_treino_mb", "memoria no treinamento"),
                       ("memoria_estimacao_mb", "memoria na estimacao")):
    tabela = pb.testar_custo(pares, coluna)
    for lib, sub in tabela.groupby("biblioteca"):
        linhas.append({"medida": rotulo, "biblioteca": lib,
                       "razao_mediana": sub["razao"].median(),
                       "ganha": int(sub["ganha"].sum()), "n": len(sub)})
fases = pd.DataFrame(linhas).pivot(index="medida", columns="biblioteca",
                                   values="razao_mediana")
print(fases.round(3).to_string())
fases.to_csv(SAIDA / "q2-fases.csv")
print(f"\nfiguras em {SAIDA}")
