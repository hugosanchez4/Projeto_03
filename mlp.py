# MLP com ReLU e saida softmax (via CrossEntropyLoss, que aplica log-softmax)
# o numero de camadas e o tamanho de cada uma sao os hiperparametros

import pandas as pd
import torch.nn as nn

import ajuda

ajuda.garantir_pastas()

# candidatas. a escolha final usa a menor perda de validacao
CANDIDATAS = {
    "mlp_128": [128],
    "mlp_256_128": [256, 128],
    "mlp_512_256_128": [512, 256, 128],
}
ROTULOS = [str(d) for d in range(10)]


class MLP(nn.Module):
    def __init__(self, ocultas):
        super().__init__()
        dims = [784] + list(ocultas) + [10]
        camadas = []
        for a, b in zip(dims[:-2], dims[1:-1]):
            camadas += [nn.Linear(a, b), nn.ReLU()]
        camadas.append(nn.Linear(dims[-2], dims[-1]))
        self.rede = nn.Sequential(*camadas)

    def forward(self, x):
        # 28x28 vira um vetor: a rede nao ve vizinhanca entre pixels
        return self.rede(x.view(x.size(0), -1))


loader_tr, loader_va, loader_te, info = ajuda.carregar_partes()
print("media", round(info["media"], 4), "desvio", round(info["desvio"], 4))

linhas = []
curvas = []
melhor_nome = None
melhor_perda = float("inf")
melhor_pacote = None

for nome, ocultas in CANDIDATAS.items():
    print("\n########", nome, ocultas, "########")
    ajuda.fixar_semente()
    modelo = MLP(ocultas)
    params = ajuda.n_parametros(modelo)
    hist, tempo, melhor_ep, perda_va = ajuda.treinar(modelo, loader_tr, loader_va)
    ajuda.plot_perda(hist, f"perda {nome}", f"figuras/perda_{nome}.png")
    for h in hist:
        curvas.append({"modelo": nome, **h})

    y_va, p_va = ajuda.prever(modelo, loader_va)
    m_va = ajuda.metricas(y_va, p_va, nome=f"{nome}_val")
    linha = {
        "modelo": nome,
        "camadas": str(ocultas),
        "params": params,
        "tempo_s": tempo,
        "melhor_epoca": melhor_ep,
        "perda_val": perda_va,
        "f1_val": m_va["f1"],
        "acc_val": m_va["acc"],
    }
    linhas.append(linha)
    if perda_va < melhor_perda:
        melhor_perda = perda_va
        melhor_nome = nome
        melhor_pacote = (modelo, linha)

print("\nescolhido pela validacao:", melhor_nome)
modelo, linha = melhor_pacote
y_te, p_te = ajuda.prever(modelo, loader_te)
m_te = ajuda.metricas(y_te, p_te, nome=f"{melhor_nome}_teste")
print("pares mais confusos (real -> predito, n):")
for n, i, j in ajuda.pares_confusos(m_te["cm"]):
    print(f"  {i} -> {j}: {n}")

ajuda.plot_matriz_hatch(
    m_te["cm"], ROTULOS, f"MLP {melhor_nome} (teste)", "figuras/mlp_cm_teste.png"
)
ajuda.plot_circulo_confusao(
    m_te["cm"], ROTULOS, f"MLP {melhor_nome}", "figuras/mlp_circ_teste.png"
)

for linha_i in linhas:
    linha_i["escolhido"] = linha_i["modelo"] == melhor_nome
linha["acc_teste"] = m_te["acc"]
linha["prec_teste"] = m_te["prec"]
linha["rec_teste"] = m_te["rec"]
linha["f1_teste"] = m_te["f1"]

pd.DataFrame(linhas).to_csv("resultados/mlp_resumo.csv", index=False)
pd.DataFrame(curvas).to_csv("resultados/mlp_curvas.csv", index=False)
pd.DataFrame(m_te["cm"], index=ROTULOS, columns=ROTULOS).to_csv("resultados/mlp_cm_teste.csv")
print("ok mlp")
