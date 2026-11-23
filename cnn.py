# CNN pequena no estilo LeNet: conv, pool, conv, pool, lineares
# em 28x28 o segundo pooling deixa 16 mapas de 5x5

import pandas as pd
import torch.nn as nn

import ajuda

ajuda.garantir_pastas()
ROTULOS = [str(d) for d in range(10)]


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.rede(x)


loader_tr, loader_va, loader_te, info = ajuda.carregar_partes()
ajuda.fixar_semente()
modelo = LeNet()
params = ajuda.n_parametros(modelo)
print("parametros", params)

hist, tempo, melhor_ep, perda_va = ajuda.treinar(modelo, loader_tr, loader_va)
ajuda.plot_perda(hist, "perda CNN LeNet", "figuras/perda_cnn.png")

y_te, p_te = ajuda.prever(modelo, loader_te)
m_te = ajuda.metricas(y_te, p_te, nome="cnn_teste")
print("pares mais confusos (real -> predito, n):")
for n, i, j in ajuda.pares_confusos(m_te["cm"]):
    print(f"  {i} -> {j}: {n}")

ajuda.plot_matriz_hatch(m_te["cm"], ROTULOS, "CNN (teste)", "figuras/cnn_cm_teste.png")
ajuda.plot_circulo_confusao(m_te["cm"], ROTULOS, "CNN LeNet", "figuras/cnn_circ_teste.png")

pd.DataFrame(
    [
        {
            "modelo": "cnn_lenet",
            "params": params,
            "tempo_s": tempo,
            "melhor_epoca": melhor_ep,
            "perda_val": perda_va,
            "acc_teste": m_te["acc"],
            "prec_teste": m_te["prec"],
            "rec_teste": m_te["rec"],
            "f1_teste": m_te["f1"],
        }
    ]
).to_csv("resultados/cnn_resumo.csv", index=False)
pd.DataFrame(hist).to_csv("resultados/cnn_curva.csv", index=False)
pd.DataFrame(m_te["cm"], index=ROTULOS, columns=ROTULOS).to_csv("resultados/cnn_cm_teste.csv")
print("ok cnn")
