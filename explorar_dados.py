# descricao do MNIST: contagem, exemplos, media e desvio usados na normalizacao

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ajuda

ajuda.garantir_pastas()
ajuda.fixar_semente()

_, _, _, info = ajuda.carregar_partes()
print("treino", info["n_treino"], "val", info["n_val"], "teste", info["n_teste"])
print("media", round(info["media"], 4), "desvio", round(info["desvio"], 4))

cont = pd.DataFrame(
    {
        "digito": list(range(10)) * 2,
        "conjunto": ["treino"] * 10 + ["teste"] * 10,
        "n": list(info["contagem_treino"]) + list(info["contagem_teste"]),
    }
)
cont.to_csv("resultados/contagem_classes.csv", index=False)
pd.DataFrame(
    [
        {
            "media": info["media"],
            "desvio": info["desvio"],
            "n_treino": info["n_treino"],
            "n_val": info["n_val"],
            "n_teste": info["n_teste"],
        }
    ]
).to_csv("resultados/normalizacao.csv", index=False)
print(cont)

(
    ajuda.ggplot(cont, ajuda.aes("factor(digito)", "n", fill="conjunto"))
    + ajuda.geom_col(position="dodge")
    + ajuda.theme_bw()
    + ajuda.labs(title="amostras por digito", x="digito", y="n", fill="")
).save("figuras/contagem_classes.png", dpi=120, width=7, height=4, verbose=False)
print("salvei figuras/contagem_classes.png")

# dois exemplos de cada classe, sem normalizar, pra enxergar o digito
base = info["base_tr"]
alvos = base.targets.numpy()
fig, axes = plt.subplots(2, 10, figsize=(10, 2.6))
for digito in range(10):
    onde = np.where(alvos == digito)[0][:2]
    for linha, i in enumerate(onde):
        axes[linha, digito].imshow(base.data[i], cmap="gray")
        axes[linha, digito].set_xticks([])
        axes[linha, digito].set_yticks([])
        if linha == 0:
            axes[linha, digito].set_title(str(digito))
plt.tight_layout()
plt.savefig("figuras/exemplos_mnist.png", dpi=120)
plt.close()
print("salvei figuras/exemplos_mnist.png")
print("fim")
