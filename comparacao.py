# MLP escolhido na validacao contra a CNN: f1, parametros e tempo de treino

import os
import pandas as pd

import ajuda

ajuda.garantir_pastas()

mlp_arq = "resultados/mlp_resumo.csv"
cnn_arq = "resultados/cnn_resumo.csv"
if not os.path.exists(mlp_arq) or not os.path.exists(cnn_arq):
    raise SystemExit("falta o resumo. rode mlp.py e cnn.py antes.")

mlp = pd.read_csv(mlp_arq)
cnn = pd.read_csv(cnn_arq)
escolhido = mlp[mlp["escolhido"] == True].iloc[0]

tab = pd.DataFrame(
    [
        {
            "modelo": escolhido["modelo"],
            "f1": escolhido["f1_teste"],
            "acc": escolhido["acc_teste"],
            "params": escolhido["params"],
            "tempo_s": escolhido["tempo_s"],
        },
        {
            "modelo": "cnn_lenet",
            "f1": cnn.iloc[0]["f1_teste"],
            "acc": cnn.iloc[0]["acc_teste"],
            "params": cnn.iloc[0]["params"],
            "tempo_s": cnn.iloc[0]["tempo_s"],
        },
    ]
)
tab.to_csv("resultados/comparacao.csv", index=False)
print(tab.to_string(index=False))
print("candidatas mlp:")
print(mlp[["modelo", "params", "perda_val", "f1_val", "tempo_s", "escolhido"]].to_string(index=False))

longo = tab.melt(id_vars="modelo", value_vars=["f1", "acc"], var_name="metrica", value_name="valor")
(
    ajuda.ggplot(longo, ajuda.aes("modelo", "valor", fill="metrica"))
    + ajuda.geom_col(position="dodge")
    + ajuda.theme_bw()
    + ajuda.coord_cartesian(ylim=(longo["valor"].min() - 0.01, min(1.0, longo["valor"].max() + 0.005)))
    + ajuda.labs(title="teste: MLP e CNN", x="", y="", fill="")
).save("figuras/painel_comparacao.png", dpi=120, width=6.5, height=4, verbose=False)
print("salvei figuras/painel_comparacao.png")

rotulos = list(range(10))
mlp_cm = pd.read_csv("resultados/mlp_cm_teste.csv", index_col=0).to_numpy()
cnn_cm = pd.read_csv("resultados/cnn_cm_teste.csv", index_col=0).to_numpy()
ajuda.plot_matrizes_e_circulos(
    [mlp_cm, cnn_cm],
    ["MLP 256-128", "CNN"],
    rotulos,
    "figuras/matrizes_teste.png",
)

print("diff f1 (cnn - mlp):", round(tab.iloc[1]["f1"] - tab.iloc[0]["f1"], 4))
print("fim")
