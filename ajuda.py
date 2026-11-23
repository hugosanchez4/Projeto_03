# funcoes em comum: dados, treino, metricas e figuras

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from plotnine import (
    ggplot,
    aes,
    geom_line,
    geom_point,
    geom_col,
    theme_bw,
    labs,
    scale_color_manual,
    scale_fill_manual,
    ylim,
    coord_cartesian,
)

SEED = 7
BATCH = 128
EPOCAS = 10
LR = 0.01  # com 0.1 a CNN diverge em algumas sementes (inclusive a 7)
MOMENTUM = 0.9
DEVICE = torch.device("cpu")


def fixar_semente(seed=SEED):
    # chamar ANTES de criar o modelo: os pesos iniciais sao sorteados no construtor
    np.random.seed(seed)
    torch.manual_seed(seed)


def carregar_partes():
    # mesma divisao em todo script
    base_tr = datasets.MNIST("dados", train=True, download=True)
    base_te = datasets.MNIST("dados", train=False, download=True)
    alvos = base_tr.targets.numpy()
    idx = np.arange(len(base_tr))
    tr_idx, va_idx = train_test_split(
        idx, test_size=0.1, random_state=SEED, stratify=alvos
    )
    pixels = base_tr.data[tr_idx].float() / 255.0
    media = pixels.mean().item()
    desvio = pixels.std(unbiased=False).item()
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((media,), (desvio,))]
    )
    treino_cheio = datasets.MNIST("dados", train=True, download=False, transform=tf)
    teste = datasets.MNIST("dados", train=False, download=False, transform=tf)
    g = torch.Generator()
    g.manual_seed(SEED)
    loader_tr = DataLoader(
        Subset(treino_cheio, tr_idx), batch_size=BATCH, shuffle=True, generator=g
    )
    loader_va = DataLoader(Subset(treino_cheio, va_idx), batch_size=BATCH, shuffle=False)
    loader_te = DataLoader(teste, batch_size=BATCH, shuffle=False)
    info = {
        "n_treino": len(tr_idx),
        "n_val": len(va_idx),
        "n_teste": len(base_te),
        "media": media,
        "desvio": desvio,
        "contagem_treino": np.bincount(alvos, minlength=10),
        "contagem_teste": np.bincount(base_te.targets.numpy(), minlength=10),
        "base_tr": base_tr,
    }
    return loader_tr, loader_va, loader_te, info


def n_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters())


def perda_no_loader(modelo, loader, crit):
    modelo.eval()
    soma, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = crit(modelo(x), y)
            soma += loss.item() * y.size(0)
            n += y.size(0)
    return soma / n


def treinar(modelo, loader_tr, loader_va, epocas=EPOCAS, lr=LR):
    # reinicia o embaralhamento: cada modelo ve os lotes na mesma ordem
    loader_tr.generator.manual_seed(SEED)
    modelo = modelo.to(DEVICE)
    opt = torch.optim.SGD(modelo.parameters(), lr=lr, momentum=MOMENTUM)
    crit = nn.CrossEntropyLoss()
    hist = []
    melhor_perda = float("inf")
    melhor_ep = 0
    melhor_estado = None
    t0 = time.perf_counter()
    for ep in range(1, epocas + 1):
        modelo.train()
        soma, n = 0.0, 0
        for x, y in loader_tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(modelo(x), y)
            loss.backward()
            opt.step()
            soma += loss.item() * y.size(0)
            n += y.size(0)
        perda_tr = soma / n
        perda_va = perda_no_loader(modelo, loader_va, crit)
        hist.append({"epoca": ep, "perda_treino": perda_tr, "perda_val": perda_va})
        print(f"epoca {ep}: treino {perda_tr:.4f}  val {perda_va:.4f}")
        if perda_va < melhor_perda:
            melhor_perda = perda_va
            melhor_ep = ep
            melhor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
    tempo = time.perf_counter() - t0
    modelo.load_state_dict(melhor_estado)
    modelo.to(DEVICE)
    print(f"melhor epoca {melhor_ep}  val {melhor_perda:.4f}  tempo {tempo:.1f}s")
    return hist, tempo, melhor_ep, melhor_perda


def prever(modelo, loader):
    modelo.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            pred = modelo(x.to(DEVICE)).argmax(1).cpu()
            ys.append(y)
            ps.append(pred)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def metricas(y_true, y_pred, nome="modelo"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    print("----", nome, "----")
    print("acuracia:", round(acc, 4))
    print("precisao:", round(prec, 4))
    print("revocacao:", round(rec, 4))
    print("f1_macro:", round(f1, 4))
    print("matriz:\n", cm)
    return {"modelo": nome, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "cm": cm}


def pares_confusos(cm, k=5):
    pares = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pares.append((int(cm[i, j]), i, j))
    pares.sort(reverse=True)
    return pares[:k]


def plot_perda(hist, titulo, saida):
    df = pd.DataFrame(hist)
    longo = df.melt(
        id_vars="epoca",
        value_vars=["perda_treino", "perda_val"],
        var_name="conjunto",
        value_name="perda",
    )
    longo["conjunto"] = longo["conjunto"].map(
        {"perda_treino": "treino", "perda_val": "validacao"}
    )
    (
        ggplot(longo, aes("epoca", "perda", color="conjunto"))
        + geom_line(size=1)
        + geom_point(size=2)
        + theme_bw()
        + scale_color_manual(values={"treino": "blue", "validacao": "red"})
        + labs(title=titulo, x="epoca", y="perda", color="")
    ).save(saida, dpi=120, width=6.5, height=4, verbose=False)
    print("salvei", saida)


def _escala_erros(cms):
    from matplotlib.colors import LinearSegmentedColormap

    cms = [np.asarray(cm) for cm in cms]
    n = cms[0].shape[0]
    fora = ~np.eye(n, dtype=bool)
    vmax = max(max(int(cm[fora].max()) for cm in cms), 1)
    cmap = LinearSegmentedColormap.from_list("erros", ["#fff7ed", "#fdba74", "#c2410c"])
    azul = np.array([0.816, 0.878, 0.945])
    return cms, n, vmax, cmap, azul


def _pinta_matriz(ax, cm, titulo, labels, cmap, vmax, azul):
    n = cm.shape[0]
    rgb = np.ones((n, n, 3))
    for i in range(n):
        for j in range(n):
            if i == j:
                rgb[i, j] = azul
            elif cm[i, j] > 0:
                rgb[i, j] = cmap(cm[i, j] / vmax)[:3]
    ax.imshow(rgb, interpolation="nearest")
    for i in range(n):
        for j in range(n):
            valor = int(cm[i, j])
            if i != j and valor == 0:
                cor = "#b4b4b4"
            elif i != j and valor > 0:
                r, g, b, _ = cmap(valor / vmax)
                luz = 0.2126 * r + 0.7152 * g + 0.0722 * b
                cor = "white" if luz < 0.55 else "black"
            else:
                cor = "black"
            ax.text(j, i, str(valor), ha="center", va="center", fontsize=7.5, color=cor)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("predito")
    ax.set_ylabel("real")
    ax.set_title(titulo)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)


def plot_matrizes(cms, titulos, labels, saida):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    cms, _, vmax, cmap, azul = _escala_erros(cms)
    fig, axes = plt.subplots(
        1,
        len(cms),
        figsize=(3.35 * len(cms) + 0.55, 3.55),
        constrained_layout=True,
    )
    if len(cms) == 1:
        axes = [axes]
    for ax, cm, titulo in zip(axes, cms, titulos):
        _pinta_matriz(ax, cm, titulo, labels, cmap, vmax, azul)
    barra = fig.colorbar(
        ScalarMappable(norm=Normalize(0, vmax), cmap=cmap),
        ax=axes,
        fraction=0.046,
        pad=0.03,
    )
    barra.set_label("erros")
    fig.savefig(saida, dpi=200)
    plt.close(fig)
    print("salvei", saida)


def plot_matrizes_e_circulos(cms, titulos, labels, saida):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.gridspec import GridSpec
    from pycirclize import Circos

    cms, _, vmax, cmap, azul = _escala_erros(cms)
    nomes = [str(x) for x in labels]
    minimo = 4
    fig = plt.figure(figsize=(7.4, 8.6))
    grade = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1, 1, 0.05],
        height_ratios=[1, 1.15],
        wspace=0.32,
        hspace=0.16,
        left=0.07,
        right=0.96,
        top=0.96,
        bottom=0.04,
    )
    for k, (cm, titulo) in enumerate(zip(cms, titulos)):
        ax = fig.add_subplot(grade[0, k])
        _pinta_matriz(ax, cm, titulo, labels, cmap, vmax, azul)
        erros = np.asarray(cm, dtype=float).copy()
        np.fill_diagonal(erros, 0)
        quadro = pd.DataFrame(erros, index=nomes, columns=nomes)

        def faixa(origem, destino, quadro=quadro, minimo=minimo):
            valor = float(quadro.loc[origem, destino])
            if valor < minimo:
                return {"alpha": 0}
            return {"alpha": 0.55 + 0.45 * (valor / vmax), "lw": 0.4}

        circos = Circos.chord_diagram(
            quadro,
            space=3,
            r_lim=(72, 100),
            cmap="tab10",
            order=nomes,
            label_kws=dict(size=9, r=108, adjust_rotation=False),
            link_kws=dict(direction=1, ec="none"),
            link_kws_handler=faixa,
        )
        polar = fig.add_subplot(grade[1, k], projection="polar")
        circos.plotfig(ax=polar)
        polar.set_rlim(0, 116)

    barra = fig.colorbar(
        ScalarMappable(norm=Normalize(0, vmax), cmap=cmap),
        cax=fig.add_subplot(grade[0, 2]),
    )
    barra.set_label("erros")
    fig.savefig(saida, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("salvei", saida)


def plot_matriz_hatch(cm, labels, titulo, saida):
    plot_matrizes([cm], [titulo], labels, saida)


def plot_circulo_confusao(cm, labels, titulo, saida):
    from pycirclize import Circos

    names = [str(x) for x in labels]
    matrix = pd.DataFrame(cm, index=names, columns=names)
    circos = Circos.initialize_from_matrix(
        matrix,
        space=2,
        r_lim=(78, 100),
        cmap="tab10",
        label_kws=dict(size=9),
        link_kws=dict(direction=1, ec="black", lw=0.3),
    )
    fig = circos.plotfig()
    fig.suptitle(titulo)
    fig.savefig(saida, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("salvei", saida)


def plot_barras(df, x, y, fill, titulo, saida, ylabel):
    (
        ggplot(df, aes(x, y, fill=fill))
        + geom_col()
        + theme_bw()
        + scale_fill_manual(values=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
        + labs(title=titulo, x="", y=ylabel, fill="")
    ).save(saida, dpi=120, width=6.5, height=4, verbose=False)
    print("salvei", saida)


def sem_cm(linha):
    return {k: v for k, v in linha.items() if k != "cm"}


def garantir_pastas():
    os.makedirs("figuras", exist_ok=True)
    os.makedirs("resultados", exist_ok=True)
    os.makedirs("dados", exist_ok=True)
