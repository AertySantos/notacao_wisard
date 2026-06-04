import os
import cv2
import numpy as np
import shutil

# =========================================================
# CONFIG
# =========================================================

DATASET_ORIGINAL = "dataset_omr"
DATASET_LIMPO = "dataset_omr_limpo"
DATASET_OUTLIERS = "dataset_omr_outliers"

TAMANHO = (32, 32)

# mais baixo = mais agressivo
FATOR_DESVIO = 1.0

os.makedirs(DATASET_LIMPO, exist_ok=True)
os.makedirs(DATASET_OUTLIERS, exist_ok=True)

# =========================================================
# PROCESSA CADA CLASSE
# =========================================================

classes = sorted(os.listdir(DATASET_ORIGINAL))

for classe in classes:

    pasta_classe = os.path.join(
        DATASET_ORIGINAL,
        classe
    )

    if not os.path.isdir(pasta_classe):
        continue

    print(f"\nProcessando: {classe}")

    imagens = []
    arquivos = []

    # =====================================================
    # CARREGA IMAGENS
    # =====================================================

    for arquivo in os.listdir(pasta_classe):

        caminho = os.path.join(
            pasta_classe,
            arquivo
        )

        img = cv2.imread(caminho, 0)

        if img is None:
            continue

        # resize
        img = cv2.resize(img, TAMANHO)

        # binarização
        _, img = cv2.threshold(
            img,
            127,
            255,
            cv2.THRESH_BINARY
        )

        imagens.append(img.flatten())
        arquivos.append(caminho)

    # pula classes vazias
    if len(imagens) == 0:
        continue

    imagens = np.array(imagens)

    # =====================================================
    # IMAGEM MÉDIA
    # =====================================================

    media = np.mean(imagens, axis=0)

    # =====================================================
    # DISTÂNCIAS
    # =====================================================

    distancias = []

    for img in imagens:

        d = np.linalg.norm(img - media)

        distancias.append(d)

    distancias = np.array(distancias)

    # =====================================================
    # LIMIAR
    # =====================================================

    media_dist = np.mean(distancias)
    desvio = np.std(distancias)

    LIMIAR = media_dist + (
        FATOR_DESVIO * desvio
    )

    print(f"Limiar: {LIMIAR:.2f}")

    # =====================================================
    # PASTAS DESTINO
    # =====================================================

    pasta_limpa = os.path.join(
        DATASET_LIMPO,
        classe
    )

    pasta_outlier = os.path.join(
        DATASET_OUTLIERS,
        classe
    )

    os.makedirs(pasta_limpa, exist_ok=True)
    os.makedirs(pasta_outlier, exist_ok=True)

    # =====================================================
    # SEPARAÇÃO
    # =====================================================

    removidas = 0

    for i, caminho in enumerate(arquivos):

        nome = os.path.basename(caminho)

        if distancias[i] > LIMIAR:

            shutil.copy(
                caminho,
                os.path.join(
                    pasta_outlier,
                    nome
                )
            )

            removidas += 1

        else:

            shutil.copy(
                caminho,
                os.path.join(
                    pasta_limpa,
                    nome
                )
            )

    print(
        f"Mantidas: {len(arquivos) - removidas} | "
        f"Removidas: {removidas}"
    )

print("\nDataset limpo criado.")