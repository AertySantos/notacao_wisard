import cv2
import numpy as np
import os
from pathlib import Path


# =========================================================
# CONFIGURAÇÕES
# =========================================================
BASE_DIR              = "package_ab"
PASTA_DATASET         = "dataset_omr"
PASTA_REFINADA        = "output_corrigido"
PASTA_INTERMEDIARIA   = "output_simbolos_intermediario"
PASTA_PRIMEIRO_REF    = "output_primeiro_refinamento"

for pasta in (PASTA_DATASET, PASTA_REFINADA, PASTA_INTERMEDIARIA, PASTA_PRIMEIRO_REF):
    os.makedirs(pasta, exist_ok=True)

resultados = []


# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================

def ler_semantic(caminho_semantic):
    with open(caminho_semantic, "r", encoding="utf-8") as f:
        conteudo = f.read().strip()
    if not conteudo:
        return []
    return [l.strip() for l in conteudo.split("\t") if l.strip()]


def _binarizar(img, bloco=19, c=9):
    """CLAHE → blur → threshold adaptativo → limpeza morfológica."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img2  = clahe.apply(img)
    blur  = cv2.GaussianBlur(img2, (1, 1), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        bloco, c
    )
    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def _remover_linhas(binary):
    """Detecta e remove linhas horizontais da pauta."""
    altura, largura = binary.shape

    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (largura // 2, 1))
    linhas = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_horizontal)
    linhas = cv2.dilate(linhas, kernel_horizontal, iterations=1)

    kernel_total   = cv2.getStructuringElement(cv2.MORPH_RECT, (largura, 1))
    linhas_completas = cv2.dilate(linhas, kernel_total, iterations=1)

    return cv2.bitwise_and(binary, cv2.bitwise_not(linhas_completas))


def _extrair_simbolos(img, sem_linhas, limiar_proj=500, espaco_max_vazio=6):
    """Fatia a imagem em símbolos via projeção vertical. Retorna (simbolos, cortes)."""
    projecao = np.sum(sem_linhas, axis=0)
    simbolos = []
    cortes   = []
    em_simbolo = False
    inicio = contador_vazio = 0

    for x, valor in enumerate(projecao):
        if valor > limiar_proj:
            if not em_simbolo:
                inicio = x
                em_simbolo = True
            contador_vazio = 0
        elif em_simbolo:
            contador_vazio += 1
            if contador_vazio > espaco_max_vazio:
                fim = x - contador_vazio
                cortes.append(fim)
                simbolos.append(img[:, max(0, inicio - 5):min(img.shape[1], fim + 5)])
                em_simbolo = False
                contador_vazio = 0

    if em_simbolo:
        fim = len(projecao) - contador_vazio
        cortes.append(fim)
        simbolos.append(img[:, max(0, inicio - 5):min(img.shape[1], fim + 5)])

    return simbolos, cortes


def verificar_pausa_inteira(img):
    """Retorna True se a imagem parece ser uma pausa inteira."""
    h, w = img.shape
    if w < 130:
        return False

    _, binario = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binario)

    if num_labels - 1 < 3:
        return False

    ws   = stats[1:, cv2.CC_STAT_WIDTH]
    hs   = stats[1:, cv2.CC_STAT_HEIGHT]
    props = ws / np.where(hs == 0, 1, hs)

    return bool(np.any((ws >= 120) & (hs >= 30) & (props > 3.0)))


def cortar_simbolo_largo(recorte, largura_base=35, dist_minima_cabecas=None):
    """
    Detecta cabeças de nota via Hough Circles e retorna os recortes
    individuais junto com as coordenadas dos centros detectados.
    """
    if dist_minima_cabecas is None:
        dist_minima_cabecas = int(largura_base * 0.5)

    # 1. Binarização
    _, binario = cv2.threshold(recorte, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Remoção de beams
    lk = int(largura_base * 0.8)
    ak = max(2, int(largura_base * 0.08))
    kernel_beam = cv2.getStructuringElement(cv2.MORPH_RECT, (lk, ak))
    beams       = cv2.morphologyEx(binario, cv2.MORPH_OPEN, kernel_beam)
    binario_tratado = cv2.subtract(binario, beams)

    # 3. Isolamento das cabeças
    tk = max(9, int(largura_base * 0.15))
    if tk % 2 == 0: tk += 1
    isolado = cv2.morphologyEx(
        binario_tratado, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tk, tk))
    )

    lf = max(3, int(largura_base * 0.10))
    af = max(7, int(largura_base * 0.20))
    if lf % 2 == 0: lf += 1
    if af % 2 == 0: af += 1
    isolado = cv2.morphologyEx(
        isolado, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lf, af))
    )

    # 4. Hough Circles
    borrado    = cv2.GaussianBlur(cv2.bitwise_not(isolado), (5, 5), 0)
    min_rad    = int(largura_base * 0.14)
    max_rad    = int(largura_base * 0.38)
    detectados = cv2.HoughCircles(
        borrado, cv2.HOUGH_GRADIENT,
        dp=1, minDist=int(largura_base * 0.25),
        param1=50, param2=10,
        minRadius=min_rad, maxRadius=max_rad
    )

    todos_circulos   = []
    coordenadas_centros = []

    if detectados is not None:
        todos_circulos = np.uint16(np.around(detectados))[0]

    # 5. Projeção vertical
    proj = np.sum(binario_tratado > 0, axis=0)

    # 6. Cortes entre notas detectadas
    cortes_finais = []

    if len(todos_circulos):
        lista_circulos = sorted(todos_circulos, key=lambda c: c[0])
        coordenadas_centros = [(int(c[0]), int(c[1])) for c in lista_circulos]

        for i in range(len(lista_circulos) - 1):
            c1, c2 = lista_circulos[i], lista_circulos[i + 1]
            ini = int(c1[0] + min_rad)
            fim = int(c2[0] - min_rad)

            if fim <= ini:
                continue

            regiao = proj[ini:fim].astype(np.float32)
            proj_suave = cv2.GaussianBlur(regiao.reshape(1, -1), (9, 1), 0).flatten()

            limiar = np.min(proj_suave) + 2
            vazio  = (proj_suave <= limiar).astype(np.uint8)

            melhor_inicio = melhor_tamanho = 0
            inicio_atual  = None

            for idx, val in enumerate(vazio):
                if val == 1:
                    if inicio_atual is None:
                        inicio_atual = idx
                else:
                    if inicio_atual is not None:
                        tamanho = idx - inicio_atual
                        if tamanho > melhor_tamanho:
                            melhor_tamanho = tamanho
                            melhor_inicio  = inicio_atual
                        inicio_atual = None

            if inicio_atual is not None:
                tamanho = len(vazio) - inicio_atual
                if tamanho > melhor_tamanho:
                    melhor_tamanho = tamanho
                    melhor_inicio  = inicio_atual

            corte_local = (
                melhor_inicio + melhor_tamanho // 2
                if melhor_tamanho >= 3
                else np.argmin(proj_suave)
            )
            cortes_finais.append(ini + corte_local)

    # 7. Fatiamento
    partes = []
    inicio = 0
    for corte in cortes_finais:
        parte = recorte[:, inicio:corte]
        if parte.shape[1] > 0:
            partes.append(parte)
        inicio = corte

    ultima = recorte[:, inicio:]
    if ultima.shape[1] > 0:
        partes.append(ultima)

    return partes, coordenadas_centros


def processar_e_cortar_simbolo(recorte, indice, largura_base=63, dist_minima_cabecas=30):
    """
    Pipeline completo: CLAHE → binarização → detecção/remoção de linhas
    → projeção vertical → extração de símbolos individuais.
    Retorna (partes, cortes).
    """
    # 1–2. Contraste e binarização
    binary = _binarizar(recorte, bloco=19, c=10)

    # 3–4. Redesenha linhas via projeção e agrupamento
    altura, largura = binary.shape
    quantidade_linhas = 0
    num = 8
    linhas = np.zeros_like(binary)
    while quantidade_linhas < 5 and (largura // num) >= 1:
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, largura // num), 1))
        num += 1
        linhas = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kh)
        n, _, _, _ = cv2.connectedComponentsWithStats(linhas, connectivity=8)
        quantidade_linhas = n - 1

    proj_linhas = np.sum(linhas > 0, axis=1)
    ys = np.where(proj_linhas > largura * 0.1)[0]
    linhas_redesenhadas = np.zeros_like(linhas)
    if len(ys):
        grupos, inicio_grupo = [], ys[0]
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] > 1:
                grupos.append((inicio_grupo, ys[i - 1]))
                inicio_grupo = ys[i]
        grupos.append((inicio_grupo, ys[-1]))
        for y1, y2 in grupos:
            linhas_redesenhadas[y1:y2 + 1, :] = 255

    linhas_erodidas = cv2.erode(
        linhas_redesenhadas,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)),
        iterations=1
    )
    sem_linhas = cv2.subtract(binary, linhas_erodidas)

    # 5–6. Projeção e extração
    return _extrair_simbolos(recorte, sem_linhas, limiar_proj=500, espaco_max_vazio=7)


# =========================================================
# LOOP PRINCIPAL
# =========================================================

for subpasta in sorted(os.listdir(BASE_DIR)):
    caminho_subpasta = os.path.join(BASE_DIR, subpasta)
    if not os.path.isdir(caminho_subpasta):
        continue

    print(f"\n{'='*48}\nProcessando: {subpasta}\n{'='*48}")

    arquivos_png = [
        f for f in os.listdir(caminho_subpasta)
        if f.endswith(".png") and not f.startswith("._")
    ]

    for arquivo_png in arquivos_png:
        nome_base        = Path(arquivo_png).stem
        caminho_png      = os.path.join(caminho_subpasta, arquivo_png)
        caminho_semantic = os.path.join(caminho_subpasta, f"{nome_base}.semantic")

        if not os.path.exists(caminho_semantic):
            continue

        img = cv2.imread(caminho_png, 0)

        # --- Extração inicial ---
        binary    = _binarizar(img, bloco=19, c=9)
        sem_linhas = _remover_linhas(binary)
        simbolos, _ = _extrair_simbolos(img, sem_linhas)

        print(f"{len(simbolos)} símbolos encontrados")

        # --- Primeiro refinamento ---
        crops_primeiro_refinamento = []

        for indice_simbolo, simbolo in enumerate(simbolos):
            if verificar_pausa_inteira(simbolo) or simbolo.shape[1] < 63:
                crops_primeiro_refinamento.append(simbolo)
                continue

            partes, _ = processar_e_cortar_simbolo(
                simbolo, indice=indice_simbolo, largura_base=63, dist_minima_cabecas=30
            )
            if not partes:
                partes = [simbolo]

            crops_primeiro_refinamento.extend(
                p for p in partes if p.shape[1] >= 8
            )

        for idx, crop in enumerate(crops_primeiro_refinamento):
            cv2.imwrite(
                os.path.join(PASTA_PRIMEIRO_REF, f"{nome_base}_{idx:05d}.png"),
                crop
            )

        # --- Segundo refinamento ---
        crops_finais = []

        for indice_crop, crop in enumerate(crops_primeiro_refinamento):
            if crop.shape[1] < 63:
                crops_finais.append(crop)
                continue

            partes_refinadas, _ = cortar_simbolo_largo(
                crop, largura_base=63, dist_minima_cabecas=30
            )
            crops_finais.extend(partes_refinadas or [crop])

        for idx, crop in enumerate(crops_finais):
            cv2.imwrite(
                os.path.join(PASTA_REFINADA, f"{nome_base}_{idx:05d}.png"),
                crop
            )

        # --- Alinhamento com o semantic ---
        labels_semantic = ler_semantic(caminho_semantic)
        qtd_crops    = len(crops_finais)
        qtd_semantic = len(labels_semantic)
        igual        = qtd_crops == qtd_semantic

        print(f"Arquivo: {nome_base} | Crops: {qtd_crops} | Semantic: {qtd_semantic} | Igual: {igual}")

        if igual:
            for i, (crop, label) in enumerate(zip(crops_finais, labels_semantic)):
                label_safe  = label.replace("/", "_").replace(":", "_")
                pasta_classe = os.path.join(PASTA_DATASET, label_safe)
                os.makedirs(pasta_classe, exist_ok=True)
                cv2.imwrite(os.path.join(pasta_classe, f"{nome_base}_{i:04d}.png"), crop)
            print("Dataset salvo")
        else:
            print("Ignorado por diferença")

        resultados.append({
            "arquivo": nome_base,
            "crops":    qtd_crops,
            "semantic": qtd_semantic,
            "igual":    igual
        })


print("\n================================================")
print("FINALIZADO")
print("================================================")