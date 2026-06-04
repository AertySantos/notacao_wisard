import cv2
import numpy as np
import os
from pathlib import Path
import shutil

PASTA_INTERMEDIARIA = "output_simbolos_intermediario"
PASTA_REFINADA = "output_corrigido"

os.makedirs(PASTA_INTERMEDIARIA, exist_ok=True)
os.makedirs(PASTA_REFINADA, exist_ok=True)

def ler_semantic(caminho_semantic):

    with open(caminho_semantic, "r", encoding="utf-8") as f:
        conteudo = f.read().strip()

    if not conteudo:
        return []

    labels = conteudo.split("\t")

    labels = [
        l.strip()
        for l in labels
        if l.strip()
    ]

    return labels


# =========================================================
# FUNÇÃO PRINCIPAL DE PROCESSAMENTO E CORTE
# =========================================================
def processar_e_cortar_simbolo(recorte, indice, largura_base=63, dist_minima_cabecas=30):
    print("\n" + "=" * 60)
    print(f"PROCESSANDO SÍMBOLO {indice}")
    print("=" * 60)

    #mostrar(recorte, f"Original {indice}")
    largura_total = recorte.shape[1]

    # 1. BINARIZAÇÃO OTSU
    _, binario = cv2.threshold(
        recorte, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    #mostrar(binario, f"Binário {indice}")

    # 2. IDENTIFICAÇÃO DE PAUSA INTEIRA (Componentes Conectados)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binario, connectivity=8
    )
    
    blocos_validos = []
    for k in range(1, num_labels):
        x = stats[k, cv2.CC_STAT_LEFT]
        y = stats[k, cv2.CC_STAT_TOP]
        bw = stats[k, cv2.CC_STAT_WIDTH]
        bh = stats[k, cv2.CC_STAT_HEIGHT]
        area = stats[k, cv2.CC_STAT_AREA]
        
        if bh > 4:  # Ignora linhas finas
            blocos_validos.append({"x": x, "y": y, "w": bw, "h": bh, "area": area, "prop": bw / max(bh, 1)})

    eh_pausa_inteira = False
    if largura_total >= 130 and len(blocos_validos) >= 3:
        for bloco in blocos_validos:
            if bloco["w"] >= 120 and bloco["h"] >= 30 and bloco["prop"] > 3.0:
                eh_pausa_inteira = True
                break

    if eh_pausa_inteira:
        print(f"[PAUSA DETECTADA] Símbolo {indice} - Pulando divisões.")
        #mostrar(recorte, f"PAUSA DETECTADA {indice}")
        return [recorte], []

    # 3. TRATAMENTO MORFOLÓGICO E BLUR
    tamanho_kernel = max(5, int(largura_base * 0.15))
    if tamanho_kernel % 2 == 0:
        tamanho_kernel += 1

    kernel_abertura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamanho_kernel, tamanho_kernel))
    isolado = cv2.morphologyEx(binario, cv2.MORPH_OPEN, kernel_abertura)
    #mostrar(isolado, f"Abertura {indice}")

    kernel_fechamento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    isolado = cv2.morphologyEx(isolado, cv2.MORPH_CLOSE, kernel_fechamento)
    #mostrar(isolado, f"Fechamento {indice}")

    recorte_tratado = cv2.bitwise_not(isolado)
    borrado = cv2.GaussianBlur(recorte_tratado, (9, 9), 0)
    #mostrar(borrado, f"Blur {indice}")

    # 4. HOUGH CIRCLES (Parâmetros calibrados para largura_base=63)
    min_rad = int(largura_base * 0.18)
    detectados = cv2.HoughCircles(
        borrado,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(largura_base * 0.55),
        param1=80,
        param2=30,
        minRadius=min_rad,
        maxRadius=int(largura_base * 0.30)
    )

    lista_circulos = []
    todos_circulos = []
    cortes = []
    proj = np.sum(binario > 0, axis=0)

    if detectados is not None:
        todos_circulos = list(np.uint16(np.around(detectados[0])))
        mantidos = list(todos_circulos)

        # PASSO 1: Remover Círculos Concêntricos
        modificado = True
        while modificado:
            modificado = False
            for i in range(len(mantidos)):
                for j in range(i + 1, len(mantidos)):
                    c1, c2 = mantidos[i], mantidos[j]
                    dist = np.hypot(int(c1[0]) - int(c2[0]), int(c1[1]) - int(c2[1]))
                    if dist < max(c1[2], c2[2]):
                        mantidos.pop(j if c1[2] >= c2[2] else i)
                        modificado = True
                        break
                if modificado: break

        # PASSO 2: Filtro por Distância Mínima entre Cabeças
        modificado = True
        while modificado:
            modificado = False
            mantidos = sorted(mantidos, key=lambda c: c[0])
            for i in range(len(mantidos) - 1):
                c1, c2 = mantidos[i], mantidos[i + 1]
                if abs(int(c2[0]) - int(c1[0])) < dist_minima_cabecas:
                    mantidos.pop(i + 1)
                    modificado = True
                    break

        lista_circulos = sorted(mantidos, key=lambda c: c[0])

        # PASSO 3: Cortes Estruturais entre as Notas Detectadas
        for i in range(len(lista_circulos) - 1):
            c1, c2 = lista_circulos[i], lista_circulos[i + 1]
            inicio_busca = int(c1[0] + min_rad)
            fim_busca = int(c2[0] - min_rad)

            if fim_busca > inicio_busca:
                regiao = proj[inicio_busca:fim_busca]
                corte = np.argmin(regiao) + inicio_busca
                cortes.append(corte)
                print(f"[HOUGH] Corte estrutural em x={corte}")

    # 5. PASSO INTELIGENTE E PREENCHIMENTO HÍBRIDO
    largura_passo = largura_base
    if len(lista_circulos) >= 2:
        distancia_real = lista_circulos[1][0] - lista_circulos[0][0]
        if distancia_real > 12:
            largura_passo = distancia_real

    blocos = sorted(list(set([0] + cortes + [largura_total])))
    for i in range(len(blocos) - 1):
        b_inicio, b_fim = blocos[i], blocos[i + 1]
        largura_bloco = b_fim - b_inicio

        if largura_bloco >= int(largura_passo * 1.25):
            num_partes_bloco = max(1, round(largura_bloco / largura_passo))
            if num_partes_bloco > 1:
                for j in range(1, num_partes_bloco):
                    x_esperado = b_inicio + int(j * largura_bloco / num_partes_bloco)
                    margem = int(largura_passo * 0.4)
                    inicio_busca = max(b_inicio, x_esperado - margem)
                    fim_busca = min(b_fim, x_esperado + margem)
                    
                    regiao = proj[inicio_busca:fim_busca]
                    if len(regiao) > 0:
                        corte_local = np.argmin(regiao) + inicio_busca
                        cortes.append(corte_local)
                        print(f"[HÍBRIDO] Preenchimento de corte em x={corte_local}")

    # 6. DETECÇÃO DE ESPAÇOS EM BRANCO (Silêncios intermediários)
    em_vazio = False
    inicio_vazio = 0
    largura_min_vazio = 10

    for x, valor in enumerate(proj):
        if valor <= 5:
            if not em_vazio:
                inicio_vazio = x
                em_vazio = True
        else:
            if em_vazio:
                largura_vazio = x - inicio_vazio
                if largura_vazio >= largura_min_vazio:
                    corte_vazio = inicio_vazio + largura_vazio // 2
                    cortes.append(corte_vazio)
                    print(f"[VAZIO] Corte adaptativo em x={corte_vazio} (largura={largura_vazio})")
                em_vazio = False

    # 7. TRAVA DE SEGURANÇA (Garante corte caso o símbolo seja excessivamente largo e sem divisões)
    if len(cortes) == 0  and len(lista_circulos) >= 1:
        margem_seguranca = max(8, int(largura_total * 0.15))
        if largura_total > margem_seguranca * 2:
            regiao_busca = proj[margem_seguranca : largura_total - margem_seguranca]
            corte_forcado = np.argmin(regiao_busca) + margem_seguranca
        else:
            corte_forcado = largura_total // 2
        cortes.append(corte_forcado)
        print(f"[TRAVA SEGURANÇA] Corte forçado em x={corte_forcado}")

    # 8. REFINAMENTO E FILTRAGEM DOS CORTES
    cortes = sorted(list(set(cortes)))
    cortes_filtrados = []
    dist_min_cortes = 10

    for c in cortes:
        if len(cortes_filtrados) == 0 or abs(c - cortes_filtrados[-1]) > dist_min_cortes:
            cortes_filtrados.append(c)
    cortes = cortes_filtrados
    print("Cortes finais aplicados:", cortes)

    # 9. RENDERIZAÇÃO DO DEBUG VISUAL
    img_debug = cv2.cvtColor(recorte, cv2.COLOR_GRAY2BGR)
    for c in todos_circulos:
        cv2.circle(img_debug, (c[0], c[1]), c[2], (0, 255, 255), 1)
    for c in lista_circulos:
        cv2.circle(img_debug, (c[0], c[1]), c[2], (0, 255, 0), 2)
        cv2.circle(img_debug, (c[0], c[1]), 2, (0, 0, 255), -1)
    for corte in cortes:
        cv2.line(img_debug, (corte, 0), (corte, recorte.shape[0]), (255, 0, 0), 2)
    
    #mostrar(img_debug, f"Círculos e Cortes Detectados {indice}")

    # Plot do gráfico de projeção original
    #plt.figure(figsize=(12, 3))
    #plt.plot(proj)
    #plt.title(f"Projeção símbolo {indice}")
    #plt.grid()
    # plt.show()

    # 10. FATIAMENTO EM PARTES
    partes = []
    inicio = 0
    for corte in cortes:
        if corte <= inicio:
            continue
        parte = recorte[:, inicio:corte]
        if parte.shape[1] > 5:
            partes.append(parte)
        inicio = corte

    ultima_parte = recorte[:, inicio:]
    if ultima_parte.shape[1] > 5:
        partes.append(ultima_parte)

    return partes, cortes


pasta_destino = "output_simbolos"

# =========================================================
# PROCESSAMENTO PRINCIPAL
# =========================================================


PASTA_DATASET = "dataset_test"

os.makedirs("output_primeiro_refinamento", exist_ok=True)

resultados = []


def verificar_pausa_inteira(img):
    """
    Analisa se a imagem corresponde a uma pausa inteira.
    Critérios: largura total grande, min. 3 blocos e um bloco horizontal principal.
    """
    h, w = img.shape
    if w < 130:
        return False

    # Binariza para detectar os componentes conectados
    _, binario = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binario)
    
    # Desconsidera o fundo (index 0)
    if num_labels - 1 < 3:
        return False

    # Extrai as propriedades geométricas dos blocos detectados
    ws = stats[1:, cv2.CC_STAT_WIDTH]
    hs = stats[1:, cv2.CC_STAT_HEIGHT]
    
    # Evita divisão por zero caso haja blocos inválidos
    hs_safe = np.where(hs == 0, 1, hs)
    props = ws / hs_safe

    # Condição: bloco muito largo, alto e com proporção horizontal clara
    condicao = (ws >= 120) & (hs >= 30) & (props > 3.0)
    
    return np.any(condicao)


def cortar_simbolo_largo(
    recorte,
    largura_base=35,
    dist_minima_cabecas=None
):
    if dist_minima_cabecas is None:
        dist_minima_cabecas = int(largura_base * 0.5)
    

    coordenadas_centros = []

    # =====================================================
    # 1. BINARIZAÇÃO
    # =====================================================
    _, binario = cv2.threshold(
        recorte,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

   
    # =====================================================
    # 1.5. REMOÇÃO DE BEAMS (BARRAS DE LIGAÇÃO)
    # =====================================================
    largura_kernel_beam = int(largura_base * 0.8)  
    altura_kernel_beam = max(2, int(largura_base * 0.08))

    kernel_horizontal = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (largura_kernel_beam, altura_kernel_beam)
    )

    beams_detectados = cv2.morphologyEx(
        binario,
        cv2.MORPH_OPEN,
        kernel_horizontal
    )

    binario_tratado = cv2.subtract(binario, beams_detectados)

   
    # =====================================================
    # 2. ISOLAMENTO DAS CABEÇAS (CORRIGIDO)
    # =====================================================
    tamanho_kernel = max(9, int(largura_base * 0.15))
    if tamanho_kernel % 2 == 0:
        tamanho_kernel += 1

    kernel_abertura = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (tamanho_kernel, tamanho_kernel)
    )

    isolado = cv2.morphologyEx(
        binario_tratado,
        cv2.MORPH_OPEN,
        kernel_abertura
    )

   

    # Ajuste adaptativo do kernel de fechamento (foco vertical)
    # para unir as metades da nota cortada pela remoção de linhas
    largura_fechamento = max(3, int(largura_base * 0.10))
    altura_fechamento = max(7, int(largura_base * 0.20)) 
    
    if largura_fechamento % 2 == 0: largura_fechamento += 1
    if altura_fechamento % 2 == 0: altura_fechamento += 1

    kernel_fechamento = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (largura_fechamento, altura_fechamento)
    )

    isolado = cv2.morphologyEx(
        isolado,
        cv2.MORPH_CLOSE,
        kernel_fechamento
    )

   

    # =====================================================
    # 3. PREPARAÇÃO PARA HOUGH
    # =====================================================
    recorte_tratado = cv2.bitwise_not(isolado)

    borrado = cv2.GaussianBlur(
        recorte_tratado,
        (5, 5),
        0
    )

   

    # =====================================================
    # 4. HOUGH CIRCLES
    # =====================================================
    min_dist = int(largura_base * 0.25)
    min_rad  = int(largura_base * 0.14)
    max_rad  = int(largura_base * 0.38)

    detectados = cv2.HoughCircles(
        borrado,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_dist,
        param1=50,
        param2=10,
        minRadius=min_rad,
        maxRadius=max_rad
    )

    img_hough = cv2.cvtColor(recorte, cv2.COLOR_GRAY2BGR)
    todos_circulos = []

    if detectados is not None:
        todos_circulos = np.uint16(np.around(detectados))[0]
        print(f"Círculos detectados: {len(todos_circulos)}")

        for c in todos_circulos:
            cv2.circle(img_hough, (c[0], c[1]), c[2], (0,255,0), 2)
            cv2.circle(img_hough, (c[0], c[1]), 2, (0,0,255), -1)
    else:
        print("Nenhum círculo detectado")

   

    # =====================================================
    # 4.5. DETECÇÃO DE NOTAS ABERTAS
    # =====================================================
    contornos, _ = cv2.findContours(
        isolado,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    img_abertas = cv2.cvtColor(recorte, cv2.COLOR_GRAY2BGR)
    cabecas_abertas = []

    for cnt in contornos:
        area = cv2.contourArea(cnt)

        if area < (largura_base * 0.15):
            continue
        if len(cnt) < 5:
            continue

        elipse = cv2.fitEllipse(cnt)
        (x, y), (w, h), angulo = elipse

        proporcao = max(w, h) / (min(w, h) + 1e-5)

        if (
            min_rad * 1.2 < w < max_rad * 3 and
            min_rad * 1.2 < h < max_rad * 3 and
            proporcao < 2.5
        ):
            mask = np.zeros_like(isolado)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            roi = cv2.bitwise_and(binario, mask)

            pixels_total = np.sum(mask > 0)
            pixels_pretos = np.sum(roi > 0)
            preenchimento = pixels_pretos / (pixels_total + 1e-5)

            if preenchimento < 0.65:
                cabecas_abertas.append(
                    (int(x), int(y), int(max(w, h)/2))
                )
                cv2.ellipse(img_abertas, elipse, (255, 0, 0), 2)
                cv2.circle(img_abertas, (int(x), int(y)), 2, (0,0,255), -1)

   
        
        # =====================================================
        # 5. PROJEÇÃO VERTICAL
        # =====================================================
        proj = np.sum(binario_tratado > 0, axis=0)

       
    else:
        proj = np.sum(binario_tratado > 0, axis=0)

    # =====================================================
    # 6. DEFINIÇÃO DOS CORTES
    # =====================================================
    cortes_finais = []

    if detectados is not None:
        lista_circulos = sorted(todos_circulos, key=lambda c: c[0])
        coordenadas_centros = [(c[0], c[1]) for c in lista_circulos]

        for i in range(len(lista_circulos)-1):
            c1 = lista_circulos[i]
            c2 = lista_circulos[i+1]

            inicio_busca = int(c1[0] + min_rad)
            fim_busca    = int(c2[0] - min_rad)

            if fim_busca > inicio_busca:
                regiao_entre = proj[inicio_busca:fim_busca].astype(np.float32)

                # SUAVIZAÇÃO DA PROJEÇÃO
                proj_suave = cv2.GaussianBlur(
                    regiao_entre.reshape(1, -1),
                    (9, 1),
                    0
                ).flatten()

                

                # ENCONTRA REGIÕES DE BAIXA DENSIDADE
                limiar = np.min(proj_suave) + 2
                vazio = (proj_suave <= limiar).astype(np.uint8)

                melhor_inicio = 0
                melhor_tamanho = 0
                inicio_atual = None

                for idx, val in enumerate(vazio):
                    if val == 1:
                        if inicio_atual is None:
                            inicio_atual = idx
                    else:
                        if inicio_atual is not None:
                            tamanho = idx - inicio_atual
                            if tamanho > melhor_tamanho:
                                melhor_tamanho = tamanho
                                melhor_inicio = inicio_atual
                            inicio_atual = None

                if inicio_atual is not None:
                    tamanho = len(vazio) - inicio_atual
                    if tamanho > melhor_tamanho:
                        melhor_tamanho = tamanho
                        melhor_inicio = inicio_atual

                # DEFINE CORTE
                if melhor_tamanho < 3:
                    corte_local = np.argmin(proj_suave)
                    print("Fallback para mínimo global")
                else:
                    corte_local = melhor_inicio + (melhor_tamanho // 2)
                    print(f"Região vazia encontrada: {melhor_tamanho}px")

                corte = inicio_busca + corte_local
                cortes_finais.append(corte)

    print("Cortes encontrados:", cortes_finais)

    # =====================================================
    # 7. VISUALIZAÇÃO DOS CORTES
    # =====================================================
    img_cortes = cv2.cvtColor(recorte, cv2.COLOR_GRAY2BGR)

    for corte in cortes_finais:
        cv2.line(img_cortes, (corte, 0), (corte, recorte.shape[0]), (255,0,0), 2)

    
    # =====================================================
    # 8. RECORTES FINAIS
    # =====================================================
    partes = []
    inicio = 0

    for corte in cortes_finais:
        parte = recorte[:, inicio:corte]
        partes.append(parte)
        inicio = corte

    partes.append(recorte[:, inicio:])
    print(f"Símbolo dividido em {len(partes)} partes")        

    return partes, coordenadas_centros


    # ==========================================
    # CARREGAR AMOSTRAS A PARTIR DO test.txt
    # ==========================================
def main():
    TEST_TXT = "test.txt"

    PACOTES = [
        Path("package_aa"),
        Path("package_ab")
    ]

    amostras = []

    with open(TEST_TXT, "r", encoding="utf-8") as f:
        nomes = [linha.strip() for linha in f if linha.strip()]

    for nome in nomes:

        pasta_encontrada = None

        for pacote in PACOTES:
            candidata = pacote / nome

            if candidata.exists():
                pasta_encontrada = candidata
                break

        if pasta_encontrada is None:
            print(f"[ERRO] Pasta não encontrada: {nome}")
            continue

        png = pasta_encontrada / f"{nome}.png"
        agnostic = pasta_encontrada / f"{nome}.agnostic"
        semantic = pasta_encontrada / f"{nome}.semantic"

        if not png.exists():
            print(f"[ERRO] PNG ausente: {png}")
            continue

        if not agnostic.exists():
            print(f"[ERRO] AGNOSTIC ausente: {agnostic}")
            continue

        if not semantic.exists():
            print(f"[ERRO] SEMANTIC ausente: {semantic}")
            continue

        amostras.append({
            "nome": nome,
            "png": str(png),
            "agnostic": str(agnostic),
            "semantic": str(semantic)
        })

    # ==========================================
    # PROCESSAMENTO
    # ==========================================

    for amostra in amostras:

        nome_base = amostra["nome"]

        print("\n================================================")
        print(f"Processando: {nome_base}")
        print("================================================")

        caminho_png = amostra["png"]
        caminho_semantic = amostra["semantic"]
        caminho_agnostic = amostra["agnostic"]

        img = cv2.imread(caminho_png, 0)

        if img is None:
            print(f"[ERRO] Não foi possível abrir {caminho_png}")
            continue


        # =====================================================
        # EXTRAÇÃO INICIAL
        # =====================================================

        # contraste
        clahe = cv2.createCLAHE(
            clipLimit=3.0,
            tileGridSize=(8,8)
        )

        img2 = clahe.apply(img)

        # blur
        blur = cv2.GaussianBlur(img2, (1,1), 0)

        # threshold adaptativo
        binary = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            19,
            9
        )

        # limpeza
        kernel = np.ones((2,2), np.uint8)

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel
        )

        #mostrar(binary, "Binarização Final")

        # =========================================================
        # 3. DETECÇÃO DAS LINHAS HORIZONTAIS
        # =========================================================

        altura, largura = binary.shape

        kernel_horizontal = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (largura // 2, 1)
        )

        linhas_detectadas = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel_horizontal
        )

        linhas_detectadas = cv2.dilate(
            linhas_detectadas,
            kernel_horizontal,
            iterations=1
        )

        #mostrar(linhas_detectadas, "Linhas Detectadas")

        # 1. Pegar as dimensões da imagem
        altura, largura = linhas_detectadas.shape

        # 2. Criar um kernel com a largura TOTAL da imagem
        kernel_total = cv2.getStructuringElement(cv2.MORPH_RECT, (largura, 1))

        # 3. Aplicar a dilatação para esticar todas as linhas horizontalmente
        linhas_completas = cv2.dilate(linhas_detectadas, kernel_total, iterations=1)

        #mostrar(linhas_completas, "Linhas Conectadas até o Final"
        # =========================================================
            # 4. REMOVER LINHAS
            # =========================================================
        sem_linhas = cv2.bitwise_and(binary, cv2.bitwise_not(linhas_completas))

        #mostrar(sem_linhas, "4. Imagem Sem Linhas")


        # =========================================================
        # RECONSTRUIR SÍMBOLOS HORIZONTAIS
        # =========================================================

        #kernel_reconstrucao = cv2.getStructuringElement(
        #    cv2.MORPH_RECT,
        #    (15,1)
        #)

        #sem_linhas = cv2.dilate(
        #    sem_linhas,
        #    kernel_reconstrucao,
        #    iterations=1
        #)

        # afina novamente
        #sem_linhas = cv2.erode(
        #    sem_linhas,
        #    kernel_reconstrucao,
        #    iterations=1
        #)

        ##mostrar(sem_linhas, "4. Imagem Sem Linhas")

        # =========================================================
        # 5. PROJEÇÃO VERTICAL
        # =========================================================
        projecao = np.sum(sem_linhas, axis=0)

        #plt.figure(figsize=(15,4))
        #plt.plot(projecao)
        #plt.title("5. Projeção Vertical")
        #plt.xlabel("Colunas")
        #plt.ylabel("Soma dos pixels")
        #plt.grid()
        #plt.show()


        # =========================================================
        # 6. EXTRAÇÃO DOS SÍMBOLOS
        # =========================================================
        simbolos = []
        crops_primeiro_refinamento = []
        em_simbolo = False
        inicio = 0

        espaco_max_vazio = 6
        contador_vazio = 0

        for x, valor in enumerate(projecao):

                if valor > 500:

                    if not em_simbolo:
                        inicio = x
                        em_simbolo = True

                    contador_vazio = 0

                elif valor <= 500 and em_simbolo:

                    contador_vazio += 1

                    if contador_vazio > espaco_max_vazio:

                        fim = x - contador_vazio

                        recorte = img[
                            :,
                            max(0, inicio-5):min(img.shape[1], fim+5)
                        ]

                        simbolos.append(recorte)

                        em_simbolo = False
                        contador_vazio = 0


        # Último símbolo
        if em_simbolo:

                fim = len(projecao) - contador_vazio

                recorte = img[
                    :,
                    max(0, inicio-5):min(img.shape[1], fim+5)
                ]

                simbolos.append(recorte)


            # =========================================================
            # 7. #MOSTRAR SÍMBOLOS
            # =========================================================
        print(f"{len(simbolos)} símbolos encontrados")

        for idx, crop in enumerate(crops_primeiro_refinamento):

            cv2.imwrite(
                os.path.join(
                    "output_primeiro_refinamento",
                    f"{nome_base}_{idx:05d}.png"
                ),
                crop
            )

        # =====================================================
        # REFINAMENTO DOS SÍMBOLOS
        # =====================================================

        

        for indice_simbolo, simbolo in enumerate(simbolos):

            largura_simbolo = simbolo.shape[1]

            if verificar_pausa_inteira(simbolo):
                crops_primeiro_refinamento.append(simbolo)
                continue

            if largura_simbolo < 63:
                crops_primeiro_refinamento.append(simbolo)
                continue

            partes, centros = cortar_simbolo_largo(
                simbolo,
                largura_base=63,
                dist_minima_cabecas=30
            )

            if len(partes) == 0:
                partes = [simbolo]

            for parte in partes:

                if parte.shape[1] < 8:
                    continue

                crops_primeiro_refinamento.append(parte)

        crops_finais = []

        for indice_crop, crop in enumerate(crops_primeiro_refinamento):

            largura_crop = crop.shape[1]

            # símbolos pequenos não precisam de novo refinamento
            if largura_crop < 63:

                crops_finais.append(crop)
                continue

            partes_refinadas, cortes = processar_e_cortar_simbolo(
                crop,
                indice=indice_crop,
                largura_base=63,
                dist_minima_cabecas=30
            )

            if len(partes_refinadas) == 0:
                partes_refinadas = [crop]

            crops_finais.extend(partes_refinadas)

        for idx, crop in enumerate(crops_finais):

            cv2.imwrite(
                os.path.join(
                    PASTA_REFINADA,
                    f"{nome_base}_{idx:05d}.png"
                ),
            crop
            )

        # =====================================================
        # SEMANTIC
        # =====================================================

        labels_semantic = ler_semantic(
            caminho_semantic
        )

        qtd_crops = len(crops_finais)
        qtd_semantic = len(labels_semantic)

        print(f"\nArquivo: {nome_base}")
        print(f"Crops   : {qtd_crops}")
        print(f"Semantic: {qtd_semantic}")
       

        # =====================================================
        # SALVA DATASET
        # =====================================================

        PASTA_CORTES = "CORTES"

        # cria pasta da partitura
        pasta_partitura = os.path.join(
            PASTA_CORTES,
            nome_base
        )

        os.makedirs(
            pasta_partitura,
            exist_ok=True
        )

        # copia semantic
        shutil.copy2(
            caminho_semantic,
            os.path.join(
                pasta_partitura,
                f"{nome_base}.semantic"
            )
        )

        # copia agnostic
        shutil.copy2(
            caminho_agnostic,
            os.path.join(
                pasta_partitura,
                f"{nome_base}.agnostic"
            )
        )

        # salva os cortes
        for i, crop in enumerate(crops_finais, start=1):

            nome_saida = f"{nome_base}_{i}.png"

            caminho_saida = os.path.join(
                pasta_partitura,
                nome_saida
            )

            cv2.imwrite(
                caminho_saida,
                crop
            )
            # breakpoint()
       
main()

print("\n================================================")
print("FINALIZADO")
print("================================================")
