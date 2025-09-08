# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 22:28:07 2025

@author: davih
"""

# pip install selenium webdriver-manager pandas lxml

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from time import sleep
from io import StringIO
from selenium.common.exceptions import TimeoutException

#%% Lista papéis da classificação setorial
#Código antigo

# setores = pd.read_excel("ClassifSetor.xlsx", sheet_name=0)
# setores = setores.ffill()
# setores = setores.dropna()
# setores_novo = setores[setores["SEGMENTO DE NEGOCIAÇÃO"] == "Novo Mercado"]
# codigos_novo = setores_novo["CÓDIGO"].tolist()

#%% Lista IBOV

# df_papeis = pd.read_csv('IBOVDiaX.csv', sep=';', encoding='latin1', skiprows=1, index_col=False)
# codigos_novo = df_papeis.iloc[:, 0]  # Obtém a primeira coluna (coluna A)
# codigos_novo = codigos_novo[:-2].tolist()  # Converte para lista

#%% Lista todos_tickers

codigos_novo = pd.read_csv("tickers.csv", header=None)[0].tolist()

#%%

options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
options.add_argument(r"--user-data-dir=C:\Users\davih\AppData\Local\Google\Chrome\SeleniumProfile")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
wait = WebDriverWait(driver, 15)

dfs = []

for papel in codigos_novo:
    print(f"Iniciando {papel.lower()}3...")
    url = f"https://investidor10.com.br/acoes/{papel.lower()}/"
    driver.get(url)
    
    # aguarda: ou carrega a seção válida, ou dá timeout e pulamos
    try:
        WebDriverWait(driver, 12).until(
            lambda d: d.find_elements(By.ID, "indicators-history") or
                      "404" in d.title.lower() or
                      d.find_elements(By.XPATH, "//h1[contains(translate(., 'ÁÂÃÀAÓÒÔÕOÉÈÊEÍÌÎIÚÙÛUÇ', 'AAAAAOOOOOEEEII IUUUC'), 'PAGINA') and contains(translate(., 'NÃO ENCONTRADA', 'NAO ENCONTRADA'), 'NAO ENCONTRADA')]")
        )
    except TimeoutException:
        print(f"Timeout em {papel}. Pulando.")
        continue
    
    # se não existe a seção, considera inválido e pula
    if not driver.find_elements(By.ID, "indicators-history"):
        print(f"Pulando {papel}: ticker inválido ou página sem histórico.")
        continue

    # --- CAPTURA INFORMAÇÕES SOBRE A EMPRESA (VALORES DETALHADOS) ---
    sec_info = WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div#info_about"))
    )
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", sec_info)
    driver.execute_script("window.scrollBy(0, -120);")
    
    # seleciona o <select> correto da seção "INFORMAÇÕES SOBRE A EMPRESA"
    sel = WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "select[name='company-values-view']"))
    )
    
    # define "Valores Detalhados" (value = detail-value) e dispara change
    driver.execute_script("""
      const s = arguments[0];
      s.value = 'detail-value';
      s.dispatchEvent(new Event('change', {bubbles: true}));
    """, sel)
    
    # capturar cards
    # rola até a seção "INFORMAÇÕES SOBRE A EMPRESA"
    sec_info = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "//div[.//h2[contains(.,'INFORMAÇÕES SOBRE A EMPRESA')]]"))
    )
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", sec_info)
    driver.execute_script("window.scrollBy(0, -120);")
        
    cards = driver.find_elements(By.CSS_SELECTOR, "#table-indicators-company .cell")
    infos = {}
    
    for c in cards:
        titulo = c.find_element(By.CSS_SELECTOR, ".title").text.strip()
        # valor pode estar num div.detail-value ou direto em .value
        try:
            valor = c.find_element(By.CSS_SELECTOR, ".detail-value").text.strip()
            if not valor:
                valor = c.find_element(By.CSS_SELECTOR, ".value").text.strip()
        except:
            valor = c.find_element(By.CSS_SELECTOR, ".value").text.strip()
    
        infos[titulo] = valor
        
    # --- RENTABILIDADE HISTÓRICA (linha "Rentabilidade") ---
    infos_rent = {}
    
    map_periodo = {
        "1 Mês": "1M",
        "3 Meses": "3M",
        "1 Ano": "1A",
        "2 Anos": "2A",
        "5 Anos": "5A",
        "10 Anos": "10A",
    }
    
    # Aguarda a presença de um título “Rentabilidade” e localiza o bloco pai
    sec_rent = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "//h3[contains(text(), 'Rentabilidade')]/ancestor::div[contains(@class, 'content')]"))
    )
    
    # Rola até a seção
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", sec_rent)
    driver.execute_script("window.scrollBy(0, -120);")
    
    # Pega todos os blocos de rentabilidade (ignora os blocos de “Rentabilidade Real”)
    blocos = sec_rent.find_elements(By.XPATH, ".//div[@class='result-period']")
    
    for bloco in blocos[:6]:  # pega apenas os 6 primeiros (rentabilidade nominal)
        try:
            valor = bloco.find_element(By.TAG_NAME, "span").text.strip()
            periodo = bloco.find_element(By.TAG_NAME, "h4").text.strip()
            chave = f"Rentab_{map_periodo.get(periodo, periodo)}"
            infos_rent[chave] = valor
        except Exception as e:
            print("Erro ao extrair rentabilidade:", e)

    # --- HISTÓRICO DE INDICADORES FUNDAMENTALISTAS ---
    
    # Localizar a seção e o botão
    secao = WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.ID, "indicators-history")))
    botao_expandir = WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="indicators-history"]/div[2]/button'))
    )
    
    # Trazer o botão para o centro da tela e ajustar offset (evita header fixo/interceptação)
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", botao_expandir)
    driver.execute_script("window.scrollBy(0, -120);")
    
    # Clicar via JavaScript (contorna ElementClickInterceptedException)
    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="indicators-history"]/div[2]/button')))
    driver.execute_script("arguments[0].click();", botao_expandir)
    
    # Opcional: aguardar a tabela expandir (mais linhas renderizadas)
    WebDriverWait(driver, 10).until(
        lambda d: len(secao.find_elements(By.CSS_SELECTOR, "tbody tr")) > 10
    )
    
    # Capturar a tabela e criar df
    tabela = secao.find_element(By.TAG_NAME, "table")
    html = tabela.get_attribute("outerHTML")
    df = pd.read_html(StringIO(html), decimal=',', thousands='.', converters={0: str})[0]

    df.rename(columns={df.columns[0]: "Indicador"}, inplace=True)
        
    # Transpor: anos viram índice e indicadores viram colunas
    df_t = df.set_index("Indicador").T.reset_index()
    
    # Renomear a coluna de índice (que são anos/atual)
    df_t.rename(columns={"index": "Ano"}, inplace=True)
    
    # Adicionar a coluna PAPEL
    df_t.insert(0, "PAPEL", papel)
    
    # incluir infos como colunas
    for k, v in infos.items():
        df_t[k] = v
        
    # incluir rentabilidade como colunas
    for k, v in infos_rent.items():
        df_t[k] = v
    
    dfs.append(df_t)
    
    sleep(1)
    
# concatenar tudo em um único dataframe
df_final = pd.concat(dfs, ignore_index=True)

driver.quit()

df_final.to_csv('df_investidor10.csv', index=False)