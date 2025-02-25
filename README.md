# Sistema de Recomendação de Notícias do G1

Repositório destinado ao desenvolvimento de um sistema de recomendação de notícias do portal **G1**, contemplando todas as etapas, desde o treinamento do modelo até a disponibilização via **API** e empacotamento em **Docker**.

## Sumário

1. [Visão Geral do Projeto](#visão-geral-do-projeto)  
2. [Arquitetura da Solução](#arquitetura-da-solução)  
3. [Coleta e Estrutura dos Dados](#coleta-e-estrutura-dos-dados)  
4. [Descrição do Modelo de Recomendação](#descrição-do-modelo-de-recomendação)  
   - [Por que usar Fatoração de Matrizes / Embeddings?](#por-que-usar-fatoração-de-matrizes--embeddings)  
   - [Como funciona o treinamento?](#como-funciona-o-treinamento)  
   - [Como funciona a recomendação (predição)?](#como-funciona-a-recomendação-predição)  
5. [Estrutura da API](#estrutura-da-api)  
   - [Endpoints Disponíveis](#endpoints-disponíveis)  
   - [Integração com o Modelo de Recomendação](#integração-com-o-modelo-de-recomendação)  
6. [Execução Local com Docker](#execução-local-com-docker)  
7. [Testes e Validação](#testes-e-validação)  
8. [Como Contribuir](#como-contribuir)  
9. [Licença](#licença)  
10. [Observação sobre o Vídeo](#observação-sobre-o-vídeo)

---

## Visão Geral do Projeto

O objetivo deste projeto é criar um sistema de recomendação para usuários do portal de notícias **G1**.  
Com base no histórico de leitura de cada usuário, o sistema sugere notícias relevantes, prevendo quais seriam as próximas notícias de maior interesse.

### Requisitos Principais

1. **Treinamento do modelo**  
2. **Salvamento do modelo**  
3. **Criação de uma API para previsões**  
4. **Empacotamento com Docker**  
5. **Testes e validação da API**

Este repositório contém o código-fonte completo da solução, desde a parte de *Machine Learning* (treinamento e salvamento de embeddings) até a disponibilização de uma **API** via **FastAPI**, além da preparação para execução em contêiner **Docker**.

---

## Arquitetura da Solução

A solução está organizada em duas grandes etapas:

1. **Treinamento do Modelo (Offline)**  
   - Leitura de dados de interações (usuário x notícia).  
   - Criação e treinamento dos embeddings de usuários e itens (notícias) usando **PyTorch**.  
   - Salvamento dos arquivos `user_embeddings.pt` e `item_embeddings.pt`.

2. **API de Recomendação (Online)**  
   - Endpoint para buscar detalhes de uma notícia específica.  
   - Endpoint para obter recomendações personalizadas para um usuário.  
   - Servido via **FastAPI**, com documentação interativa e empacotamento em **Docker**.

---

## Coleta e Estrutura dos Dados

- **Dados de Notícias**: Armazenados em `news.csv`, contendo:
  - **id** (ID da notícia)  
  - **title** (título)  
  - **issued** e **modified** (datas de criação e modificação)  
  - **url** (endereço da notícia)  

- **Embeddings**: Gerados após o treinamento do modelo, sendo:
  - `user_embeddings.pt`: Embeddings para cada usuário.  
  - `item_embeddings.pt`: Embeddings para cada notícia (item).

Os serviços da API carregam esses arquivos (CSV e `.pt`) para entregar as funcionalidades de consulta e recomendação.

---

## Descrição do Modelo de Recomendação

### Por que usar Fatoração de Matrizes / Embeddings?

Para problemas de recomendação, a técnica de **Filtragem Colaborativa** baseada em embeddings (fatoração de matrizes) é muito utilizada por:

- **Escalabilidade**: É possível lidar com grandes bases de usuários e itens.  
- **Representação Latente**: Cada dimensão do embedding captura uma característica implícita das preferências do usuário.  
- **Eficiência na Inferência**: Basta calcular similaridades (por exemplo, via cosseno) entre o embedding do usuário e dos itens para produzir recomendações.

### Como funciona o treinamento?

1. Coletamos interações entre usuários e notícias (por exemplo, cliques ou leituras).  
2. Definimos um modelo em **PyTorch** que gera:
   - Um embedding para cada usuário (`user_embedding`).  
   - Um embedding para cada item (notícia) (`item_embedding`).  
3. Durante o treinamento, ajustamos esses embeddings para maximizar a similaridade entre usuário e notícias efetivamente lidas/gostadas, e minimizar a similaridade com notícias que não foram lidas ou não são relevantes.  
4. Ao final, salvamos os arquivos de embeddings para uso em produção.

### Como funciona a recomendação (predição)?

1. Ao receber um `user_id`, carregamos o embedding correspondente (ex.: `user_embeddings[user_id]`).  
2. Calculamos a **similaridade de cosseno** entre esse vetor e todos os vetores de itens.  
3. Ordenamos os itens pela maior similaridade e retornamos os **top K** itens.  
4. Para cada item (notícia), buscamos detalhes (título, URL e datas) e retornamos ao cliente.

---

## Estrutura da API

A API é construída em **FastAPI** e possui rotas organizadas no diretório `app/controllers`.  
A configuração principal está no arquivo `main.py`.

### Endpoints Disponíveis

1. **`GET /news/{news_id}`**  
   - **Descrição**: Retorna os dados de uma notícia específica (ID, título, datas, URL).

2. **`GET /news/recommendation/{user_id}?top_k=N`**  
   - **Descrição**: Retorna as **top_k** notícias recomendadas para um determinado usuário.  
   - **Parâmetros**:  
     - `user_id`: ID do usuário.  
     - `top_k`: número de itens a serem retornados (padrão = 5, máximo = 20).

### Integração com o Modelo de Recomendação

- **Carregamento de Embeddings**: Nos serviços de recomendação (`GetRecommendationNewsByUserIdService`), usamos `load_saved_user_embeddings()` e `load_saved_item_embeddings()` para carregar os arquivos `.pt`.  
- **Cálculo de Similaridade**: Usamos `torch.nn.functional.cosine_similarity` para medir a proximidade entre o embedding do usuário e cada embedding de item.  
- **Retorno das Recomendações**: Após identificar os itens mais similares, retornamos detalhes de cada notícia por meio do serviço `GetNewsService`.

---

## Execução Local com Docker

Para executar a aplicação em contêiner via Docker:

------------- EM CONSTRUÇÃO -------------

## API - Recomendação de Notícias do Portal G1

### Rodando o projeto

Para configurar e executar o projeto, siga os seguintes passos:

1. Edite as variáveis do arquivo `.env-example` para corresponder ao seu
ambiente, em seguida renomeie o arquivo apenas para `.env`. As variáveis são:
   - RESOURCES_PATH (diretório onde estão salvos os arquivos do Pytorch `.pt`)

2. Instalar as dependências necessárias (lembre-se de utilizar virtual env):
```
pip install -r requirements.txt
```

3. Executar a aplicação:
```
uvicorn main:app --reload
```
