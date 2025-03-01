import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import time
import torch.nn.functional as F
import scipy.sparse as sp

# Diretórios dos arquivos
DIR_TREINO = Path("C:/Users/maxue/Music/treino")
DIR_ITENS = Path("C:/Users/maxue/Music/itens")

# Definir o tamanho do lote
chunk_size = 100000  # Tamanho do lote, ajuste conforme necessário

# Função para processar o lote de treino
def processar_lote(df):
    # Remover a coluna 'timestampHistory_new', se existir
    df.drop(columns=['timestampHistory_new'], errors='ignore', inplace=True)

    # Converter colunas de listas (garantindo tratamento de NaN)
    list_columns = [
        'history', 'timestampHistory', 'numberOfClicksHistory',
        'timeOnPageHistory', 'scrollPercentageHistory', 'pageVisitsCountHistory'
    ]
    
    for col in list_columns:
        df[col] = df[col].fillna('').astype(str).apply(lambda x: [i.strip() for i in x.split(',')] if x else [])

    # Conversão de tipos apropriados
    df['timestampHistory'] = df['timestampHistory'].apply(lambda x: [int(i) for i in x if i.isdigit()])
    df['numberOfClicksHistory'] = df['numberOfClicksHistory'].apply(lambda x: [int(i) for i in x if i.isdigit()])
    df['timeOnPageHistory'] = df['timeOnPageHistory'].apply(lambda x: [int(i) for i in x if i.isdigit()])
    df['scrollPercentageHistory'] = df['scrollPercentageHistory'].apply(lambda x: [float(i) for i in x if i.replace(".", "", 1).isdigit()])

    # Normalizar as colunas
    scaler = MinMaxScaler()

    for col in ['timeOnPageHistory', 'scrollPercentageHistory']:
        df[col] = df[col].apply(lambda x: np.array(x).reshape(-1, 1) if len(x) > 0 else np.array([]))
        df[col] = df[col].apply(lambda x: scaler.fit_transform(x).flatten() if len(x) > 0 else x)

    df['pageVisitsCountHistory'] = df['pageVisitsCountHistory'].apply(lambda x: [int(i) for i in x if i.isdigit()])

    return df

# Função para processar os arquivos de itens
def processar_lote_itens(df):
    # Remover colunas indesejadas
    df.drop(columns=['body', 'caption', 'url'], errors='ignore', inplace=True)

    return df

# Carregar e processar os arquivos de itens
arquivos_itens = list(DIR_ITENS.glob("*.csv"))

df_itens_processado = []

for arquivo in arquivos_itens:
    for chunk in pd.read_csv(arquivo, dtype=str, chunksize=chunk_size, low_memory=False):
        df_itens_processado.append(processar_lote_itens(chunk))

# Concatenar todos os lotes processados
df_itens = pd.concat(df_itens_processado, ignore_index=True) if df_itens_processado else pd.DataFrame()

# Carregar e processar os arquivos de treino
arquivos_treino = list(DIR_TREINO.glob("*.csv"))

df_treino_processado = []

for arquivo in arquivos_treino:
    for chunk in pd.read_csv(arquivo, dtype=str, chunksize=chunk_size, low_memory=False):
        df_treino_processado.append(processar_lote(chunk))

# Concatenar todos os lotes processados
df_treino = pd.concat(df_treino_processado, ignore_index=True) if df_treino_processado else pd.DataFrame()

# Explodir 'history' e garantir que outras colunas de lista também sejam explodidas
if not df_treino.empty:
    df_treino['history_len'] = df_treino['history'].apply(len)
    df_treino = df_treino[df_treino['history_len'] > 0]  # Remove linhas onde 'history' está vazio
    df_treino = df_treino.explode(['history', 'timestampHistory', 'numberOfClicksHistory', 
                                    'timeOnPageHistory', 'scrollPercentageHistory', 'pageVisitsCountHistory'])

# Unir os dados de treino com os itens usando 'history' como chave (equivalente à coluna 'page' nos itens)
df_unido = df_treino.merge(df_itens, left_on='history', right_on='page', how='left')
df_unido.fillna({'issued': '', 'modified': '', 'title': ''}, inplace=True)

# Agrupar novamente para reconstruir o formato original, mas com informações dos itens associadas
df_final = df_unido.groupby(['userId', 'userType', 'historySize']).agg(
    {
        'history': lambda x: list(x),
        'timestampHistory': lambda x: list(x),
        'numberOfClicksHistory': lambda x: list(x),
        'timeOnPageHistory': lambda x: list(x),
        'scrollPercentageHistory': lambda x: list(x),
        'pageVisitsCountHistory': lambda x: list(x),
        'page': lambda x: list(x.dropna()),
        'issued': lambda x: list(x.dropna()),
        'modified': lambda x: list(x.dropna()),
        'title': lambda x: list(x.dropna())
    }
).reset_index()

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # Dropout para evitar overfitting
        self.dropout = nn.Dropout(p=0.2)  # 🔹 Ajuste a taxa conforme necessário

        # Camadas de embedding para usuários e itens
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.reset_parameters()  # Inicialização Xavier

    def reset_parameters(self):
        """ Inicializa os embeddings usando Xavier Uniform """
        torch.nn.init.xavier_uniform_(self.user_embedding.weight)
        torch.nn.init.xavier_uniform_(self.item_embedding.weight)
        # torch.nn.init.normal_(self.user_embedding.weight, mean=0, std=0.01)
        # torch.nn.init.normal_(self.item_embedding.weight, mean=0, std=0.01)

    def forward(self, user, item):
        """ Retorna os embeddings de usuários e itens dados os IDs de usuário e item """
        # user_embeddings = self.user_embedding(user)
        # item_embeddings = self.item_embedding(item)
        user_embeddings = self.dropout(self.user_embedding(user))
        item_embeddings = self.dropout(self.item_embedding(item))
        return user_embeddings, item_embeddings

# Função para criar o grafo de interações
def criar_grafo(df, alpha=0.9, beta=0.8):
    df = df.explode(['history', 'timeOnPageHistory', 'scrollPercentageHistory'])

    # Codificar usuários e itens
    df['userId_code'] = df['userId'].astype('category').cat.codes
    df['item_code'] = df['history'].astype('category').cat.codes

    # Converter colunas para valores numéricos
    df['timeOnPageHistory'] = df['timeOnPageHistory'].astype(float)
    df['scrollPercentageHistory'] = df['scrollPercentageHistory'].astype(float)

    # Normalizar os valores entre 0 e 1
    scaler = MinMaxScaler()
    df[['timeOnPageHistory', 'scrollPercentageHistory']] = scaler.fit_transform(df[['timeOnPageHistory', 'scrollPercentageHistory']])

    # Calcular os pesos das interações
    df['interaction_weight'] = alpha * df['timeOnPageHistory'] + beta * df['scrollPercentageHistory']

    # Criar a matriz esparsa
    user_ids = df['userId_code'].values
    item_ids = df['item_code'].values
    weights = df['interaction_weight'].values  # Agora os pesos variam, em vez de serem apenas 1s

    interaction_matrix = coo_matrix((weights, (user_ids, item_ids)))

    return interaction_matrix

# Criar matriz de interação ponderada
interaction_matrix = criar_grafo(df_final)

# Converter para formato esparso adequado
interaction_matrix = interaction_matrix.tocsr()

# Separar os pares usuário-item
nonzero_users, nonzero_items = interaction_matrix.nonzero()
weights = interaction_matrix.data  # Pega os pesos das interações

user_item_pairs = list(zip(nonzero_users, nonzero_items, weights))

# Separar treino e teste
train_pairs, test_pairs = train_test_split(user_item_pairs, test_size=0.2, random_state=42)

# Criar matrizes esparsas para treino e teste usando os pesos
train_matrix = csr_matrix(
    (np.array([w for _, _, w in train_pairs]), 
     (np.array([u for u, i, _ in train_pairs]), np.array([i for u, i, _ in train_pairs]))),
    shape=interaction_matrix.shape
).tocoo()

test_matrix = csr_matrix(
    (np.array([w for _, _, w in test_pairs]), 
     (np.array([u for u, i, _ in test_pairs]), np.array([i for u, i, _ in test_pairs]))),
    shape=interaction_matrix.shape
).tocoo()

print(f"Treino: {train_matrix.shape}, Teste: {test_matrix.shape}")

# Dataset para o DataLoader
class RecommendationDataset(Dataset):
    def __init__(self, interaction_matrix):
        self.user_item_matrix = interaction_matrix

    def __len__(self):
        return len(self.user_item_matrix.row)

    def __getitem__(self, idx):
        user = self.user_item_matrix.row[idx]
        item = self.user_item_matrix.col[idx]
        return user, item

# DataLoader para carregar os dados
train_dataset = RecommendationDataset(train_matrix)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=False)

# Criando o dataset de teste
test_dataset = RecommendationDataset(test_matrix)

# Criando o DataLoader de teste
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=False)

# Inicializar e treinar o modelo LightGCN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LightGCN(num_users=interaction_matrix.shape[0], num_items=interaction_matrix.shape[1], embedding_dim=64).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, data_loader, optimizer, device):
# def train(model, data_loader, optimizer, device, max_batches=500): #debug
    model.train()
    total_loss = 0
    start_time = time.time()  # Iniciar temporizador

    # Calcular a perda usando BCEWithLogitsLoss
    loss_fn = nn.BCEWithLogitsLoss()
    loop = tqdm(data_loader, total=len(data_loader), desc="Treinando", leave=True)

    for batch_idx, (user, item) in enumerate(loop):
        # if batch_idx >= max_batches:  # Parar após 1000 iterações (Debug)
        #     break
        user = user.to(device)
        item = item.to(device)

        # Obter embeddings do usuário e item
        user_embedding, item_embedding = model(user, item)

        # Normalizar os embeddings
        user_embedding = F.normalize(user_embedding, p=2, dim=1)
        item_embedding = F.normalize(item_embedding, p=2, dim=1)

        # Computar as pontuações (produto escalar)
        scores = (user_embedding * item_embedding).sum(dim=1)

        # # Exibir mínimo e máximo dos scores
        # print(f"Scores mínimos e máximos: {scores.min().item()}, {scores.max().item()}")
        # print(f"Norma dos embeddings - Usuário: {torch.norm(user_embedding, p=2).item()}, Item: {torch.norm(item_embedding, p=2).item()}")

        # Criar rótulos: 1 para interações positivas
        labels = torch.ones_like(scores)
        
        # Calcular a perda com regularização   
        loss = loss_fn(scores, labels) + 0.0001 * (torch.norm(user_embedding, p=2) + torch.norm(item_embedding, p=2))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # # Exibir gradientes
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name} - Gradiente médio: {param.grad.abs().mean().item()}")

        optimizer.step()

        total_loss += loss.item()

        loop.set_postfix(loss=total_loss / (batch_idx + 1))  # Evitar divisão por 'n'

    epoch_time = time.time() - start_time
    print(f"Epoch time: {epoch_time:.2f} seconds")
    return total_loss / len(data_loader)

def get_average_item_embedding(model, train_items):
    # Converter train_items para uma lista ou tensor antes da indexação
    train_items = torch.tensor(list(train_items), dtype=torch.long, device=model.item_embedding.weight.device)
    
    # Calcular a média dos embeddings dos itens treinados
    item_embeddings = model.item_embedding.weight[train_items]
    return torch.mean(item_embeddings, dim=0)

def evaluate(model, test_loader, train_items, device, k=20):
    model.eval()
    total_precision, total_recall, total_ndcg = 0, 0, 0
    num_samples = 0

    valid_items = set(train_items)  # Itens que o modelo já viu no treino
    avg_item_embedding = get_average_item_embedding(model, train_items).to(device)  # Média dos embeddings

    with torch.no_grad():
        for user, item in test_loader:
            user = user.to(device)
            item = item.to(device)

            # Se o item não foi visto no treino, usa o embedding médio
            item_embedding = model.item_embedding(item)
            item_embedding = torch.where(item_embedding == 0, avg_item_embedding, item_embedding)

            user_embedding, item_embedding = model(user, item)
            # Computar as pontuações (produto escalar)
            scores = (user_embedding * item_embedding).sum(dim=1)

            # Obtém os top-K itens recomendados
            _, top_k_items = torch.topk(scores, k)

            # Garantir que `top_k_items` seja uma lista plana de inteiros
            top_k_items = [int(i) for i in top_k_items.flatten().tolist()]

            # Filtrar itens recomendados que não estavam no treino
            top_k_filtered = [i for i in top_k_items if i in valid_items]

            # Interseção entre os itens recomendados e os itens verdadeiros
            intersecao = set(top_k_filtered) & set(item.tolist())

            # Calcula métricas de precisão e recall
            relevant = (item.unsqueeze(1) == torch.tensor(top_k_items, dtype=torch.int32, device=device)).float()
            
            # Precisão (dentro dos k recomendados)
            precision = relevant.sum() / k  # Calculando a precisão para top-k
            
            # Recall (item relevante no top-k, dividido pelo número total de itens relevantes)
            recall = relevant.sum() / len(set(item.tolist()))  # Dividindo pelo número total de itens relevantes

            # Cálculo de NDCG (Normalização por log da posição)
            log_positions = torch.log2(torch.arange(2, k + 2, device=device).float())

            # Se houver interseção, calcule DCG e IDCG
            if len(intersecao) > 0:
                # Computar DCG (Discounted Cumulative Gain)
                dcg = (relevant / log_positions).sum()

                # Ideal DCG (IDCG) para o top-k
                ideal_relevant = torch.tensor([1.0] * len(intersecao), device=device)  # IDCG é 1 se todos os itens forem relevantes

                if len(intersecao) > 0:
                    idcg = (ideal_relevant / log_positions[:len(intersecao)]).sum()
                else:
                    idcg = 0  # ou idcg = 1, dependendo da implementação

                # NDCG = DCG / IDCG
                ndcg = dcg / idcg if idcg > 0 else 0  # Evitar divisão por zero

            else:
                # Caso não haja interseção, defina NDCG como 0
                ndcg = 0

            total_precision += precision.item()
            total_recall += recall.item()
            total_ndcg += ndcg
            num_samples += 1

    avg_precision = total_precision / num_samples
    avg_recall = total_recall / num_samples
    avg_ndcg = total_ndcg / num_samples

    return avg_precision, avg_recall, avg_ndcg

# Identificar itens vistos no treino e no teste
train_items = set(train_matrix.col)
test_items = set(test_matrix.col)

# print(f"Itens no treino: {len(train_items)}, Itens no teste: {len(test_items)}")
# print(f"Itens do teste que NÃO estão no treino: {len(test_items - train_items)}")
# print("Variância dos embeddings de usuário:", model.user_embedding.weight.var().item())
# print("Variância dos embeddings de item:", model.item_embedding.weight.var().item())

# Parâmetros do Early Stopping
patience = 5  # Número de épocas sem melhoria para parar o treinamento
best_loss = float('inf')  # Melhor perda (começa como infinita)
patience_counter = 0  # Contador de paciência

num_epochs = 10
for epoch in range(num_epochs):
    # Treinando o modelo
    loss = train(model, train_loader, optimizer, device)

    # Avaliando as métricas
    precision, recall, ndcg = evaluate(model, test_loader, train_items, device, k=20)

    # Exibindo os resultados da época
    #print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Precision@20: {precision:.4f}, Recall@20: {recall:.4f}, NDCG@20: {ndcg:.4f}")
    
    # Verificando se a perda melhorou
    if loss < best_loss:
        best_loss = loss  # Atualiza a melhor perda
        patience_counter = 0  # Reinicia o contador de paciência
    else:
        patience_counter += 1  # Incrementa o contador se não houver melhoria

    # Verifica se o treinamento deve parar (baseado na paciência)
    if patience_counter >= patience:
        print(f"Early stopping: no improvement after {patience} epochs.")
        break

# Salvar o modelo completo
torch.save(model.state_dict(), "lightgcn_model.pt")

# Salvar apenas os embeddings
torch.save(model.user_embedding.weight.detach().cpu(), "user_embeddings.pt")
torch.save(model.item_embedding.weight.detach().cpu(), "item_embeddings.pt")

print("Modelo e embeddings salvos com sucesso! ✅")