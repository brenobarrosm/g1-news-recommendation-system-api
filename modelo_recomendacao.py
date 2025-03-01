from modelo import LightGCN  # Importe a classe correta
import torch
import torch.nn.functional as F

# Ajuste o número de usuários e itens para corresponder ao modelo salvo
num_usuarios = 577942  # Número de usuários do modelo salvo
num_itens = 255603     # Número de itens do modelo salvo
dim_emb = 64           # Dimensão do embedding (já estava configurada)

# Recria o modelo com as mesmas configurações usadas no treinamento
modelo = LightGCN(num_usuarios, num_itens, dim_emb)

# Carrega os pesos salvos
modelo.load_state_dict(torch.load("lightgcn_model.pt", map_location=torch.device('cpu')))
modelo.eval()  # Coloca o modelo em modo de avaliação

print("Modelo carregado e pronto para inferência!")

# Carrega os embeddings do usuário e do item
user_embeddings = torch.load("user_embeddings.pt", map_location=torch.device('cpu'))
item_embeddings = torch.load("item_embeddings.pt", map_location=torch.device('cpu'))

def recomendar(user_id, top_k=10):
    """Retorna os top_k itens recomendados para um usuário específico."""
    if user_id >= len(user_embeddings):
        raise ValueError(f"User ID {user_id} fora do range ({len(user_embeddings)} usuários disponíveis).")

    # Pega o embedding do usuário
    user_vector = user_embeddings[user_id].unsqueeze(0)  # Adiciona uma dimensão
    
    # Calcula a similaridade entre o usuário e todos os itens
    scores = F.cosine_similarity(user_vector, item_embeddings)  # Usa similaridade de cosseno
    
    # Pega os top_k itens mais similares
    top_items = torch.argsort(scores, descending=True)[:top_k]
    
    return top_items.tolist()  # Retorna os índices dos itens recomendados

if __name__ == "__main__":
    user_id = 0  # Alterar para testar com diferentes IDs
    top_k = 5  # Quantidade de itens recomendados
    print(f"Itens recomendados para o usuário {user_id}: {recomendar(user_id, top_k)}")
