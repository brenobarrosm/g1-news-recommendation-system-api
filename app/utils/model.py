import logging

import torch
import torch.nn as nn

from app.utils.settings import settings

class LightGCN(nn.Module):
    def __init__(self, num_users: int, num_items:int, embedding_dim: int):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        self.dropout = nn.Dropout(p=0.2)

        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """ Inicializa os embeddings usando Xavier Uniform """
        torch.nn.init.xavier_uniform_(self.user_embedding.weight)
        torch.nn.init.xavier_uniform_(self.item_embedding.weight)
        torch.nn.init.normal_(self.user_embedding.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.item_embedding.weight, mean=0, std=0.01)

    def forward(self, user, item):
        """ Retorna os embeddings de usuários e itens dados os IDs de usuário e item """
        user_embeddings = self.dropout(self.user_embedding(user))
        item_embeddings = self.dropout(self.item_embedding(item))
        return user_embeddings, item_embeddings


def get_model(num_users: int, num_items: int, embedding_dim: int):
    model = LightGCN(num_users, num_items, embedding_dim)
    model.load_state_dict(torch.load(settings.RESOURCES_PATH + "lightgcn_model.pt",
                                     map_location=torch.device('cpu')))
    model.eval()
    logging.info("Modelo carregado com sucesso!")
    return model
