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