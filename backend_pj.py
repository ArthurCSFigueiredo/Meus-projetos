from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


app = Flask(__name__)
CORS(app)  # Permitir todas as origens

# Carregar e processar a tabela
tabela = pd.read_csv("clientes.csv")
novos_id_cliente = range(100000)
tabela['id_cliente'] = novos_id_cliente

# Codificação dos dados categóricos
codificador = LabelEncoder()
tabela["profissao"] = codificador.fit_transform(tabela["profissao"])
tabela["mix_credito"] = codificador.fit_transform(tabela["mix_credito"])
tabela["comportamento_pagamento"] = codificador.fit_transform(tabela["comportamento_pagamento"])

# Preparação dos dados para o treinamento do modelo
y = tabela["score_credito"]
x = tabela.drop(columns=["score_credito"])
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.35)

# Treinamento do modelo
modelo_arvoredecisao = RandomForestClassifier()
modelo_arvoredecisao.fit(x_treino, y_treino)

def obter_informacoes_cliente_por_id(id_cliente):
    cliente = tabela[tabela['id_cliente'] == id_cliente]
    if cliente.empty:
        return None
    else:
        return cliente.drop(columns=["score_credito"])

@app.route('/prever/<int:id_cliente>', methods=['GET'])
def prever_score(id_cliente):
    cliente = obter_informacoes_cliente_por_id(id_cliente)
    if cliente is None:
        return jsonify({"error": "Cliente não encontrado"}), 404

    previsao = modelo_arvoredecisao.predict(cliente)
    return jsonify({"id_cliente": id_cliente, "score_credito_previsto": previsao[0]})

if __name__ == '__main__':
    app.run(debug=True)

