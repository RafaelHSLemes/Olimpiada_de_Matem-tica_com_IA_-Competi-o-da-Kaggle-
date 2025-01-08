![image](https://github.com/user-attachments/assets/712141d8-78ce-4ee2-84ff-2e703af78924)

![image](https://github.com/user-attachments/assets/7053c40a-51cf-454c-b614-d2bfc7231d3f)

![image](https://github.com/user-attachments/assets/14ef6861-c1a1-45ec-bd4c-97e4f6aa57bf)

Estrutura do Repositório:

O repositório conta com arquivos de entrada para o desenvolvimento de um modelo de IA, para a realização de predição de resultados das situações-problema matemáticas da olimpíada em questão, bem como exemplos de saída e arquivo de entrada teste. A pasta "kaggle_evaluation" contém a API requisito da competição com seus respectivos módulos para uso. 

As questões da olimpíada são apresentadas no arquivo PDF "AIMO_Progress_Prize_2_Reference_Problems_Solutions" e o notebook com o código submetido na competição é o arquivo "AIMO_RHSLnotebook.py" com sua saída em "submission.csv".

Configuração do Ambiente:

As bibliotecas utilizadas foram OS, Pandas e Polars para o tratamento de dados e Scikit-Learn e XGBoost para o desempenho do modelo de IA desenvolvido.

Fase 1: Estimativa de 100 pontos

![image](https://github.com/user-attachments/assets/152be07a-99e7-4216-8338-230034055468)
Considerando as 1000 palavras mais relevantes do texto, esse vetorizador TF-IDF está sendo usado para transformar texto em expressões numéricas de modo que possam ser usadas como entrada para modelos de machine learning;

![image](https://github.com/user-attachments/assets/cefe47b8-b419-4d1c-b00d-a6b38ab788eb)
Instanciando o algoritmo de machine learning XGBoost, baseado em árvores de decisão, o modelo de classificação é projetado para prever categorias ou classes em vez de valores contínuos. No processo de boosting usado nesse código utiliza 100 árvores de decisão simples para criar um modelo mais forte e mais robusto, onde cada árvore tenta corrigir os erros das árvores anteriores. O codificador de rótulos embutido é desativado por conta de os rótulos já estarem codificados, lidando com classificação multiclasse, o modelo retorna o índice da classe prevista. 
