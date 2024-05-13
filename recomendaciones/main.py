import pandas as pd
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import classification_report, confusion_matrix

# Leer el JSON
with open('data/australian_user_reviews.json', 'r') as file:
    data = file.readlines()

# Define lists to store parsed data
user_ids = []
user_urls = []
review_items = []
review_texts = []
recommendations = []

for line in data:
    entry = eval(line)  # Se usa eval para parse JSON
    
    # Extraer información de usuario
    user_id = entry['user_id']
    user_url = entry['user_url']
    
    # Extraer reviews
    for review in entry['reviews']:
        user_ids.append(user_id)
        user_urls.append(user_url)
        review_items.append(review['item_id'])
        review_texts.append(review['review'])
        recommendations.append(review['recommend'])

# Crea DataFrame
df = pd.DataFrame({
    'user_id': user_ids,
    'user_url': user_urls,
    'item_id': review_items,
    'review': review_texts,
    'recommend': recommendations
})

# Paso #1: Vectorizar
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['review'])

# Paso #2: Calcular similitudes
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

# Paso #3: Calcular recomendaciones
def recommend(user_id, cosine_similarities=cosine_similarities):
	user_index = df[df['user_id'] == user_id].index.tolist()	
	scores = list(enumerate(cosine_similarities[user_index[0]]))
	scores = sorted(scores, key=lambda x: x[1], reverse=True)
	scores = scores[1:11]
	item_indices = [i[0] for i in scores]
	return df['item_id'].iloc[item_indices].tolist()

# Paso #4: Evaluar resultados con un usuario aleatorio
random_user_id = random.choice(df['user_id'].unique())
recommendations = recommend(random_user_id)
print("La recomendación para el usuario", random_user_id, "es:" , recommendations)
