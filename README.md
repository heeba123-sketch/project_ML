# Déploiement — Détecteur Fruits & Légumes (Streamlit)

Cette app charge ton modèle `best_model_finetuned.tflite` (EfficientNetB0
fine-tuné, 28 classes : 14 fruits/légumes × sain/pourri) et permet de
tester une image via un simple upload.

## Contenu du dossier
```
streamlit_app/
├── app.py                     # l'application
├── requirements.txt           # dépendances
├── best_model_finetuned.tflite  # ton modèle (à copier ici)
└── README.md
```

## 1. Tester en local

```bash
cd streamlit_app
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Ça ouvre automatiquement `http://localhost:8501` dans ton navigateur.

## 2. Déployer gratuitement sur Streamlit Community Cloud

C'est la méthode la plus simple, gratuite, et ne demande pas de gérer un serveur.

1. **Crée un dépôt GitHub** et mets-y ces 3 fichiers (`app.py`,
   `requirements.txt`, et `best_model_finetuned.tflite` à la racine ou
   dans le même dossier — le fichier fait 8,4 Mo, ça passe largement).
2. Va sur **https://share.streamlit.io** et connecte-toi avec ton compte GitHub.
3. Clique sur **"New app"**, choisis ton dépôt, la branche, et indique
   `app.py` comme fichier principal.
4. Clique sur **Deploy**. Au bout de quelques minutes ton app est en ligne
   avec une URL du type `https://ton-app.streamlit.app`.

⚠️ Le seul point d'attention : la version de `tensorflow-cpu` dans
`requirements.txt` doit rester compatible avec Python 3.11/3.12 (celle
fournie fonctionne sur Streamlit Cloud).

## 3. Alternative — Hugging Face Spaces (gratuit aussi)

1. Crée un compte sur https://huggingface.co puis un nouveau **Space**.
2. Choisis le SDK **Streamlit**.
3. Upload les mêmes fichiers (`app.py`, `requirements.txt`, le `.tflite`).
4. Le Space se construit et se lance automatiquement.

## 4. Alternative — Docker (si tu veux héberger toi-même, ex. VPS)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Puis :
```bash
docker build -t fruits-app .
docker run -p 8501:8501 fruits-app
```
Tu peux ensuite déployer ce conteneur sur Railway, Render, Fly.io, un VPS, etc.

## Notes techniques importantes

- **Taille d'image attendue : 224×224**, RGB.
- **Pas de division par 255** : le modèle EfficientNetB0 intègre déjà sa
  normalisation en interne (`preprocess_input` d'EfficientNet est en fait
  une fonction identité) — c'est déjà géré dans `app.py`, ne pas y toucher.
- **Ordre des 28 classes** : trié alphabétiquement, déjà codé dans `app.py`
  (`CLASS_NAMES`). Si tu changes de dataset/modèle, il faut regénérer
  cette liste avec `sorted(os.listdir(...))` pour rester cohérent.
