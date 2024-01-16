import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk, joblib, re, string
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

nltk.download("punkt")
nltk.download("stopwords")

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
stemmer = StemmerFactory().create_stemmer()


def read_data(file):
    if file is not None:
        df = pd.read_excel(file)
        return df
    return None


def pelabelan(data):
    data["label"] = data["label"].map({"positif": 1, "negatif": 0})
    data = data[["text", "label"]]
    return data


def cleaningText(text):
    text = re.sub(r"@[A-Za-z0-9]+", "", text)
    text = re.sub(r"#[A-Za-z0-9]+", "", text)
    text = re.sub(r"RT[\s]", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[0-9]+", "", text)

    text = text.replace("\n", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.strip(" ")
    return text


def casefoldingText(text):
    text = text.lower()
    return text


slangword_dict = {
    "bgt": "banget",
    "gak": "tidak",
    "yg": "yang",
}


def convert_slangwords(text, slangword_dict):
    words = text.split()
    converted_words = []
    for word in words:
        if word in slangword_dict:
            converted_words.append(slangword_dict[word])
        else:
            converted_words.append(word)
    return " ".join(converted_words)


def tokenizingText(text):
    text = word_tokenize(text)
    return text


def filteringText(text):
    stopwords_indonesian = stopwords.words("indonesian")
    stopwords_indonesian.remove("ada")
    stopwords_indonesian.remove("kurang")
    stopwords_indonesian.remove("tidak")
    filtered = []
    for txt in text:
        if txt not in stopwords_indonesian:
            filtered.append(txt)
    text = filtered
    return text


def stemmingText(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = [stemmer.stem(word) for word in text]
    return text


def join_text_list(data):
    data["text"] = data["text"].apply(lambda x: " ".join(x))
    return data


def preprocess(text):
    text = cleaningText(text)
    text = casefoldingText(text)
    text = convert_slangwords(text, slangword_dict)
    text = tokenizingText(text)
    text = filteringText(text)
    text = stemmingText(text)
    return text


def preprocess_and_print(text):
    processed_text = preprocess(text)
    print(f"Original Text: {text}")
    print(f"Processed Text: {processed_text}")
    print("")
    return processed_text


def tf_idf(data):
    vectorizer = TfidfVectorizer()
    tfidf_df = pd.DataFrame(
        vectorizer.fit_transform(data["text"]).toarray(),
        columns=vectorizer.get_feature_names_out(),
    )
    tfidf_vectors = vectorizer.fit_transform(data["text"])
    return tfidf_df, tfidf_vectors


def split_data(a, b):
    from sklearn.model_selection import train_test_split

    X = a
    y = b["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def proses_pca(a, b, n):
    data = split_data(a, b)
    pca = PCA(n_components=n)
    X_train_pca = pca.fit_transform(data[0].toarray())
    return X_train_pca, pca.explained_variance_


def fold(a, b, n, c):
    data = split_data(a, b)[2]
    classifier = GaussianNB()
    kf = KFold(n_splits=n, shuffle=True, random_state=42)
    results = []
    for fold, (train_index, test_index) in enumerate(kf.split(c), 1):
        X_train_fold, X_test_fold = c[train_index], c[test_index]
        y_train_fold, y_test_fold = data.iloc[train_index], data.iloc[test_index]
        classifier.fit(X_train_fold, y_train_fold)
        y_pred_fold = classifier.predict(X_test_fold)
        accuracy = round(accuracy_score(y_test_fold, y_pred_fold) * 100, 2)
        precision = round(
            precision_score(
                y_test_fold, y_pred_fold, average="weighted", zero_division=1
            )
            * 100,
            2,
        )
        recall = round(
            recall_score(y_test_fold, y_pred_fold, average="weighted", zero_division=1)
            * 100,
            2,
        )
        f1 = round(
            f1_score(y_test_fold, y_pred_fold, average="weighted", zero_division=1)
            * 100,
            2,
        )
        results.append(
            {
                "K-Fold": fold,
                "Akurasi": accuracy,
                "Presisi": precision,
                "Recall": recall,
                "F1-Score": f1,
            }
        )
    results_df = pd.DataFrame(results)
    return results_df


def fold_no(a, b, n):
    data = split_data(a, b)
    classifier = GaussianNB()
    kf = KFold(n_splits=n, shuffle=True, random_state=42)
    results_gs = []
    for fold, (train_index, test_index) in enumerate(kf.split(data[0]), 1):
        X_train_fold, X_test_fold = data[0][train_index], data[0][test_index]
        y_train_fold, y_test_fold = data[2].iloc[train_index], data[2].iloc[test_index]
        classifier.fit(X_train_fold.toarray(), y_train_fold)
        y_pred_fold = classifier.predict(X_test_fold.toarray())
        accuracy = round(accuracy_score(y_test_fold, y_pred_fold) * 100, 2)
        precision = round(
            precision_score(
                y_test_fold, y_pred_fold, average="weighted", zero_division=1
            )
            * 100,
            2,
        )
        recall = round(
            recall_score(y_test_fold, y_pred_fold, average="weighted", zero_division=1)
            * 100,
            2,
        )
        f1 = round(
            f1_score(y_test_fold, y_pred_fold, average="weighted", zero_division=1)
            * 100,
            2,
        )
        results_gs.append(
            {
                "K-Fold": fold,
                "Akurasi": accuracy,
                "Presisi": precision,
                "Recall": recall,
                "F1-Score": f1,
            }
        )
    results_gss = pd.DataFrame(results_gs)
    return results_gss


def eval_akhir(data):
    rata_rata = data[["Akurasi", "Presisi", "Recall", "F1-Score"]].mean()
    return rata_rata


def pred_sentimen(text):
    lib_negative = [
        "kecewa",
        "marah",
        "sedih",
        "frustrasi",
        "kesal",
        "bosan",
        "cemas",
        "pusing",
        "merana",
        "sebal",
        "jahat",
        "tidak",
        "adil",
        "sedih",
        "sakit",
        "ngeri",
        "rugi",
        "kecewa",
        "rugi",
        "patah",
        "hati",
        "malas",
        "gugup",
        "kelam",
        "puruk",
        "zalim",
        "buruk",
        "rupa",
        "tidak",
        "daya",
        "sial",
        "kacau",
        "miskin",
        "hancur",
        "batas",
        "hilang",
        "beban",
        "sesat",
        "tidak",
        "harga",
        "ganggu",
        "pencil",
        "susah",
        "rusak",
        "pinggir",
    ]
    l_p = joblib.load("p_model.pkl")
    l_n = joblib.load("n_model.pkl")
    l_t = joblib.load("tf_idf.pkl")
    input_text = [text]
    preprocess_input = preprocess(input_text[-1])
    if any(word in preprocess_input for word in lib_negative):
        sentiment_label = "Negatif"
    else:
        X_input_tfidf = l_t.transform([" ".join(preprocess_input)])
        X_input_pca = l_p.transform(X_input_tfidf.toarray())
        predictions = l_n.predict(X_input_pca)

        sentiment_label = "Positif" if predictions[0] == 1 else "Negatif"
    return sentiment_label
