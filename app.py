import streamlit as st
import os
import json
from openai import OpenAI
from langfuse import Langfuse
from langfuse import observe
import pickle
import pandas as pd

# --------------------------------------------------------------------------------
# Konfiguracja Aplikacji
# --------------------------------------------------------------------------------

st.set_page_config(
    page_title="Kalkulator Czasu Półmaratonu",
    page_icon="🏃‍♀️"
)

st.title("🏃‍♀️ Szacowany czas ukończenia półmaratonu")
st.markdown("""
Wpisz poniżej kilka słów o sobie: podaj swój wiek, płeć oraz typowe tempo biegu na 5 km. 
Nasza AI przeanalizuje tekst, a model postara się oszacować Twój czas w półmaratonie!
""")

# --------------------------------------------------------------------------------
# Panel boczny (Sidebar) do wprowadzania kluczy API
# --------------------------------------------------------------------------------

with st.sidebar:
    st.header("🔐 Konfiguracja API")
    st.markdown("Wprowadź swoje klucze API, aby uruchomić aplikację.")

    # Pole na klucz OpenAI
    openai_api_key = st.text_input(
        "Klucz API OpenAI",
        type="password",
        help="Możesz go znaleźć na https://platform.openai.com/api-keys"
    )

    st.divider()

    st.subheader("Langfuse (Opcjonalnie)")
    # Pola na klucze Langfuse
    langfuse_public_key = st.text_input("Langfuse Public Key", type="password")
    langfuse_secret_key = st.text_input("Langfuse Secret Key", type="password")
    langfuse_host = st.text_input("Langfuse Host URL", value="https://cloud.langfuse.com")

# Inicjalizacja klienta Langfuse, jeśli klucze są dostępne
langfuse = None
if langfuse_public_key and langfuse_secret_key:
    try:
        langfuse = Langfuse(
            public_key=langfuse_public_key,
            secret_key=langfuse_secret_key,
            host=langfuse_host
        )
        st.sidebar.success("Połączono z Langfuse!")
    except Exception as e:
        st.sidebar.error(f"Błąd połączenia z Langfuse: {e}")

# Sprawdzenie, czy klucz OpenAI został wprowadzony
if not openai_api_key:
    st.warning("Proszę wprowadzić klucz API OpenAI w panelu bocznym, aby kontynuować.")
    st.stop()

# Inicjalizacja klienta OpenAI
client = OpenAI(api_key=openai_api_key)

# --------------------------------------------------------------------------------
# Funkcje pomocnicze
# --------------------------------------------------------------------------------

@observe(name="data_extraction") # Dekorator Langfuse do śledzenia wywołania
def extract_runner_data(user_description: str) -> dict:
    """
    Używa modelu GPT do wyekstrahowania danych biegacza z tekstu.

    Args:
        user_description: Tekst wprowadzony przez użytkownika.

    Returns:
        Słownik (dict) z wyekstrahowanymi danymi lub pusty słownik w razie błędu.
    """
    system_prompt = """
    Twoim zadaniem jest precyzyjne wyekstrahowanie trzech informacji z tekstu podanego przez użytkownika:
    1.  `wiek` (jako liczba całkowita)
    2.  `płeć` (jako string: "Kobieta" lub "Mężczyzna")
    3.  `tempo_5km` (jako liczba zmiennoprzecinkowa, w minutach na kilometr, np. 6.5 dla 6:30 min/km)

    Zwróć odpowiedź WYŁĄCZNIE w formacie JSON, który zawiera te trzy klucze.
    Jeśli którejś informacji brakuje, przypisz jej wartość null.
    Przykład: {"wiek": 35, "płeć": "Mężczyzna", "tempo_5km": 5.75}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_description}
            ]
        )
        # Parsowanie odpowiedzi JSON
        extracted_data = json.loads(response.choices[0].message.content)
        return extracted_data
    except Exception as e:
        st.error(f"Wystąpił błąd podczas komunikacji z API OpenAI: {e}")
        return {}

# --- MIEJSCE NA MÓJ MODEL ---
import boto3
import pickle
import pandas as pd
import os

# Funkcja do pobrania i wczytania modelu z chmury
@st.cache_resource
def load_model_from_space():
    try:
        # Pobieramy klucze ze zmiennych środowiskowych serwera ---
        DO_SPACES_ACCESS_KEY = os.environ.get("DO_SPACES_ACCESS_KEY")
        DO_SPACES_SECRET_KEY = os.environ.get("DO_SPACES_SECRET_KEY")
        DO_SPACES_ENDPOINT_URL = os.environ.get("DO_SPACES_ENDPOINT_URL")
        DO_SPACES_BUCKET_NAME = os.environ.get("DO_SPACES_BUCKET_NAME")
        MODEL_FILE_NAME = "marathon_model.pkl"

        # Tworzymy klienta s3
        session = boto3.session.Session()
        s3_client = session.client('s3',
                                   endpoint_url=DO_SPACES_ENDPOINT_URL,
                                   aws_access_key_id=DO_SPACES_ACCESS_KEY,
                                   aws_secret_access_key=DO_SPACES_SECRET_KEY)
        
        # Pobieramy plik modelu z chmury do pamięci podręcznej aplikacji
        s3_client.download_file(DO_SPACES_BUCKET_NAME, MODEL_FILE_NAME, f"/tmp/{MODEL_FILE_NAME}")
        
        # Wczytujemy pobrany plik z folderu tymczasowego
        with open(f"/tmp/{MODEL_FILE_NAME}", 'rb') as file:
            model = pickle.load(file)
        return model

    except Exception as e:
        st.error(f"Błąd podczas ładowania modelu z chmury: {e}")
        return None

# Wczytujemy model
model = load_model_from_space()

# --- ZMIANA W KLIENCIE OPENAI ---
# Aplikacja w chmurze automatycznie użyje zmiennej OPENAI_API_KEY
client = OpenAI()

def predict_time(age: int, gender: str, pace_5k: float) -> str:
    """
    Używa wczytanego modelu do przewidywania czasu półmaratonu.
    """
    if model is None:
        return "Błąd: Model nie jest załadowany."

    gender_mapped = 'K' if gender.lower().startswith('k') else 'M'

    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender_mapped],
        'Pace_5k': [pace_5k]
    })

    predicted_seconds = model.predict(input_data)[0]

    hours = int(predicted_seconds // 3600)
    minutes = int((predicted_seconds % 3600) // 60)
    seconds = int(predicted_seconds % 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
# --- KONIEC MIEJSCA NA MODEL ---


# --------------------------------------------------------------------------------
# Główny interfejs aplikacji
# --------------------------------------------------------------------------------

user_input = st.text_area(
    "Opowiedz nam o sobie:",
    height=150,
    placeholder="Np. 'Cześć, jestem Ania, mam 28 lat. Biegam od roku, a moje tempo na 5 km to około 6 minut na kilometr.'"
)

if st.button("Szacuj czas półmaratonu", type="primary"):
    if user_input:
        with st.spinner("Analizuję Twoje dane i liczę... 🤖"):
            # Krok 1: Ekstrakcja danych za pomocą LLM
            extracted_data = extract_runner_data(user_input)

            if not extracted_data:
                st.error("Nie udało się przetworzyć danych. Spróbuj ponownie.")
            else:
                # Krok 2: Walidacja wyekstrahowanych danych
                missing_info = []
                if extracted_data.get("wiek") is None:
                    missing_info.append("wiek")
                if extracted_data.get("płeć") is None:
                    missing_info.append("płeć")
                if extracted_data.get("tempo_5km") is None:
                    missing_info.append("tempo na 5 km")

                if missing_info:
                    st.warning(f"**Niestety, nie znalazłem wszystkich potrzebnych informacji.**\n\nBrakuje mi: **{', '.join(missing_info)}**.\n\nSpróbuj opisać siebie jeszcze raz, bardziej szczegółowo.")
                else:
                    # Krok 3: Wywołanie modelu predykcyjnego
                    age = extracted_data["wiek"]
                    gender = extracted_data["płeć"]
                    pace_5k = extracted_data["tempo_5km"]

                    st.info(f"**Wyekstrahowane dane:**\n- Wiek: **{age}**\n- Płeć: **{gender}**\n- Tempo na 5km: **{pace_5k} min/km**")

                    predicted_time = predict_time(age, gender, pace_5k)
                    
                    # Krok 4: Wyświetlenie wyniku
                    st.success(f"### Twój szacowany czas na półmaraton to: **{predicted_time}**")
                    st.balloons()
                    
                    # Opcjonalne śledzenie w Langfuse
                    if langfuse:
                        langfuse.flush() # Wysłanie zebranych danych do Langfuse
                        st.sidebar.info("Dane z ekstrakcji LLM zostały zapisane w Langfuse.")
    else:
        st.warning("Proszę wpisać kilka słów o sobie w polu tekstowym.")