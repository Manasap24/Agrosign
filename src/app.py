import streamlit as st
import pandas as pd
import os

# Synonym dictionary
synonyms = {

    # Crops / Plants
    "plant": "crops",
    "plants": "crops",
    "crop": "crops",
    "harvest": "harvesting",
    "harvesting": "harvesting",
    "seedling": "seed",
    "seeds": "seed",
    "farming": "farm",
    "agriculture": "farm",

    # Water / Irrigation
    "irrigation": "water",
    "watering": "water",
    "pump": "machine_water_pump",
    "handpump": "handpump_water",

    # Farmers / Farm
    "farms": "farm",
    "farmers": "farmer",
    "cultivation": "farm",

    # Machines
    "tractoring": "tractor",
    "plowing": "plough",
    "ploughing": "plough",
    "machine": "machine_water_pump",
    "harvester": "harvester",

    # Animals
    "buffaloes": "buffalo",
    "cows": "cow",
    "goats": "goat",
    "hens": "chicken",

    # Agriculture Materials
    "fertilizers": "fertilizer",
    "pesticides": "pesticide",
    "organic": "organic_food",

    # Weather / Soil
    "climate": "weather",
    "rain": "weather",
    "mud": "soil",
    "land": "soil"
}

# Load dataset
df = pd.read_csv("../dataset/agro_terms.csv")

st.title("🌾 AgroSign AI")

text = st.text_input("Enter agriculture text:")

if text:
    words = text.lower().split()

    st.subheader("Detected Signs:")

    for word in words:   # ✅ NOW INSIDE

        # Basic normalization
        if word.endswith("ing"):
            word = word[:-3]

        if word.endswith("s"):
            word = word[:-1]

        # Apply synonym mapping
        if word in synonyms:
            word = synonyms[word]

        match = df[df["keyword"] == word]

        if not match.empty:
           video_path = match.iloc[0]["video_path"]
           video_path = os.path.join("..", video_path)
           st.write(f"👉 {word}")
           st.video(video_path)