import os
import json
import glob
import imagehash
import numpy as np
import networkx as nx
import cv2
import time
from PIL import Image
from bs4 import BeautifulSoup
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

print("Starting program...")

# 🔹 Configurare Selenium WebDriver cu opțiuni optimizate
chrome_options = Options()
chrome_options.add_argument("--headless")  # Rulează în mod headless (fără interfață grafică)
chrome_options.add_argument("--disable-gpu")  # Dezactivează GPU
chrome_options.add_argument("--window-size=1280x1024")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--allow-file-access-from-files")  # Permite accesul la fișiere locale fără probleme SSL
chrome_options.add_argument("--ignore-certificate-errors")  # Ignoră erorile SSL
chrome_options.add_argument("--disable-web-security")  # Dezactivează restricțiile de securitate pentru conținut mixt
chrome_options.add_argument("--disable-webgl")  # Dezactivează WebGL pentru a evita fallback-ul lent
chrome_options.add_argument("--disable-software-rasterizer")  # Dezactivează rasterizarea software

print("Setting up Chrome WebDriver...")
try:
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_page_load_timeout(10)  # Timeout redus la 10 secunde
    print("WebDriver initialized successfully")
except Exception as e:
    print(f"Error initializing WebDriver: {e}")
    driver = None  # Dacă WebDriver-ul nu funcționează, continuăm fără el


# 🔹 Funcții pentru procesarea fișierelor HTML

def extract_text_from_html(file_path):
    """Extrage textul vizibil dintr-un fișier HTML."""
    print(f"Extracting text from {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")
            return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def extract_dom_structure(file_path):
    """Extrage structura DOM (distribuția tagurilor HTML)."""
    print(f"Extracting DOM structure from {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")
            tags = [tag.name for tag in soup.find_all()]
            return Counter(tags)  # Returnează un dicționar cu frecvența fiecărui tag
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return Counter()


def extract_css_styles(file_path):
    """Extrage clasele CSS utilizate în HTML."""
    print(f"Extracting CSS styles from {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")
            styles = [style.get("class", []) for style in soup.find_all(True)]
            return Counter(
                [cls for sublist in styles for cls in sublist])  # Returnează un dicționar cu frecvența claselor CSS
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return Counter()


def capture_screenshot(file_path):
    """Face un screenshot al paginii HTML folosind Selenium."""
    print(f"Capturing screenshot for {file_path}")

    if driver is None:
        print("WebDriver not available, skipping screenshot")
        return None

    try:
        full_path = f"file://{os.path.abspath(file_path)}"
        print(f"Loading URL: {full_path}")

        driver.get(full_path)

        # Așteptăm maxim 3 secunde pentru ca pagina să se încarce
        time.sleep(3)

        screenshot_path = file_path.replace(".html", ".png")
        print(f"Saving screenshot to {screenshot_path}")
        driver.save_screenshot(screenshot_path)
        return screenshot_path
    except Exception as e:
        print(f"Error capturing screenshot for {file_path}: {e}")
        return None  # Ignorăm erorile și trecem la următorul fișier


def compute_visual_hash(image_path):
    """Calculează hash-ul perceptual al unei imagini pentru compararea vizuală."""
    if image_path is None:
        print("No image path provided for visual hash")
        return None

    print(f"Computing visual hash for {image_path}")
    try:
        img = Image.open(image_path).convert("L")  # Convertim imaginea în grayscale
        return imagehash.phash(img)  # Returnează hash-ul perceptual
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def compute_similarity(html_files, threshold=0.8):
    """Compară fișierele HTML și grupează paginile similare."""
    print(f"Computing similarity for {len(html_files)} HTML files")

    text_data = [extract_text_from_html(f) for f in html_files]
    dom_data = [extract_dom_structure(f) for f in html_files]
    css_data = [extract_css_styles(f) for f in html_files]

    screenshots = [capture_screenshot(f) for f in html_files]
    visual_hashes = [compute_visual_hash(s) if s else None for s in screenshots]

    print("Creating TF-IDF matrix...")
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(text_data) if any(text_data) else np.zeros(
        (len(html_files), len(html_files)))
    text_similarity = cosine_similarity(tfidf_matrix)

    clusters = []
    seen = set()

    print("Forming clusters...")
    for i, file1 in enumerate(html_files):
        if file1 in seen:
            continue
        cluster = [file1]
        seen.add(file1)

        for j, file2 in enumerate(html_files):
            if i == j or file2 in seen:
                continue

            dom_sim = 1 - (sum(
                abs(dom_data[i][tag] - dom_data[j][tag]) for tag in set(dom_data[i]) | set(dom_data[j])) / max(
                len(dom_data[i]), 1))
            css_sim = 1 - (sum(
                abs(css_data[i][cls] - css_data[j][cls]) for cls in set(css_data[i]) | set(css_data[j])) / max(
                len(css_data[i]), 1))
            visual_sim = (1 - (visual_hashes[i] - visual_hashes[j]) / 64) if visual_hashes[i] and visual_hashes[
                j] else 0

            overall_sim = (0.5 * text_similarity[i, j]) + (0.2 * dom_sim) + (0.2 * css_sim) + (0.1 * visual_sim)

            if overall_sim >= threshold:
                cluster.append(file2)
                seen.add(file2)

        clusters.append(cluster)

    print(f"Total clusters found: {len(clusters)}")
    return clusters


def save_results(clusters):
    """Salvează rezultatele într-un fișier JSON."""
    print("Saving results...")
    OUTPUT_DIR = os.path.join("clones_list", "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {"clusters": [{"group": i + 1, "files": cluster} for i, cluster in enumerate(clusters)]}
    output_path = os.path.join(OUTPUT_DIR, "results.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    BASE_DIR = "clones_list"
    html_files = glob.glob(os.path.join(BASE_DIR, "**", "*.html"), recursive=True)

    if html_files:
        clusters = compute_similarity(html_files)
        save_results(clusters)

    if driver:
        driver.quit()

    print("Program completed!")
