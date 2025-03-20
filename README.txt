Description

This script analyzes HTML files to detect similar pages based on text content,
DOM structure, CSS styles, and visual similarities. It uses Selenium, BeautifulSoup,
scikit-learn, and imagehash to compare multiple pages and generate a report grouping similar files.

Installation

1. Install Python and Dependencies

Make sure you have Python installed (version 3.7+).
Then, run the following command to install all necessary dependencies:

2. Install ChromeDriver

The script uses Selenium and requires ChromeDriver.
The easiest way is to use webdriver-manager,
which automatically downloads the correct version:

Alternatively, you can manually download ChromeDriver from:
 https://chromedriver.chromium.org/downloads and set its path in the script.

Usage

Run the script using the command:

pip install -r requirements.txt
python html_clone_detector.py


The script will generate a JSON file with the results in clones_list/output/results.json, where similar files are grouped together.