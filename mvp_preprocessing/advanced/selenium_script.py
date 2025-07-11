from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd

options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)
years = list(range(1991, 2022))
# Load the page
url_base = "https://www.basketball-reference.com/leagues/NBA_{}_advanced.html#advanced"

for year in years:
    url = url_base.format(year)
    print(f"Fetching data for {year}...")
    driver.get(url)
    html = driver.page_source
    with open("advanced_data/{}.html".format(year), "w+")  as f:
        f.write(html)
