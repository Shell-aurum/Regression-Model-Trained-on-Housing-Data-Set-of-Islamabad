from bs4 import BeautifulSoup
import requests
import pandas as pd
import re

prices, bathrooms, bedrooms, locations, areas = [], [], [], [], []
for page in range(2, 70):
    url = f"https://www.zameen.com/Homes/Islamabad-3-{page}.html"
    html_text = requests.get(url).text
    soup = BeautifulSoup(html_text, 'lxml')
    divs = soup.find_all('div', class_ = 'b22b6883')
    

    for div in divs:
        price_tag = div.find('span', {'aria-label': 'Price'})
        bed_tag = div.find('span', {'aria-label' : 'Beds'})
        bath_tag = div.find('span', {'aria-label' : 'Baths'})
        location_tag = div.find('div', {'aria-label' : 'Location'})#location is in div tag
        size_tag = div.find('span', {'aria-label' : 'Area'})
        if price_tag and bed_tag and bath_tag and location_tag and size_tag:
            prices.append(price_tag.text.strip())
            bathrooms.append(bath_tag.text.strip())
            bedrooms.append(bed_tag.text.strip())
            locations.append(location_tag.text.strip())
            areas.append(size_tag.text.strip())
        

data = {
    'Price (Cr)' : prices,
    'Location' : locations,
    'Area (Marla)' : areas, 
    'Bedrooms' : bedrooms,
    'Bathrooms' : bathrooms
}

df = pd.DataFrame(data)
def price_toCrore(price_str):
    if not isinstance(price_str, str):
        return None
    price_str = price_str.replace(',', ' ').strip().lower()
    if('lakh' in price_str):
        num = float(price_str.split()[0])
        return num / 100 # 1 crore = 100 lakh
    elif('crore' in price_str):
        num = float(price_str.split()[0])
        return num
    elif ('arab' in price_str):
        num = float(price_str.split()[0])
        return num * 100 # 1 arab = 100 crore
    elif ('thousand' in price_str):
        num = float(price_str.split()[0])
        return num / 100000 # 1 crore = 100000 thousands
    else:
        return None

def area_toMarla(area_str):
    if not isinstance(area_str, str):
        return None
    area_str = area_str.replace(',' , ' ').strip().lower()
    if('marla' in area_str):
        area = float(area_str.split()[0])
        return area
    elif 'kanal' in area_str:
        area = float(area_str.split()[0])
        return area*20 # 1 kanal = 20 marla
    else:
        return None

df['Price (Cr)'] = df['Price (Cr)'].apply(price_toCrore)
df['Area (Marla)'] = df['Area (Marla)'].apply(area_toMarla)
areas_map = {
    # --- Major Housing Societies ---
    'DHA Defence': 'DHA Defence',
    'DHA Phase': 'DHA Defence',
    'DHA Valley': 'DHA Valley',
    'Bahria Enclave': 'Bahria Enclave',
    'Bahria Town': 'Bahria Town',
    'Bahria Garden City': 'Bahria Garden City',
    'MPCHS': 'MPCHS Multi Gardens',
    'Faisal Town': 'Faisal Town',
    'Faisal Hills': 'Faisal Hills',
    'Faisal Margalla City': 'Faisal Margalla City',
    'Soan Garden': 'Soan Garden',
    'PWD': 'PWD Housing Scheme',
    'CBR Town': 'CBR Town',
    'Park View City': 'Park View City',
    'Park View': 'Park View City',
    'Top City': 'Top City 1',
    'Mumtaz City': 'Mumtaz City',
    'Eighteen': 'Eighteen',
    'Naval Anchorage': 'Naval Anchorage',
    'CBR': 'CBR Town',
    'Gulberg Greens': 'Gulberg Greens',
    'Gulberg Residencia': 'Gulberg Residencia',
    'Gulberg': 'Gulberg',
    'AGOCHS': 'AGOCHS',
    'Jinnah Gardens': 'Jinnah Gardens',
    'Jinnah Gardens Phase 1': 'Jinnah Gardens',
    'University Town': 'University Town',
    'PECHS': 'PECHS',
    'PAF Tarnol': 'PAF Tarnol',
    'Pakistan Town': 'Pakistan Town',
    'National Police Foundation': 'National Police Foundation O-9',
    'Margalla View Housing Society': 'Margalla View Housing Society',
    'Engineers Co-operative Housing': 'Engineers Co-operative Housing',
    'Capital Residencia': 'Capital Residencia',
    'Karakoram Greens': 'Karakoram Greens',
    'Karakoram Diplomatic Enclave': 'Karakoram Diplomatic Enclave',
    'Emaar Canyon Views': 'Emaar Canyon Views',
    'Canyon Views': 'Emaar Canyon Views',
    'Telegardens': 'F-17',
    'Fazaia Housing Scheme': 'Fazaia Housing Scheme',
    'Burma Town': 'Burma Town',
    'Rawat': 'Rawat',
    'Bhara kahu': 'Bhara kahu',
    'Bani Gala': 'Bani Gala',
    'Shah Allah Ditta': 'Shah Allah Ditta',
    'Diplomatic Enclave': 'Diplomatic Enclave',
    'Kuri Road': 'Kuri Road',

    # --- Commercial / Mixed Use ---
    'Centaurus': 'F-8',
    'Blue Area': 'Blue Area',
    'RJs Lifestyle': 'Islamabad Expressway',
    'Islamabad Expressway': 'Islamabad Expressway',

    # --- Apartment Projects (mapped to parent areas) ---
    'Askari': 'DHA Defence',
    'Goldcrest': 'DHA Defence',
    'Al-Ghurair Giga': 'DHA Defence',
    'Regent One': 'DHA Defence',
    'River Hills': 'DHA Defence',
    'Ivory': 'Bahria Town',
    'Hyde Park': 'Canyon Views',
    'Cube Apartments': 'Bahria Enclave',
    'Zarkon Heights': 'G-15',
    'Radisson BLU': 'Mumtaz City',
    'Smama Star': 'Gulberg Greens',
    'Diamond Mall': 'Gulberg Greens',
    'AJ Towers': 'Gulberg Greens',
    'Skypark One': 'Gulberg Greens',
    'Capital Square': 'B-17',
    'Capital Heights': 'G-11',
    'Executive Heights': 'F-11',
    'Silver Oaks': 'F-10',
    'Warda Hamna': 'G-11',
}

def generalize_location(loc):
    loc = str(loc)
    # CDA Sectors (F-, G-, E-, D-, I-series)
    match = re.search(r'\b([A-Z]-\d+)', loc)
    if match:
        return match.group(1)
        
    # Other Housing Societies in Map Dictionary
    for key, value in areas_map.items():
        if key.lower() in loc.lower():
            return value
    return loc.split(',')[0].strip()

df['Location'] = df['Location'].apply(generalize_location)
df.to_csv('zameen_islamabad.csv', index=False)
print(df.head())