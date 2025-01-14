import requests
from bs4 import BeautifulSoup

# Mongolian Cyrillic alphabet
mongolian_alphabet = "абвгдеёжзийклмноөпрстуүфхцчшъыьэюя"

# Generate all two-letter combinations
combinations = [a + b for a in mongolian_alphabet for b in mongolian_alphabet]

# List to store extracted values
extracted_values = []

# Base URL
base_url = "https://www.magadlal.com/name"

# Iterate through each combination
for combo in combinations:
    # Construct the URL with the current combination
    url = f"{base_url}?text={combo}&match=begins-with"
    
    # Send a GET request
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Find the <tbody> tag
        tbody = soup.find("tbody")
        if tbody:
            # Find all <tr> tags within <tbody>
            tr_tags = tbody.find_all("tr")
            for tr in tr_tags:
                # Find all <td> tags within the current <tr>
                td_tags = tr.find_all("td")
                if len(td_tags) > 1:  # Ensure there are at least two <td> tags
                    second_td = td_tags[1].text.strip()  # Extract text from the second <td>
                    extracted_values.append(second_td)
                    print(f"Combination: {combo}, Extracted Value: {second_td}")
                else:
                    print(f"Combination: {combo}, No second <td> found in this <tr>")
        else:
            print(f"Combination: {combo}, No <tbody> found")
    else:
        print(f"Combination: {combo}, Failed to fetch URL (Status Code: {response.status_code})")

# Save the extracted values to a text file
output_file = "extracted_values.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for value in extracted_values:
        f.write(value + "\n")

print(f"Extracted values saved to {output_file}")