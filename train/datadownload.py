import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Function to download files
def download_files(url, download_dir):
    # Create the download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)

    # Fetch the directory listing
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Parse the HTML content
    soup = BeautifulSoup(response.content, "html.parser")

    # Find all the file links in the directory listing
    file_links = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and (href.endswith(".csv") or href.endswith(".xml") or href.endswith(".mseed")):
            file_links.append(href)

    # Download each file
    count = 0
    for file_name in file_links:
        file_url = urljoin(url, file_name)
        file_path = os.path.join(download_dir, os.path.basename(file_name))

        # Check if the file already exists
        if os.path.exists(file_path):
            print(f"File {file_name} already exists. Skipping download.")
            continue

        print(f"Downloading {file_url}...")
        try:
            file_response = requests.get(file_url, timeout=10)
            file_response.raise_for_status()  # Raise an exception for HTTP errors

            with open(file_path, "wb") as file:
                file.write(file_response.content)

            print(f"Downloaded {file_name} to {file_path}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {file_name}: {e}")
        count = count+1

    print("All files downloaded successfully.")
    print(count)

# Main function to get user input and call the download function
def main():
    # Get the URL from the user
    url = input("Enter the URL of the website: ")

    # Directory to save the downloaded files
    download_dir = "downloaded_files"

    # Download the files
    download_files(url, download_dir)
    

if __name__ == "__main__":
    main()
