import os
import requests
from tqdm import tqdm


# Function to download the file with a progress bar
def download_file(url, dest_folder, filename):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    file_path = os.path.join(dest_folder, filename)

    # Check if the file already exists
    if os.path.exists(file_path):
        print(f"{filename} already exists in {dest_folder}. Skipping download.")
        return
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(file_path, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

    print(f"{filename} has been downloaded successfully.")


def main():
    # URLs of the EEG datasets to be downloaded (replace these with actual URLs from DOIs)
    data_urls = {
        'MODA_EEG.zip': 'https://doi.org/10.1109/ICCV51070.2023.02104',  # Example placeholder URL
        'STEW_EEG.zip': 'https://doi.org/10.54941/ahfe1004172',  # Example placeholder URL
        'SJTU_Emotion_EEG.zip': 'https://doi.org/10.1109/TIM.2022.3216829',  # Example placeholder URL
        'Sleep-EDF_EEG.zip': 'https://doi.org/10.5665/sleep.5774'  # Example placeholder URL
    }

    # Destination folder for downloaded data
    dest_folder = 'data'

    # Download each file
    for filename, url in data_urls.items():
        download_file(url, dest_folder, filename)


if __name__ == '__main__':
    main()
