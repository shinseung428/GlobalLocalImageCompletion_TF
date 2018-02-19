import requests
import os

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    


if __name__ == "__main__":
    import sys
    if not os.path.exists("./checkpoints/"):
        os.makedirs("./checkpoints/")

    file_ids = ['1PGsorKfp53AQ6o95c6QUJZN9yfgwEMP2', '1PrX1SEvs1aIlXRW5PFqdfO8Z1j2U0h7_','1igPcIP44U6YEYnbJZCQz0lY3A0We7F8U','1_3bG8POwj2T0WYlu5BytHNJDFSb9YBQo']
    destinations = ['checkpoint', 'model-258.data-00000-of-00001','model-258.index','model-258.meta']
    for idx, file_id in enumerate(file_ids):
        print "Downloading " + destinations[idx]
        download_file_from_google_drive(file_id, os.path.join("./checkpoints/",destinations[idx]))            

    print "Done."
