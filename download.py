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

    file_ids = ['1KL5079tA_5CVizekbNDINVFXB9LMvl4Y', '1ciuCxtOeUqBzDaqkitKKX9xdlmO0oEOI','1Fk27xukl6ts8bBntEmpASl6KvkOZzTYm','1KX_2sQ4Ryogv6SAn4NimSTI9MsBg5nUX']
    destinations = ['checkpoint', 'model-81.data-00000-of-00001','model-81.index','model-81.meta']
    for idx, file_id in enumerate(file_ids):
        print "Downloading " + destinations[idx]
        download_file_from_google_drive(file_id, os.path.join("./checkpoints/",destinations[idx]))            

    print "Done."
        #1KL5079tA_5CVizekbNDINVFXB9LMvl4Y
        #1ciuCxtOeUqBzDaqkitKKX9xdlmO0oEOI
        #1Fk27xukl6ts8bBntEmpASl6KvkOZzTYm
        #1KX_2sQ4Ryogv6SAn4NimSTI9MsBg5nUX