import requests
import argparse

"""
code from Stack overflow user `turdus-merula`
https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
"""
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

"""
code from Stack overflow user `turdus-merula`
https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
"""
def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None
"""
code from Stack overflow user `turdus-merula`
https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
"""
def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument('--type', type=str, default='full',
                        help='Set the type of download, (full, train, infer).')

    FLAGS, unparsed = parser.parse_known_args()


    downloads = {}
    if FLAGS.type == 'full':
        downloads = {
        "1hPb_KXh--9i0qDz-1Ofjh7NGUTQiqZ5p": "storage/salicon/fixations_train2014.json",
        "12fuhhQQTBzfKr4FdqnFW4PkSWdCqiQvq": "storage/salicon/fixations_val2014.json",
        "1pqStevWP4FXRvi9wW4ZuvqDoTaCZ0htu": "storage/salicon/val_images.zip",
        "12mNb_GHahzWx6lJJcp-cbNg7Cn63To2d": "storage/salicon/train_images.zip",
        "1d_iCP85VZZ57vCLoJagdPudzZqeeBUqM": "storage/FiWi/dataset.zip"
        }
    elif FLAGS.type == 'train':
        downloads = {
        "1BMyMreJrNMxthz0kDCd-OxHfm7XlL90Z": "storage/dataset.zip",
        }
    elif FLAGS.type == 'infer':
        raise NotImplemented("inference is not yet implemented")
        downloads = {
        "1hPb_KXh--9i0qDz-1Ofjh7NGUTQiqZ5p": "storage/weights/final_weights.h5",
        }

    for file_id, destination in downloads.items():
        print("Downloading from Google drive to", destination)
        download_file_from_google_drive(file_id, destination)
        