import requests
import os
import sys
import hashlib
from tqdm import tqdm
import tarfile
import json 
import argparse

useremail = "gradprojuni2025@gmail.com"
password = 'Gradproj@2025'

# Download to kaggle/working; extract to /data/Nuscenes

download_folder = "/kaggle/working/"
extract_folder = "/data/Nuscenes"
os.makedirs(extract_folder, exist_ok=True)

region = 'us'  # 'us' or 'asia'



def login(username, password):
    headers = {
        "Content-Type": "application/x-amz-json-1.1",
        "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth",
    }
    data = json.dumps({
        "AuthFlow": "USER_PASSWORD_AUTH",
        "ClientId": "7fq5jvs5ffs1c50hd3toobb3b9",
        "AuthParameters": {"USERNAME": username, "PASSWORD": password},
        "ClientMetadata": {}
    })
    response = requests.post(
        "https://cognito-idp.us-east-1.amazonaws.com/",
        headers=headers,
        data=data,
    )
    if response.status_code == 200:
        try:
            token = response.json()["AuthenticationResult"]["IdToken"]
            return token
        except KeyError:
            print("Authentication failed. 'AuthenticationResult' not found in the response.")
    else:
        print("Failed to login. Status code:", response.status_code)
    return None

def download_file(url, save_file, md5):
    if os.path.exists(save_file):
        print(save_file, "has downloaded")
        return save_file

    response = requests.get(url, stream=True)
    file_size = int(response.headers.get('Content-Length', 0))
    progress_bar = tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024, desc=save_file, ascii=True)
    md5obj = hashlib.md5()

    with open(save_file, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                md5obj.update(chunk)
                file.write(chunk)
                progress_bar.update(len(chunk))
    progress_bar.close()

    hash = md5obj.hexdigest()
    if hash != md5:
        print(f"MD5 mismatch for {save_file}: {hash} != {md5}")
    else:
        print(f"MD5 correct for {save_file}")
    return save_file

def extract_tgz_to_folder(tgz_file_path, extract_to):
    print(f"Extracting {tgz_file_path} to {extract_to}")
    os.makedirs(extract_to, exist_ok=True)
    with tarfile.open(tgz_file_path, 'r:gz') as tar:
        tar.extractall(path=extract_to)

def extract_tar_to_folder(tar_file_path, extract_to):
    print(f"Extracting {tar_file_path} to {extract_to}")
    os.makedirs(extract_to, exist_ok=True)
    with tarfile.open(tar_file_path, 'r') as tar:
        tar.extractall(path=extract_to)

def main():
    print("Logging in...")
    bearer_token = login(useremail, password)
    headers = {
        'Authorization': f'Bearer {bearer_token}',
        'Content-Type': 'application/json',
    }
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <n>")
        sys.exit(1)
    
    n = sys.argv[1]
    checksums_keyframes = ["1441a2b3ada1ba2b7a67fa3d26f58a16", "201fd9833e73c4d30451e98242fd2419", 
                           "c0f1a2b3ada1ba2b7a67fa3d26f58a16", "201fd9833e73c4d30451e98242fd2419", 
                           "c0f1a2b3ada1ba2b7a67fa3d26f58a16", "201fd9833e73c4d30451e98242fd2419", 
                           "c0f1a2b3ada1ba2b7a67fa3d26f58a16", "201fd9833e73c4d30451e98242fd2419", 
                           "c0f1a2b3ada1ba2b7a67fa3d26f58a16", "201fd9833e73c4d30451e98242fd2419"]
    checksums_camera_blobs = ["100da06a6276583e717f2156f1a6733d", "f626455bac84b1a4efe771ca62086278",
                              "c0f1a2b3ada1ba2b7a67fa3d26f58a16", "201fd9833e73c4d30451e98242fd2419",
                              "c0f1a2b3ada1ba2b7a67fa3d26f58a16", "201fd9833e73c4d30451e98242fd2419",
                              "c0f1a2b3ada1ba2b7a67fa3d26f58a16", "201fd9833e73c4d30451e98242fd2419",
                              "c0f1a2b3ada1ba2b7a67fa3d26f58a16", "201fd9833e73c4d30451e98242fd2419"]
    download_files = {
    f"v1.0-trainval0{n}_keyframes.tgz": checksums_keyframes[int(n)-1],
    f"v1.0-trainval0{n}_blobs_camera.tgz": checksums_camera_blobs[int(n)-1],
    "v1.0-trainval_meta.tgz": "537d3954ec34e5bcb89a35d4f6fb0d4a",
}
    print("Getting download urls...")
    download_data = {}
    for filename, md5 in download_files.items():
        api_url = f'https://o9k5xn5546.execute-api.us-east-1.amazonaws.com/v1/archives/v1.0/{filename}?region={region}&project=nuScenes'
        response = requests.get(api_url, headers=headers)
        if response.status_code == 200:
            print(filename, 'request success')
            download_url = response.json()['url']
            save_path = os.path.join(download_folder, filename)
            download_data[filename] = [download_url, save_path, md5]
        else:
            print(f'request failed : {response.status_code}')
            print(response.text)

    
    for output_name, (download_url, save_file, md5) in download_data.items():
        print("Downloading file...")
        save_file = download_file(download_url, save_file, md5)
        download_data[output_name] = [download_url, save_file, md5]
        print("Extracting file...")
        if output_name.endswith(".tgz"):
            extract_tgz_to_folder(save_file, extract_folder)
        elif output_name.endswith(".tar"):
            extract_tar_to_folder(save_file, extract_folder)
        else:
            print("Unknown file type", output_name)
        if os.path.exists(output_name):
            os.remove(output_name)
            print(f"{output_name} deleted successfully.")
    print("Done!")

if __name__ == "__main__":
    main()