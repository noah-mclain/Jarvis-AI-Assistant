import zipfile

def is_zipfile(filepath):
    return zipfile.is_zipfile(filepath)

def process_zip(zip_path):
    texts = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if not file_info.is_dir():
                    with zip_ref.open(file_info) as file:
                        try:
                            texts.append(file.read().decode('utf-8'))
                        except UnicodeDecodeError:
                            continue
    except zipfile.BadZipFile as e:
       raise ValueError("Invalid zip file") from e
    return texts
    