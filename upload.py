import zipfile
import os

# 압축 파일 경로
zip_file_path = "/content/[라벨]사과_0.정상.zip"

# 압축 해제할 폴더 경로
extract_folder_path = "/content/[라벨]사과_0.정상"

# 압축 해제
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder_path)

# 압축 해제된 파일 목록 출력
extracted_files = os.listdir(extract_folder_path)
print("압축 해제된 파일 목록:")
print(extracted_files)
