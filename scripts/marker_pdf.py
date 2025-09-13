import subprocess

# Replace with suitable input and output directories
subprocess.run(
    ["marker", "../data/pdf", "--output_dir", "../content/US20010014424A1"],
)

import zipfile
import os

def zip_folder(folder_path, zip_name):
    # Create a ZipFile object
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the folder
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Create a relative path for files to avoid absolute paths in the zip
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                # Add file to the zip archive
                zipf.write(file_path, arcname)

# Specify the folder path and zip file name
folder_path = '../data/pdf/US20010014424A1.pdf'  # Replace with the path to your folder
zip_name = '../content/US20010014424A1.zip'  # Replace with the desired zip file name

# Call the function
zip_folder(folder_path, zip_name)