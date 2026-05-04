import zipfile
import os

def create_zip():
    zip_name = os.path.join("models", "exports", "maixhub_upload_v9.zip")
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add the ONNX model
        zipf.write(os.path.join("models", "exports", "face_detect_v9.onnx"), arcname="face_detect_v9.onnx")
        
        # Add the images directory explicitly
        # MaixHub requires the 'images' directory entry to exist
        zipf.writestr("images/", "")
        
        # Add all images inside the images directory
        images_dir = os.path.join("data", "images")
        for root, dirs, files in os.walk(images_dir):
            for file in files:
                if file.endswith('.jpg'):
                    file_path = os.path.join(root, file)
                    # arcname defines the path inside the zip
                    arcname = os.path.join("images", file).replace("\\", "/")
                    zipf.write(file_path, arcname=arcname)
                    
    print(f"Created {zip_name} successfully!")

if __name__ == "__main__":
    create_zip()
