import os, glob

dirs = [r'd:\AIoT_DoAn', r'D:\AIoT_DoAn\..']
for d in dirs:
    for f in sorted(glob.glob(os.path.join(d, '**', 'face_detect*.pth'), recursive=True)):
        size = os.path.getsize(f) / 1024 / 1024
        print(f'{size:.1f} MB  {f}')
