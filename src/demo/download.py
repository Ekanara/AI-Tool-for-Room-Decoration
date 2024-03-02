import os
import subprocess
import shlex



def download_all():
    # create examples
    os.makedirs('examples/appearance', exist_ok=True)
    os.makedirs('examples/drag', exist_ok=True)
    os.makedirs('examples/move', exist_ok=True)
    os.makedirs('examples/paste', exist_ok=True)
    os.makedirs('examples/face', exist_ok=True)
    """
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/move/001.png -O examples/move/001.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/move/002.png -O examples/move/002.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/move/003.png -O examples/move/003.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/move/004.png -O examples/move/004.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/move/005.png -O examples/move/005.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/paste/001_replace.png -O examples/paste/001_replace.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/paste/002_replace.png -O examples/paste/002_replace.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/paste/002_base.png -O examples/paste/002_base.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/paste/003_replace.jpg -O examples/paste/003_replace.jpg'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/paste/003_base.jpg -O examples/paste/003_base.jpg'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/paste/004_replace.png -O examples/paste/004_replace.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/paste/004_base.png -O examples/paste/004_base.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/paste/005_replace.png -O examples/paste/005_replace.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/paste/005_base.png -O examples/paste/005_base.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/face/001_reference.png -O examples/face/001_reference.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/face/001_base.png -O examples/face/001_base.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/face/002_reference.png -O examples/face/002_reference.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/face/002_base.png -O examples/face/002_base.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/face/003_reference.png -O examples/face/003_reference.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/face/003_base.png -O examples/face/003_base.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/face/004_reference.png -O examples/face/004_reference.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/face/004_base.png -O examples/face/004_base.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/face/005_reference.png -O examples/face/005_reference.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/face/005_base.png -O examples/face/005_base.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/drag/001.png -O examples/drag/001.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/drag/003.png -O examples/drag/003.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/drag/004.png -O examples/drag/004.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/drag/005.png -O examples/drag/005.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/drag/006.png -O examples/drag/006.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/appearance/001_replace.png -O examples/appearance/001_replace.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/appearance/001_base.png -O examples/appearance/001_base.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/appearance/002_replace.png -O examples/appearance/002_replace.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/appearance/002_base.png -O examples/appearance/002_base.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/appearance/003_replace.png -O examples/appearance/003_replace.png'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/appearance/003_base.jpg -O examples/appearance/003_base.jpg'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/appearance/004_replace.jpeg -O examples/appearance/004_replace.jpeg'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/appearance/004_base.jpg -O examples/appearance/004_base.jpg'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/appearance/005_replace.jpg -O examples/appearance/005_replace.jpg'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/examples/appearance/005_base.jpeg -O examples/appearance/005_base.jpeg'))
    """
    # download checkpoints
    os.makedirs('models', exist_ok=True)
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/model/ip_sd15_64.bin -O models/ip_sd15_64.bin'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/model/shape_predictor_68_face_landmarks.dat -O models/shape_predictor_68_face_landmarks.dat'))
    subprocess.run(shlex.split('wget https://huggingface.co/Adapter/DragonDiffusion/resolve/main/model/efficient_sam_vits.pt -O models/efficient_sam_vits.pt'))

def check_and_download():
   """Checks if the 'examples' and 'models' directories exist and are not empty,
      and calls the download_all() function if either condition is not met.
   """
   main_project_dir = os.getcwd()  # Get the current working directory

   for directory in ["examples", "models"]:
       directory_path = os.path.join(main_project_dir, directory)

       if not os.path.exists(directory_path) or os.listdir(directory_path):
           download_all()
