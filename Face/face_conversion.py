import subprocess


def face(input_path):
    command = f"python Face/demo.py --config Face/config/vox-adv-256.yaml --driving_video {input_path} --source_image Face/avi.jpeg --checkpoint Requirements/Models/vox-adv-cpk.pth.tar --relative --adapt_scale"
    subprocess.run(command, shell=True)
