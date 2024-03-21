import numpy as np
from IPython import display as disp
import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio
import os
import pickle
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn import preprocessing
import subprocess


def noise_reduction(input_path, edit_voice):
    model = pretrained.dns64().cuda()
    wav, sr = torchaudio.load(input_path)
    wav = convert_audio(wav.cuda(), sr, model.sample_rate, model.chin)
    with torch.no_grad():
        denoised = model(wav[None])[0]
    # disp.display(disp.Audio(wav.data.cpu().numpy(), rate=model.sample_rate))
    # disp.display(disp.Audio(denoised.data.cpu().numpy(), rate=model.sample_rate))

    denoised_audio = denoised.cpu()
    torchaudio.save(edit_voice, denoised_audio, model.sample_rate)


def get_MFCC(sr, audio):
    features = mfcc.mfcc(audio, sr, 0.025, 0.01, 13, appendEnergy=False)
    features = preprocessing.scale(features)
    return features


def get_gender(modelpath, edit_voice):
    # Load the pretrained models for gender
    gmm_female = pickle.load(open(os.path.join(modelpath, "female.gmm"), "rb"))
    gmm_male = pickle.load(open(os.path.join(modelpath, "male.gmm"), "rb"))

    # Load the pretrained models for type of gender
    gmm_highF = pickle.load(open(os.path.join(modelpath, "highFemale.gmm"), "rb"))
    gmm_deepF = pickle.load(open(os.path.join(modelpath, "deepFemale.gmm"), "rb"))

    gmm_highM = pickle.load(open(os.path.join(modelpath, "highMale.gmm"), "rb"))
    gmm_deepM = pickle.load(open(os.path.join(modelpath, "deepMale.gmm"), "rb"))

    sr, audio = read(edit_voice)
    features = get_MFCC(sr, audio)

    log_likelihood_male = gmm_male.score(features).sum()
    log_likelihood_female = gmm_female.score(features).sum()

    log_likelihood_hMale = gmm_highM.score(features).sum()
    log_likelihood_dMale = gmm_deepM.score(features).sum()

    log_likelihood_hFemale = gmm_highF.score(features).sum()
    log_likelihood_dFemale = gmm_deepF.score(features).sum()

    # Predict and print the gender
    if log_likelihood_female > log_likelihood_male:
        if log_likelihood_hFemale > log_likelihood_dFemale:
            g = 1  # high female
        else:
            g = 2  # deep female
    else:
        if log_likelihood_hMale > log_likelihood_dMale:
            g = 3  # high male
        else:
            g = 4  # deep male
    return g


def voice(input_path):
    base_name, extension = os.path.splitext(os.path.basename(input_path))
    edit_voice_filename = base_name + "-temp.wav"
    output_audio_filename = base_name + "-converted" + extension
    edit_voice = os.path.join("Temp", edit_voice_filename)
    output_audio = os.path.join("Temp", output_audio_filename)
    modelpath = "Requirements/Models/Models_gender"
    noise_reduction(input_path, edit_voice)
    g = get_gender(modelpath, edit_voice)
    if g == 1:
        f0 = -7
    elif g == 2:
        f0 = -3
    elif g == 4:
        f0 = 2
    else:
        f0 = 0

    command = f"python Voice/tools/infer_cli.py --f0up_key {f0} --input_path {edit_voice} --index_path Requirements/Models/added_IVF963_Flat_nprobe_1_justhire_v3_v2.index --f0method harvest --opt_path {output_audio} --model_name 'justhire_v3.pth' --index_rate 0.66 --device cuda:0 --is_half 'True' --filter_radius 3 --resample_sr 0 --rms_mix_rate 1 --protect 0.33"
    subprocess.run(command, shell=True)
    noise_reduction(output_audio, output_audio)
