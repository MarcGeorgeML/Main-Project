import librosa
import opensmile
def extract_audio_features(audio_path):
    """OpenSMILE ComParE matching many IEMOCAP baselines"""
    wav, sr = librosa.load(audio_path, sr=16000)
    wavfile.write('temp.wav', sr, wav.astype(np.float32))
    
    feats = smile.process_file('temp.wav')
    features = feats.values.flatten()[:1582]  # Truncate/PCA to match
    
    os.remove('temp.wav')
    return features