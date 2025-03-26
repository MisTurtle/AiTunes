from flask import Flask, abort, request, jsonify, send_file
import numpy as np
from scipy.io.wavfile import write
import os , io
from aitunes.audio_processing.processing_interface import AudioProcessingInterface
from aitunes.experiments.autoencoder_experiment import SpectrogramBasedAutoencoderExperiment
from aitunes.user_controls.headless import HeadlessActionPipeline, ModelIdentifier 

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_DIR = "Generated_Melodies"
OUTPUT_WAV_PATH = os.path.join(OUTPUT_DIR, "generated_melody.wav")
demo_path = os.path.join("assets","Demo")
list_model = HeadlessActionPipeline.list_production_releases()


@app.route("/api/generate_debug_melody", methods=["GET"])
def generate_debug_melody():
    duration = int(request.args.get('t', 60))  
    sample_rate = 44100 

    freq = 440.0  

    t = np.linspace(0., duration, int(sample_rate * duration))
    signal = 0.5 * np.sin(2. * np.pi * freq * t)

    wav_buffer=io.BytesIO()
    write(wav_buffer, sample_rate, signal.astype(np.float32))
    

    return send_file(wav_buffer ,mimetype="audio/wav", as_attachment=True , download_name="Debug.wav")


@app.route("/api/generate_melody", methods=["GET"])
def generate_melody ():
    audio_time = request.args.get('Time', 10)
    file_name = request.args.get('Name',"out.wav")
    model_id = request.args.get('Model', 0)
    if model_id <0 or model_id >=len(list_model):
        return abort(404)
    
    model: ModelIdentifier = list_model[model_id]
    experiment: SpectrogramBasedAutoencoderExperiment = model.experiment.instantiate(model.scenario , model.model_path)
    
    latent = experiment.model.sample()
    y = experiment.model.decode(latent)
    features=experiment.mode
    audio = io.BytesIO()
    i = AudioProcessingInterface.create_for("", mode="log_mel", data=y, sr=features.sample_rate, n_fft=features.n_fft, hop_length=features.hop_length)
    i.save_to(audio)

    return send_file(audio ,mimetype="audio/wav", as_attachment=True , download_name=file_name)

@app.route("/api/models" , methods=["GET"])
def list_models():
    ret = list(map(lambda model: (model.experiment.identifier, model.scenario.identifier, model.scenario.description), list_model))     
    return jsonify(ret)
    
@app.route("/api/download_demo", methods=["GET"])
def download_demo():
    file_id = request.args.get('id',0)
    file_names=os.listdir(demo_path)
    if file_id >= len(file_names) or file_id < 0 :
        return abort(404)
    return send_file(os.path.join(demo_path,file_names[file_id]), as_attachment=True)
    

@app.route("/api/list_demo", methods=["GET"])
def list_Demo():
    files = os.listdir(demo_path)
    audio_files = [file for file in files]
    return jsonify({"audios": audio_files})


if __name__ == "__main__":
    app.run(debug=True, port=3030)
