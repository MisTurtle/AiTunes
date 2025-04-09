import os, io
import numpy as np
from collections import OrderedDict
from flask import Flask, abort, request, jsonify, send_file
from flask_cors import CORS, cross_origin
from scipy.io.wavfile import write

from aitunes import utils
from aitunes.experiments.autoencoder_experiment import SpectrogramBasedAutoencoderExperiment
from aitunes.user_controls.headless import HeadlessActionPipeline, ModelIdentifier 


app = Flask(__name__)
cors = CORS(app, origins="http://localhost:3000")
app.config['CORS_HEADERS'] = 'Content-Type'

utils.quiet = True
utils.summaries = False

demo_path = os.path.join("assets", "Demo")
prod_models = OrderedDict({ model_i.scenario.prod_name: model_i for model_i in HeadlessActionPipeline.list_production_releases() })
default_model = next(iter(prod_models))


@app.route("/api/generate_debug_melody", methods=["GET"])
def generate_debug_melody():
    duration = int(request.args.get('t', 60))  
    sample_rate = 44100 

    freq = 440.0  

    t = np.linspace(0., duration, int(sample_rate * duration))
    signal = 0.5 * np.sin(2. * np.pi * freq * t)

    wav_buffer=io.BytesIO()
    write(wav_buffer, sample_rate, signal.astype(np.float32))
    
    return send_file(wav_buffer, mimetype="audio/wav", as_attachment=True , download_name="Debug.wav")


@app.route("/api/generate_melody", methods=["GET"])
def generate_melody ():
    audio_time = request.args.get('t', 10)
    file_name = request.args.get('name', "out.wav")
    model_id = request.args.get('model', None)

    if model_id is not None and not model_id in prod_models:
        return abort(404)

    model: ModelIdentifier = prod_models[model_id if model_id is not None else default_model]
    experiment: SpectrogramBasedAutoencoderExperiment = model.experiment.instantiate(model.scenario, model.model_path)
    
    audio_file = io.BytesIO()
    audio_file.name = file_name

    latent, i = experiment.sample_audio(1, audio_time)
    i.save_to(audio_file)
    audio_file.seek(0)

    return send_file(audio_file, mimetype="audio/wav", as_attachment=True , download_name=file_name)

@app.route("/api/models" , methods=["GET"])
def list_models():
    ret = list(map(lambda model: {
        "name": model.scenario.prod_name,
        "desc": model.scenario.prod_desc
    }, prod_models.values()))
    return jsonify(ret)
    
@app.route("/api/download_demo", methods=["GET"])
def download_demo():
    file_id = request.args.get('id',0)
    file_names=os.listdir(demo_path)
    if file_id >= len(file_names) or file_id < 0 :
        return abort(404)
    return send_file(os.path.join(demo_path,file_names[file_id]), as_attachment=True)
    

@app.route("/api/list_demo", methods=["GET"])
def list_demo():
    files = os.listdir(demo_path)
    audio_files = [file for file in files]
    return jsonify({"audios": audio_files})


if __name__ == "__main__":
    app.run(debug=True, port=3030)
