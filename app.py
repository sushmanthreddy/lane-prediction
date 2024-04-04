import pickle
from utils import * 
from moviepy.editor import VideoFileClip
from flask import Flask, redirect, render_template, url_for, request, jsonify
import os 


app = Flask(__name__)

UPLOAD_DIR = './uploads'
OUTPUT_DIR = os.path.join('static', 'assets')

def load_models():
    torch_model = pickle.load( open( "./model.p", "rb" ) )
    mtx = torch_model["mtx"]
    dist = torch_model["dist"]
    return torch_model, mtx, dist


def process_video(input_video_path, output_video_path):
    line_l = Line()
    line_r = Line()
    torch_model, mtx, dist = load_models()
    clip1 = VideoFileClip(input_video_path)
    video_clip = clip1.fl_image(lambda img: lane_finding(img, line_l, line_r, mtx, dist))
    video_clip.write_videofile(output_video_path, audio=False)



@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        video_file = request.files['video']
        video_file.save(os.path.join(UPLOAD_DIR, 'input.mp4'))

        print("POST METHOD")

        if os.path.exists(os.path.join(OUTPUT_DIR, 'output.mp4')):
            os.remove(os.path.join(OUTPUT_DIR, 'output.mp4'))
        
        # Process the video here or call your process_video function
        input_video_path = os.path.join(UPLOAD_DIR, 'input.mp4')
        output_video_path = os.path.join(OUTPUT_DIR, 'output.mp4')

        process_video(input_video_path, output_video_path)
        
        return render_template('result.html', output=output_video_path)

    return render_template('index.html')




if __name__=='__main__':
    app.run(debug=True)

