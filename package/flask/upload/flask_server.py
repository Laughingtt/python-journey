import os
from flask import Flask, render_template, send_from_directory, request, jsonify

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__, template_folder=basedir)

UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = ''
ALLOWED_EXTENSIONS = set(list(['csv', 'txt', 'png', 'jpg', 'xls', 'JPG', 'PNG', 'xlsx', 'gif', 'GIF']))  # 允许上传的文件后缀


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def upload_test():
    return render_template('upload.html')


@app.route('/api/upload', methods=['POST'], strict_slashes=False)
def api_upload():
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    print(file_dir)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    f = request.files['myfile']
    if f and allowed_file(f.filename):
        fname = f.filename
        f.save(os.path.join(file_dir, fname))
        return jsonify({"errno": 0, "errmsg": "upload success"})
    else:
        return jsonify({"errno": 1001, "errmsg": "upload failed"})


@app.route("/download/<path:filename>")
def downloader(filename):
    dirpath = os.path.join(app.root_path, '')
    return send_from_directory(dirpath, filename, as_attachment=True)


if __name__ == '__main__':
    app.run(port=8818)
