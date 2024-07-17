from flask import *
from flask import Flask, render_template
from flask import Flask, flash, request, redirect,send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import os
import cv2
import pickle
from PIL import Image
from flask_mysqldb import MySQL
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from predict import GaussianNB_mine
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score


app = Flask(__name__, static_url_path='')



ALLOWED_EXTENSIONS = {'jpg','jpeg','png'}

# Make sure nothing malicious is uploaded
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

UPLOAD_FOLDER = 'uploads'

app.config['SECRET_KEY'] = 'super secret key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#koneksi database
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'klasifikasi'
mysql = MySQL(app)

def mean(values):
    return sum(values) / len(values)

def standar_deviasi(values):
    n = len(values)
    mean_val = mean(values)
    deviations = [(val - mean_val)**2 for val in values]
    varian = sum(deviations) / (n - 1)
    return varian ** 0.5

def skewness(values):
    n = len(values)
    mean_val = mean(values)
    sd = standar_deviasi(values)
    numerator = sum([(val - mean_val)**3 for val in values]) / n
    denominator = sd**3
    return numerator / denominator


@app.route('/', methods=['GET','POST'])
def uploader():
    if request.method == 'GET':
        return render_template('main.html')
    else:
        files = request.files.getlist('folder')
        for file in files:
            if file:
                filename = secure_filename(file.filename)
                file.save(os.path.join('uploads/dataset', filename))
                path = r"uploads/dataset"
                image_fitur = []
                for file in os.listdir(path):
                    image=cv2.imread(os.path.abspath(path +"/"+file))
                    print(file)
                    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    b,g,r = cv2.split(bgr)

                    meanB, standar_devR, skewness_R = mean(np.asarray(b).ravel()), standar_deviasi(np.asarray(r).ravel()), skewness(np.asarray(r).ravel())
                    meanG, standar_devG, skewness_G = mean(np.asarray(g).ravel()), standar_deviasi(np.asarray(g).ravel()), skewness(np.asarray(g).ravel())
                    meanR, standar_devB, skewness_B = mean(np.asarray(r).ravel()), standar_deviasi(np.asarray(b).ravel()), skewness(np.asarray(b).ravel())

            #untuk labeling
                    kelas = 0
                    if file.startswith("segar") :
                        kelas = 1
                    else :
                        kelas = 2

                    fitur = {

                        "file" : file,
                        "meanB" : meanB,
                        "meanG" : meanG,
                        "meanR" : meanR,
                        "standar_devB" : standar_devB,
                        "standar_devG" : standar_devG,
                        "standar_devR" : standar_devR,
                        "skewness_B" : skewness_B,
                        "skewness_G" : skewness_G,
                        "skewness_R" : skewness_R,
                        "kelas" : kelas
                    }

                    image_fitur.append(fitur)
                    csv_columns = ['file','meanB','meanG', 'meanR','standar_devB','standar_devG','standar_devR','skewness_B','skewness_G','skewness_R','kelas']

                    csv_file = "data_ayam.csv"
                    try :
                        with open(csv_file, 'w', newline='') as csvfile :
                            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                            writer.writeheader()
                            for data  in image_fitur :
                                writer.writerow(data)
                    except IOError:
                        print("I/O error")
        flash('Dataset berhasil diupload!')
        #upload = pd.DataFrame(fitur, index=[0])
        upload = pd.read_csv('data_ayam.csv')
        upload_html = upload.to_html(index=False,classes='table')
        return render_template('main.html', upload_html = upload_html)
        

@app.route('/csv', methods=['POST'])
def upload_csv():
    file = request.files['file2']
    if file.filename == '':
        flash('Tidak ada file yang dipilih !')
        return redirect('/')
    else:
        
        file.save(os.path.join('uploads',file.filename))
        flash('File csv berhasil disimpan')
        return redirect('/model')

@app.route('/download')
def download():
    path = 'data_ayam.csv'
    return send_file(path, as_attachment=True)


@app.route('/model', methods=['GET','POST'])
def buat_model():
    if request.method == 'GET':
        return render_template('uji.html')
    else:
        file = request.files['file2']
        file.save(os.path.join('uploads',file.filename))
        
        data = pd.read_csv('data_ayam.csv')
        data = data.drop(columns=['file'])
        X = data.drop([data.columns[-1]], axis=1)
        y = data[data.columns[-1]]
        X = np.float32(X)
        y = np.float32(y)
        testsize = float(request.form['test_size'])
        random = int(request.form['rand'])
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=testsize,random_state=random)
        model = GaussianNB_mine()
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        presisi = precision_score(y_test,y_pred)
        akurasi = accuracy_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        f1score = f1_score(y_test,y_pred)
        res = pd.DataFrame({'prediksi': y_pred, 'target': y_test})
        res.to_csv("hasil.csv",index=False)
        upload = pd.read_csv('hasil.csv')
        upload_html2 = upload.to_html(classes="table",border=3,index=False,bold_rows=3)
        cursor = mysql.connection.cursor()
        cursor.execute(''' INSERT INTO tb_latih(random_state,test_size,akurasi,presisi,recall,
        f1_score)  VALUES(%s,%s,%s,%s,%s,%s)''',(random,testsize,akurasi,presisi,recall,f1score))
        mysql.connection.commit()
        cursor.close()

        pickle.dump(model, open('mymodel.pkl','wb'))
        flash('Model berhasil dibuat!')
        return render_template('uji.html', upload_html2=upload_html2,presisi=presisi,akurasi=akurasi,recall=recall,f1score=f1score)
    #return redirect('/cek')

@app.route('/testing')
def akurasi():
    return redirect('/cek')


@app.route('/cek', methods=['GET','POST'])
def upload_file():
    #initial webpage load
    if request.method == 'GET':
        return render_template('login.html')
    else: # if request method == 'POST'
        if 'file' not in request.files:
            flash('No File part')
            return redirect(request.url)
        file = request.files['file']
        # tidak ada foto+tekan submit
        if file.filename == '':
            flash('Tidak ada foto yang dipilih !')
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash('Masukan file gambar dengan tipe  '+str(ALLOWED_EXTENSIONS))
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = request.files['file']
            file.save(os.path.join('uploads',file.filename))

            filename = Image.open(os.path.join('uploads',file.filename))
            filename = filename.resize((256,256))
            filename.save(os.path.join('uploads','resized.png'), 'PNG')
            #img = Image.open('uploads/resized.png')
            img = cv2.imread('uploads/resized.png')
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #cv2.imshow('citra awal',bgr)
            #cv2.waitKey(0)
            r,g,b = cv2.split(bgr)
            height = len(img)
            width = len(img[0])

            #HISTOGRAM
            hist_b = np.zeros((256))
            for i in range(height):
                for j in range(width):
                        #proses matrix B
                        pixel = b[i][j]
                        hist_b[pixel] += 1

            tot_histb = max(hist_b)
            cursor = mysql.connection.cursor()
            cursor.execute(''' INSERT INTO tb_histogram(max_histB)  VALUES(%s)''',(tot_histb,))
            mysql.connection.commit()
            cursor.close()

            model = pickle.load(open("mymodel.pkl", "rb"))

            for i in range(height):
                for j in range(width):
                    if tot_histb < 1000:
                        return render_template('hasil2.html')
                    if tot_histb > 10000:
                        return render_template('hasil2.html')
                else:
           
                    meanB, standar_devR, skewness_R = mean(np.asarray(r).ravel()), standar_deviasi(np.asarray(r).ravel()), skewness(np.asarray(r).ravel())
                    meanG, standar_devG, skewness_G = mean(np.asarray(g).ravel()), standar_deviasi(np.asarray(g).ravel()), skewness(np.asarray(g).ravel())
                    meanR, standar_devB, skewness_B = mean(np.asarray(b).ravel()), standar_deviasi(np.asarray(b).ravel()), skewness(np.asarray(b).ravel())
                    image = [[meanB, meanG, meanR, standar_devB,standar_devG,standar_devR,skewness_B,skewness_G,skewness_R]]
                    prediction = model.predict(image)
                    prediction = 'Segar' if prediction[0] == 1 else 'Tidak Segar'
                    
                    cursor = mysql.connection.cursor()
                    cursor.execute(''' INSERT INTO uji_ayam(mean_b,mean_g,mean_r,standar_devb,standar_devg,standar_devr,
                    skewness_b,skewness_g,skewness_r,kelas)  VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''',(meanB, meanG, meanR, standar_devB,
                    standar_devG,standar_devR,skewness_B,skewness_G,skewness_R,prediction))
                    mysql.connection.commit()
                    cursor.close()
                    
                    meanB,meanG,meanR = np.float16(meanB),np.float16(meanG),np.float16(meanR)
                    standar_devB,standar_devG,standar_devR = np.float16((standar_devB,standar_devG,standar_devR))
                    skewness_B, skewness_G, skewness_R = np.float16(skewness_B),np.float16(skewness_G),np.float16(skewness_R)
                    
                    return render_template('hasil.html',uploaded_image=filename, meanB=meanB, 
                    meanG=meanG,meanR=meanR,standar_devB=standar_devB,standar_devG=standar_devG,
                    standar_devR=standar_devR,skewness_B=skewness_B,skewness_G=skewness_G,skewness_R=skewness_R,
                    prediction_text = "{}".format(prediction))

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], 'resized.png')



if __name__ == '__main__':
    app.run(debug=True)


