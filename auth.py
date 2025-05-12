
from flask import Blueprint, render_template, request, flash, redirect, url_for
from .models import User
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
from . import db   ##means from __init__.py import db
from flask_login import login_user, login_required, logout_user, current_user
import os
from werkzeug.utils import secure_filename
from flask import render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from .models import MedicalRecord, db
import joblib
import cv2
import numpy as np
from keras.models import load_model
from werkzeug.utils import secure_filename

model_diabetes = joblib.load('ml_models\model4.joblib')
model_heart = joblib.load('ml_models\model2.joblib')
model_kidney = joblib.load('ml_models\model3.joblib')

auth = Blueprint('auth', __name__)

model_pneumonia = load_model(r'C:\Users\Arunava Chakraborty\Desktop\__MAIN__\dl_models\Pnumonia_detection\model_weights\vgg19_model_01.h5')



@auth.route('/login', methods=['GET', 'POST'])
def login():
    message=None
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        # print(email,password)

        user = User.query.filter_by(email=email).first()
        if user:
            if check_password_hash(user.password, password):
               message='Logged in successfully!'
               login_user(user, remember=True)
               return redirect(url_for('views.home'))
            else:
                message='Incorrect password, try again.'
        else:
            message='Email does not exist.'
    print(message)
    return render_template("login.html",message=message,user=current_user)


@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))


@auth.route('/signup', methods=['GET', 'POST'])
def sign_up():
    message = None
    if request.method == 'POST':
        email = request.form.get('email')
        first_name = request.form.get('firstName')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')
        
        # print(email,first_name)

        user = User.query.filter_by(email=email).first()
        if user:
            message='Email already exists.'
        elif len(email) < 4:
            message='Email must be greater than 3 characters.'
        elif len(first_name) < 2:
            print("name is too short")
            message='First name must be greater than 1 character.'
        elif password1 != password2:
            message='Passwords don\'t match.'
        elif len(password1) < 7:
            message='Password must be at least 7 characters.'
        else:
            # return redirect(url_for('views.home'))
            new_user = User(email=email, first_name=first_name, password=generate_password_hash(
                password1, method='pbkdf2:sha256'))
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user, remember=True)
            message='done'
            # flash('Account created!', category='success')
            return redirect(url_for('views.home'))
            

    return render_template("signup.html",message=message,user=current_user)


@auth.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            # Check file type
            if not image_file.filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                flash('Invalid file type. Please upload an image.', category='error')
                return redirect(url_for('auth.upload'))

            # Save the image using the method from the MedicalRecord model
            new_record = MedicalRecord(user_id=current_user.id)
            new_record.save_image(image_file)  # Image is saved, and path is stored here
            print(image_file)
            db.session.add(new_record)
            db.session.commit()

            flash('Image uploaded successfully!', category='success')
            # return redirect(url_for('auth.upload'))
    
    return render_template("upload.html", user=current_user)


@auth.route('/delete_image/<int:image_id>', methods=['POST'])
@login_required
def delete_image(image_id):
    # Query the image record by ID
    record = MedicalRecord.query.get_or_404(image_id)
    
    # Check if the current user is the owner of the image
    if record.user_id != current_user.id:
        flash('You do not have permission to delete this image.', category='error')
        return redirect(url_for('auth.upload'))

    # Construct the correct path to the image file
    image_path = os.path.join('static', record.image_path)  # Correctly reference the relative path
    
    # Delete the image file from the file system
    try:
        os.remove(image_path)
    except Exception as e:
        flash(f'Error deleting image file: {e}', category='error')
    
    # Delete the image record from the database
    db.session.delete(record)
    db.session.commit()

    flash('Image deleted successfully!', category='success')
    return redirect(url_for('auth.upload'))


@auth.route('/mlpred')
def ml_pred():
    return render_template('mlpred.html')

@auth.route('/dlpred')
def dlpred():
    return render_template('dl_model.html')

@auth.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    result = None  # Initialize result variable

    if request.method == 'POST':
        # Collect the form data
        pregnancies = int(request.form['Pregnancies'])
        glucose = int(request.form['Glucose'])
        blood_pressure = int(request.form['BloodPressure'])
        skin_thickness = int(request.form['SkinThickness'])
        insulin = int(request.form['Insulin'])
        bmi = float(request.form['BMI'])
        dpf = float(request.form['DiabetesPedigreeFunction'])
        age = int(request.form['Age'])
        
        # Prepare the input for the model
        input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]
        
        # Get prediction
        prediction = model_diabetes.predict(input_data)[0]
        result = 'Yes, You May Have Diabetes Accoding to your Health Condition' if prediction == 1 else 'No,You dont have Diabetes Accoding to your Health Condition'

    return render_template('diabetes_form.html', prediction=result)


@auth.route('/heart', methods=['GET', 'POST'])
def heart_disease():
    result = None  # Initialize result variable

    if request.method == 'POST':
        # Collect the form data
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        # Prepare the input for the model
        input_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]

        # Get prediction
        prediction = model_heart.predict(input_data)[0]
        result = 'Yes, You May Have Heart Disease According to your Health Condition' if prediction == 1 else 'No, You Don\'t Have Heart Disease According to your Health Condition'

    return render_template('heart_form.html', prediction=result)


@auth.route('/kidney', methods=['GET', 'POST'])
def kidney_disease():
    result = None  # Initialize result variable

    if request.method == 'POST':
        # Collect the form data
        age = int(request.form['age'])
        blood_pressure = float(request.form['blood_pressure'])
        specific_gravity = float(request.form['specific_gravity'])
        albumin = int(request.form['albumin'])
        sugar = int(request.form['sugar'])
        red_blood_cells = int(request.form['red_blood_cells'])
        pus_cell = int(request.form['pus_cell'])
        pus_cell_clumps = int(request.form['pus_cell_clumps'])
        bacteria = int(request.form['bacteria'])
        blood_glucose_random = float(request.form['blood_glucose_random'])
        blood_urea = float(request.form['blood_urea'])
        serum_creatinine = float(request.form['serum_creatinine'])
        sodium = float(request.form['sodium'])
        potassium = float(request.form['potassium'])
        haemoglobin = float(request.form['haemoglobin'])
        packed_cell_volume = float(request.form['packed_cell_volume'])
        white_blood_cell_count = float(request.form['white_blood_cell_count'])
        red_blood_cell_count = float(request.form['red_blood_cell_count'])
        hypertension = int(request.form['hypertension'])
        diabetes_mellitus = int(request.form['diabetes_mellitus'])
        coronary_artery_disease = int(request.form['coronary_artery_disease'])
        appetite = int(request.form['appetite'])
        peda_edema = int(request.form['peda_edema'])
        aanemia = int(request.form['aanemia'])

        # Prepare the input for the model
        input_data = [[age, blood_pressure, specific_gravity, albumin, sugar,
                       red_blood_cells, pus_cell, pus_cell_clumps, bacteria,
                       blood_glucose_random, blood_urea, serum_creatinine,
                       sodium, potassium, haemoglobin, packed_cell_volume,
                       white_blood_cell_count, red_blood_cell_count,
                       hypertension, diabetes_mellitus, coronary_artery_disease,
                       appetite, peda_edema, aanemia]]

        # Get prediction
        prediction = model_kidney.predict(input_data)[0]
        result = 'Yes, You May Have Kidney Disease According to your Health Condition' if prediction == 1 else 'No, You Don\'t Have Kidney Disease According to your Health Condition'

    return render_template('kidney_form.html', prediction=result)


# @auth.route('/view_uploads')
# @login_required
# def view_uploads():
#     records = MedicalRecord.query.filter_by(user_id=current_user.id).all()
#     print([record.image_path for record in records])
#     return render_template("view_uploads.html", records=records,user=current_user)
@auth.route('/pneumonia', methods=['GET', 'POST'])
def pneumonia():
    result = None  # Initialize result variable
    prediction_label=None
    if request.method == 'POST':
        # Get the uploaded file from the form
        image_file = request.files['xray_image']
        
        if image_file:
            # Save the image file temporarily
            filename = secure_filename(image_file.filename)
            image_path = os.path.join('static/uploads', filename)
            image_file.save(image_path)

            # Read the image in grayscale
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                flash('Invalid image uploaded. Please try again.', category='error')
                return redirect(url_for('auth.pneumonia'))

            # Convert grayscale to RGB (duplicate the single channel to 3 channels)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # Preprocess the image (resize to the model's input size)
            img_resized = cv2.resize(img_rgb, (128, 128))
            img_resized = img_resized / 255.0  # Normalize pixel values to the range [0, 1]
            img_resized = np.reshape(img_resized, (1, 128, 128, 3))  # Add batch dimension

            # Get prediction
            prediction = model_pneumonia.predict(img_resized)
            prediction_label = 'NORMAL' if prediction[0][0] > 0.5 else 'PNEUMONIA'
            print(prediction[0][0])
            result = f'The model predicts: {prediction_label}'

            # Optionally remove the uploaded image file after processing
            os.remove(image_path)

    return render_template('pneumonia_form.html', prediction=result,disp=prediction_label)


@auth.route('/breastcancer', methods=['GET', 'POST'])
def breastcancer():
    result = None  # Initialize result variable
    prediction_label=None
    if request.method == 'POST':
        # Get the uploaded file from the form
        image_file = request.files['xray_image']
        
        if image_file:
            # Save the image file temporarily
            filename = secure_filename(image_file.filename)
            image_path = os.path.join('static/uploads', filename)
            image_file.save(image_path)

            # Read the image in grayscale
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                flash('Invalid image uploaded. Please try again.', category='error')
                return redirect(url_for('auth.breastcancer'))

            # Convert grayscale to RGB (duplicate the single channel to 3 channels)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # Preprocess the image (resize to the model's input size)
            img_resized = cv2.resize(img_rgb, (128, 128))
            img_resized = img_resized / 255.0  # Normalize pixel values to the range [0, 1]
            img_resized = np.reshape(img_resized, (1, 128, 128, 3))  # Add batch dimension

            # Get prediction
            prediction = model_pneumonia.predict(img_resized)
            prediction_label = 'Malignant' if prediction[0][0] <0.02 else 'Benign'
            print(prediction[0][0])
            print(prediction_label)
            result = f'The model predicts: {prediction_label}'

            # Optionally remove the uploaded image file after processing
            os.remove(image_path)

    return render_template('breast_cancerform.html', prediction=result,disp=prediction_label)


@auth.route('/nearby')
def nearby():
    return render_template('nearby.html')