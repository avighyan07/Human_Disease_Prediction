from . import db
from flask_login import UserMixin
import os
from werkzeug.utils import secure_filename
class MedicalRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    data = db.Column(db.String(10000), nullable=True)  # Nullable if not used
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    image_path = db.Column(db.String(200))  # Field to store the relative image file path

    def save_image(self, image_file):
        """Save the uploaded image and return the file path."""
        upload_dir = 'static/uploads/medical_records'
        
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)

        filename = secure_filename(image_file.filename)
        filepath = os.path.join(upload_dir, filename)
        image_file.save(filepath)
        print("Saving image to:", filepath)  # Check where the image is being saved
        # Store relative path without 'static/'
        self.image_path = f'uploads/medical_records/{filename}'  # No 'static/' here
       
        print("Image path stored in record:", self.image_path)  # Check what path is being stored




# uploads/medical_records/unnamed.jpg
# 'uploads/medical_records/unnamed.jpg'

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
    records = db.relationship('MedicalRecord', backref='user', lazy=True)
