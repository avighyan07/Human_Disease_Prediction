from flask import Blueprint, render_template, request, flash, jsonify #1
from flask_login import login_required, current_user
# from .models import Note
from . import db
from .models import MedicalRecord
# import json

views = Blueprint('views', __name__) # 2


@views.route('/') #3
@login_required
def home():
    

    return render_template("home.html",user=current_user)


# @views.route('/delete-note', methods=['POST'])
# def delete_note():  
#     note = json.loads(request.data) # this function expects a JSON from the INDEX.js file 
#     noteId = note['noteId']
#     # note = Note.query.get(noteId)
#     if note:
#         if note.user_id == current_user.id:
#             db.session.delete(note)
#             db.session.commit()

#     return jsonify({})


