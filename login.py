import hashlib
import uuid
import pymongo
import certifi
import secrets
import datetime
import smtplib

client = pymongo.MongoClient("mongodb+srv://TessaDK:Equals2022@userdetails.smpsogr.mongodb.net/?retryWrites=true&w=majority",tlsCAFile=certifi.where())
db = client["Userdetails"]
col = db["Registrations"]

## Registration
def userregistration(name,email,password):
    hashed_password = hashlib.sha256(password.encode('utf-8')).hexdigest()
    user_details = {"_id":uuid.uuid4().hex,"Name":name,"Email":email,"Password":hashed_password}
    check = col.find_one({"Email":email})
    if check:
        return 'User already exists'
    else:
        col.insert_one(user_details)
        return 'Registration is successful'

## Login
def login(email,password):
    check = col.find_one({"Email":email})
    if check:
        hash_password = hashlib.sha256(password.encode('utf-8')).hexdigest()
        if check["Password"] == hash_password:
            return 'Success'
        else:
            return 'Wrong password'
    else:
        return 'No user found'

def updatepassword (email, password):
    hashed_password = hashlib.sha256(password.encode('utf-8')).hexdigest()
    check = col.find_one({"Email":email})
    if check:
        col.update_one({"Email":email},{"$set":{'Password':hashed_password}})
        return 'Password updated'
    else:
        return 'Email address not registered'


## Request new password
# Add email address sender
# Hide page to reset password
# Add link to page to reset password
# Add server
# def send_password_reset_email(email):
#     token = secrets.token_hex(16)
#     sender = 'noreply@example.com'
#     recipient = email
#     subject = 'Password Reset Request'
#     body = 'Click this link to reset your password: http://localhost:8000/reset-password?token=' + token
#     message = f'Subject: {subject}\n\n{body}'
#     server = smtplib.SMTP('smtp.example.com')
#     server.sendmail(sender, recipient, message)
#     server.quit()
#
# ## Send token
# def sendtoken(email):
#     check = col.find_one({"Email":email})
#     if check:
#         token = secrets.token_hex(16)
#         db.password_reset.insert_one({
#             'email': email,
#             'token': token,
#             'timestamp': datetime.datetime.utcnow()
#         })
#         send_password_reset_email(email)
#         return 'Email sent!'
#
# ## Reset password
# def resetpassword(password):
#     token = secrets.token_hex(16)
#     reset_request = db.password_reset.find_one({'token': token})
#     hashed_password = hashlib.sha256(password.encode('utf-8')).hexdigest()
#     if reset_request:
#         if datetime.datetime.utcnow() - reset_request['timestamp'] < datetime.timedelta(hours=24):
#             col.update_one({'email': reset_request['email']}, {'$set': {'password': hashed_password}})
#             return 'Password updated'
#         else:
#             return 'Token expired'
#     else:
#         return 'Invalid token'
