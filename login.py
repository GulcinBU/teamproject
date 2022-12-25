import hashlib
import uuid
import pymongo
import certifi

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

## Update password
def updatepassword (name, email,password):
    hashed_password = hashlib.sha256(password.encode('utf-8')).hexdigest()
    check = col.find_one({"Email":email})
    if check:
        col.update_one({"Email":email},{"$set":{'Password':hashed_password}})
        return 'Password updated'
    else:
        return 'Email address not registered'