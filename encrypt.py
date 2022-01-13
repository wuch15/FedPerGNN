from Crypto import Random
from tqdm import tqdm
import base64
from Crypto.PublicKey import RSA 
from Crypto.Signature import PKCS1_v1_5 as PKCS1_signature
from Crypto.Cipher import PKCS1_v1_5 as PKCS1_cipher
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Hash import SHA256


def generate_key():
    random_generator = Random.new().read
    rsa = RSA.generate(2048, random_generator)
    public_key = rsa.publickey().exportKey()
    private_key = rsa.exportKey()
    
    with open('rsa_private_key.pem', 'wb')as f:
        f.write(private_key)
        
    with open('rsa_public_key.pem', 'wb')as f:
        f.write(public_key)
    

def get_key(key_file):
    with open(key_file) as f:
        data = f.read()
        key = RSA.importKey(data)
    return key    

def sign(msg):
    private_key = get_key('rsa_private_key.pem')
    signer = PKCS1_signature.new(private_key)
    digest = SHA256.new()
    digest.update(bytes(msg.encode("utf8")))
    return signer.sign(digest)

def verify(msg, signature):
    #use signature because the rsa encryption lib adds salt defaultly
    pub_key = get_key('rsa_public_key.pem')
    signer = PKCS1_signature.new(pub_key)
    digest = SHA256.new()
    digest.update(bytes(msg.encode("utf8")))
    return signer.verify(digest, signature)
    
def encrypt_data(msg): 
    pub_key = get_key('rsa_public_key.pem')
    cipher =encryptor = PKCS1_OAEP.new(pub_key)
    encrypt_text = base64.b64encode(cipher.encrypt(bytes(msg.encode("utf8"))))
    return encrypt_text.decode('utf-8')

def decrypt_data(encrypt_msg): 
    private_key = get_key('rsa_private_key.pem')
    cipher = PKCS1_OAEP.new(private_key)
    back_text = cipher.decrypt(base64.b64decode(encrypt_msg))
    return back_text.decode('utf-8')