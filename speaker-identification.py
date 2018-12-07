# -*- coding: utf-8 -*-

import socket
import sys
import json
import threading
import numpy as np
import pickle
from features import FeatureExtractor
import os

# TODO: Replace the string with your user ID
user_id = "1337"

# Load the classifier:
output_dir = 'training_output'
classifier_filename = 'classifier.pickle'

with open(os.path.join(output_dir, classifier_filename), 'rb') as f:
    classifier = pickle.load(f)

if classifier == None:
    print("Classifier is null; make sure you have trained it!")
    sys.exit()

feature_extractor = FeatureExtractor(debug=False)

def onSpeakerDetected(speaker):
    """
    Notifies the user of the current speaker
    """
    print("Speaker is {}.".format(speaker))
    sys.stdout.flush()

def predict(window):
    """
    Given a window of audio data, predict the speaker.
    Then use the onSpeakerDetected(speaker) method to notify the
    user. You must use the same feature
    extraction method that you used to train the model.
    """
    X = feature_extractor.extract_features(np.asarray(window))
    X = np.reshape(X,(1,-1))

    # TODO: Fill in speaker names. Make sure labels match your training data
    classes = ['low', 'high', 'silence'] #...

    index = classifier.predict(X)
    speaker = classes[int(index)]

    onSpeakerDetected(speaker)

    return



#################   Server Connection Code  ####################

'''
    This socket is used to receive data from the data collection server
'''
receive_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
receive_socket.connect(("none.cs.umass.edu", 8888))
# ensures that after 1 second, a keyboard interrupt will close
receive_socket.settimeout(1.0)

msg_request_id = "ID"
msg_authenticate = "ID,{}\n"
msg_acknowledge_id = "ACK"

def authenticate(sock):
    """
    Authenticates the user by performing a handshake with the data collection server.

    If it fails, it will raise an appropriate exception.
    """
    message = sock.recv(256).strip().decode('ascii')
    if (message == msg_request_id):
        print("Received authentication request from the server. Sending authentication credentials...")
        sys.stdout.flush()
    else:
        print("Authentication failed!")
        raise Exception("Expected message {} from server, received {}".format(msg_request_id, message))
    sock.send(msg_authenticate.format(user_id).encode('utf-8'))

    try:
        message = sock.recv(256).strip().decode('ascii')
    except:
        print("Authentication failed!")
        raise Exception("Wait timed out. Failed to receive authentication response from server.")

    if (message.startswith(msg_acknowledge_id)):
        ack_id = message.split(",")[1]
    else:
        print("Authentication failed!")
        raise Exception("Expected message with prefix '{}' from server, received {}".format(msg_acknowledge_id, message))

    if (ack_id == user_id):
        print("Authentication successful.")
        sys.stdout.flush()
    else:
        print("Authentication failed!")
        raise Exception("Authentication failed : Expected user ID '{}' from server, received '{}'".format(user_id, ack_id))


try:
    print("Authenticating user for receiving data...")
    sys.stdout.flush()
    authenticate(receive_socket)

    print("Successfully connected to the server! Waiting for incoming data...")
    sys.stdout.flush()

    previous_json = ''
    speech_audio_data = []

    while True:
        try:
            message = receive_socket.recv(1024).strip().decode('ascii')
            json_strings = message.split("\n")
            json_strings[0] = previous_json + json_strings[0]
            for json_string in json_strings:
                try:
                    data = json.loads(json_string)
                except:
                    previous_json = json_string
                    continue
                previous_json = '' # reset if all were successful
                sensor_type = data['sensor_type']
                if (sensor_type == u"SENSOR_AUDIO"):
                    t=data['data']['t'] # timestamp isn't used
                    audio_buffer=data['data']['values']
                    print("Received audio data of length {}".format(len(audio_buffer)))
                    t = threading.Thread(target=predict, args=(np.asarray(audio_buffer),))
                    t.start()

            sys.stdout.flush()
        except KeyboardInterrupt:
            # occurs when the user presses Ctrl-C
            print("User Interrupt. Quitting...")
            break
        except Exception as e:
            # ignore exceptions, such as parsing the json
            # if a connection timeout occurs, also ignore and try again. Use Ctrl-C to stop
            # but make sure the error is displayed so we know what's going on
            if (str(e) != "timed out"):  # ignore timeout exceptions completely
                print(e)
            pass
except KeyboardInterrupt:
    # occurs when the user presses Ctrl-C
    print("User Interrupt. Quitting...")
finally:
    print('Closing socket for receiving data')
    receive_socket.shutdown(socket.SHUT_RDWR)
    receive_socket.close()
