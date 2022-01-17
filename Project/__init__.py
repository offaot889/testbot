#from logging import root
from flask import Flask, request, abort
import requests
import json
import random
#from numpy.core.defchararray import array
#from numpy.core.fromnumeric import argmax
#from tensorflow.python.ops.gen_math_ops import arg_max
import tflearn
import numpy
import nltk
import pickle
from tensorflow.python.framework import ops
from nltk.stem.lancaster import LancasterStemmer
from linebot import LineBotApi
import PIL.Image as Image
import io
import os 
from linebot.models import (
    LocationSendMessage,TextSendMessage, QuickReply, QuickReplyButton, LocationAction, CameraAction, FlexSendMessage
)
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords

#from firebase_admin import db
#import tensorflow 
# #import re

# ถ้ามีเวลาเหลือ ทำที่แอดคำใหม่เข้าไปด้วย จากฝั่ง admins
# แล้วถ้าเจอคำใหม่ๆที่ไม่อยู่ใน dataset เลย แต่ความหมายมันลื่อในปัญหานั้น จะทำยังไงให้มันรู้


#ที่อยู่รูปที่บันทึก
dir_path = 'C:/Users/thanasaksomroob/Desktop/python/pic'

Channel_access_token = 'kDT/5enzQDKjpq4uoMqlR9gMXundmCi1AAalqGVVkiV7oZOwQ0cPWOwT7f2Xoy+td4QJec6AYPiLyMH3HYs0RXYcC82N0qSsLn/SHFoWp1NOyIGNcZ8wAW51d7BZhGLcZ/HBflHCehNf4dJlirS0MQdB04t89/1O/w1cDnyilFU='
Channel_secret = '82e05412afa9c30fe70f7cc2e4637ebc'
basic_id = '@857ebjud'
LINE_API = 'https://api.line.me/v2/bot/message/reply'

line_bot_api = LineBotApi(Channel_access_token)

app = Flask(__name__)
print("name",   app)


@app.route('/webhook', methods=['POST','GET'])
def webhook():
    if request.method == 'POST':
        #print(request.json)
        dataload = request.json
        print(dataload)
        # ทางไลน์ส่งมา
        Reply_token = dataload['events'][0]['replyToken']
        #print(Reply_token)
        #greeting(Reply_token, "test")
        message_type = dataload['events'][0]['message']['type']
        if message_type == 'text':
            message = dataload['events'][0]['message']['text']
            #print(message)
            inp = message
            """ if inp.lower() == "quit":
                return  """       
            results = model.predict([bag_of_word(inp, words)])
            results_index = numpy.argmax(results)
            print("re",results)
            print("max", results_index)
            print(labels)
            tag = labels[results_index]
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    # บันทึก patterns ใหม่ที่ได้จาก line เอาไว้ใช้ครั้งต่อไป
                    if tag == "greeting":
                        with open("intents.json", "r+", encoding="utf-8") as file:
                            new_patterns = json.load(file)
                            file_data = inp
                            # เดี๋ยวมาแก้ตรงนี้ต่อ
                            """ if file_data not in data["intents"][0]["patterns"]:
                                new_patterns["intents"][0]["patterns"].append(file_data)
                                file.seek(0)
                                json.dump(new_patterns, file, ensure_ascii=False, indent=3) """
                    elif tag == "flood problem":
                        with open("intents.json", "r+", encoding="utf-8") as file:
                            new_patterns = json.load(file)
                            file_data = inp
                            """ if file_data not in data["intents"][1]["patterns"]:
                                new_patterns["intents"][1]["patterns"].append(file_data)
                                file.seek(0)
                                json.dump(new_patterns, file, ensure_ascii=False, indent=3) """ 
                    elif tag == "road problem":
                        with open("intents.json", "r+", encoding="utf-8") as file:
                            new_patterns = json.load(file)
                            file_data = inp
                            """ if file_data not in data["intents"][2]["patterns"]:
                                new_patterns["intents"][2]["patterns"].append(file_data)
                                file.seek(0)
                                json.dump(new_patterns, file, ensure_ascii=False, indent=3) """   
                    elif tag == "electrical problem":
                        with open("intents.json", "r+", encoding="utf-8") as file:
                            new_patterns = json.load(file)
                            file_data = inp
                            """ if file_data not in data["intents"][3]["patterns"]:
                                new_patterns["intents"][3]["patterns"].append(file_data)
                                file.seek(0)
                                json.dump(new_patterns, file, ensure_ascii=False, indent=3) """     
                    elif tag == "garbage problem":
                        with open("intents.json", "r+", encoding="utf-8") as file:
                            new_patterns = json.load(file)
                            file_data = inp
                            """ if file_data not in data["intents"][4]["patterns"]:
                                new_patterns["intents"][4]["patterns"].append(file_data)
                                file.seek(0)
                                json.dump(new_patterns, file, ensure_ascii=False, indent=3) """
                    responses = tg['responses']
                    re_m = random.choices(responses)
                    Reply_messasge = ' '.join([str(elem) for elem in re_m])
                    print("RE",Reply_messasge)
                    if tg['tag'] == "greeting":
                        greeting(Reply_token, Reply_messasge)
                        #ReplyMessage(Reply_token,Reply_messasge,Channel_access_token,message_type)
                    elif tg['tag'] == "flood problem" or tg['tag'] =="road problem" or tg['tag'] =="electrical problem" or tg['tag'] =="garbage problem":
                        Camera_Action(Reply_token, Reply_messasge)
                    elif tg['tag'] == "no_picture":
                        Quick_Reply(Reply_token)

            print(tag)
            print(results_index)        
            print(Reply_messasge)    
            ReplyMessage(Reply_token,Reply_messasge,Channel_access_token,message_type)
        elif message_type == 'location':
            message_type = dataload['events'][0]['message']['type']
            Reply_messasge = 'ขอบคุณสำหรับปัญหาที่แจ้งมาทางเราจะดำเนินการอย่างเร่งด่วน'
            ReplyMessage(Reply_token,Reply_messasge,Channel_access_token,message_type)
            """ line_bot_api.reply_message(
                    Reply_token,
                    LocationSendMessage(
                        title='Location', address=(dataload['events'][0]['message']['address']),
                        latitude=(dataload['events'][0]['message']['latitude']), longitude=(dataload['events'][0]['message']['longitude'])
                    )
                )  """

            """ Reply_messasge = 'บันทึกที่อยู่สำเร็จ'
            ReplyMessage(Reply_token,Reply_messasge,Channel_access_token,message_type) """
        elif message_type == 'image':
            #message_id = dataload['events'][0]['message']['id']
            #b = []  
            #arr_mid_m = []
            for x in dataload['events']:
                arr_mid_m = []
                arr_mid_m.append(x['message']['id'])
                message_id = x['message']['id']
                message_contents = line_bot_api.get_message_content(x['message']['id'])
                with open('file_path', 'wb') as fd:
                        for chunk in message_contents.iter_content(chunk_size=720*720):
                            # ขนาดอันรูปอันก่อน 1024*1024
                            fd.write(chunk)
                with open('file_path', mode='rb') as file:
                    filepath = file.read()
                    img = Image.open(io.BytesIO(filepath))
                    #img.show()

                print(arr_mid_m)    
                filename = 'pic{}.jpg'.format(message_id)
                file_path_save = os.path.join( dir_path, filename)
                img.save(file_path_save)     
                """ time.sleep(3)  
                path_on_cloud = "image/{}.jpg".format(message_id)
                path_lacal = '{}'.format(file_path_save)
                storage.child(path_on_cloud).put(path_lacal)  """ 
                
                #time.sleep(3)
                # get url
                """ url_image = storage.child("{}".format(path_on_cloud)).get_url(None)
                print(url_image)  """
                
                # save Realtime Database
                """ date_now = datetime.now()
                timestamp = datetime.timestamp(date_now) """
                #ชื่อโฟรเดอร์ ตอน save database
                """ ref = db.reference('DataImage')
                ref.push({
                    'Timestamp': timestamp,
                    'Url_image': url_image
                })   """    
            #มีไวเพื่อล้างข้อมูลในไฟล์ file_path     
            with open('file_path', 'wb') as fd:    
                fd.close()
            # locationaction    
            Quick_Reply(Reply_token) 
        return request.json, 200
    elif request.method == 'GET':

        return 'this is method GET!! เอาไว้ส่งข้อมูลไปมากับไลน์', 200
    else: abort(400)    
     
@app.route('/')
def Hello():
    return 'Hello',200    

def sendtextunderstand(Reply_token, TextMessage):
    line_bot_api.reply_message(
            Reply_token,
            TextSendMessage(text=TextMessage))

def Quick_Reply(Reply_token):
    line_bot_api.reply_message(
            Reply_token,
            TextSendMessage(
                text='แชร์ที่อยู่บริเวณนั้นมาหน่อยครับ',
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=LocationAction(label="Send location")
                        ),
                    ])))
def Camera_Action(Reply_token, TextMessage):
    line_bot_api.reply_message(
            Reply_token,
            TextSendMessage(
                text= TextMessage,
                quick_reply=QuickReply(
                    items=[
                        QuickReplyButton(
                            action=CameraAction(label="กล้องถ่ายรูป")
                        )
                    ])))

def greeting(Reply_token, TextMessage):
    bubble_string = """
    {
       "type": "carousel",
        "contents": [
            {
            "type": "bubble",
            "hero": {
                "type": "image",
                "url": "https://cdn1.vectorstock.com/i/1000x1000/01/45/flood-rescue-vector-19930145.jpg",
                "size": "full",
                "aspectRatio": "20:13",
                "aspectMode": "cover"
            },
            "body": {
                "type": "box",
                "layout": "vertical",
                "spacing": "sm",
                "backgroundColor": "#AFE6F1FF",
                "contents": [
                {
                    "type": "text",
                    "text": "ปัญหาน้ำท่วม",
                    "weight": "bold",
                    "size": "xl",
                    "align": "center",
                    "wrap": true,
                    "contents": []
                },
                {
                    "type": "text",
                    "text": "Flood problem",
                    "weight": "bold",
                    "align": "center",
                    "contents": []
                }
                ]
            },
            "footer": {
                "type": "box",
                "layout": "vertical",
                "spacing": "sm",
                "backgroundColor": "#AFE6F1FF",
                "contents": [
                {
                    "type": "button",
                    "action": {
                    "type": "message",
                    "label": "แจ้งปัญหาน้ำท่วม",
                    "text": "แจ้งปัญหาน้ำท่วม"
                    },
                    "style": "primary"
                }
                ]
            }
            },
            {
            "type": "bubble",
            "hero": {
                "type": "image",
                "url": "https://media.istockphoto.com/vectors/road-surface-repair-vector-id908629216?k=20&m=908629216&s=612x612&w=0&h=I_9xoYdlwLsm4OQ5vgMee47GO5tJ8o4ewuBncrCHPwI=",
                "size": "full",
                "aspectRatio": "20:13",
                "aspectMode": "cover"
            },
            "body": {
                "type": "box",
                "layout": "vertical",
                "spacing": "sm",
                "backgroundColor": "#FAFABBFF",
                "contents": [
                {
                    "type": "text",
                    "text": "ปัญหาเกี่ยวกับถนน",
                    "weight": "bold",
                    "size": "xl",
                    "align": "center",
                    "wrap": true,
                    "contents": []
                },
                {
                    "type": "text",
                    "text": "Road problem",
                    "weight": "bold",
                    "align": "center",
                    "contents": []
                }
                ]
            },
            "footer": {
                "type": "box",
                "layout": "vertical",
                "spacing": "sm",
                "backgroundColor": "#FAFABBFF",
                "contents": [
                {
                    "type": "button",
                    "action": {
                    "type": "message",
                    "label": "แจ้งปัญหาถนน",
                    "text": "แจ้งปัญหาถนน"
                    },
                    "flex": 2,
                    "style": "primary"
                }
                ]
            }
            },
            {
            "type": "bubble",
            "hero": {
                "type": "image",
                "url": "https://leverageedu.com/blog/wp-content/uploads/2019/09/Electrical.png",
                "size": "full",
                "aspectRatio": "20:13",
                "aspectMode": "cover"
            },
            "body": {
                "type": "box",
                "layout": "vertical",
                "spacing": "sm",
                "backgroundColor": "#E5F6C0FF",
                "contents": [
                {
                    "type": "text",
                    "text": "ปัญหาไฟฟ้า",
                    "weight": "bold",
                    "size": "xl",
                    "align": "center",
                    "wrap": true,
                    "contents": []
                },
                {
                    "type": "text",
                    "text": "Electrical problem",
                    "weight": "bold",
                    "align": "center",
                    "contents": []
                }
                ]
            },
            "footer": {
                "type": "box",
                "layout": "vertical",
                "spacing": "sm",
                "backgroundColor": "#E5F6C0FF",
                "contents": [
                {
                    "type": "button",
                    "action": {
                    "type": "message",
                    "label": "แจ้งปัญหาไฟฟ้า",
                    "text": "แจ้งปัญหาไฟฟ้า"
                    },
                    "style": "primary"
                }
                ]
            }
            },
            {
            "type": "bubble",
            "hero": {
                "type": "image",
                "url": "https://c.neh.tw/thumb/f/720/comvecteezy297944.jpg",
                "size": "full",
                "aspectRatio": "20:13",
                "aspectMode": "cover"
            },
            "body": {
                "type": "box",
                "layout": "vertical",
                "spacing": "sm",
                "backgroundColor": "#DFC9AFFF",
                "contents": [
                {
                    "type": "text",
                    "text": "ปัญหาขยะ",
                    "weight": "bold",
                    "size": "xl",
                    "align": "center",
                    "wrap": true,
                    "contents": []
                },
                {
                    "type": "box",
                    "layout": "baseline",
                    "flex": 1,
                    "contents": [
                    {
                        "type": "text",
                        "text": "Garbage problem",
                        "weight": "bold",
                        "size": "md",
                        "align": "center",
                        "contents": []
                    }
                    ]
                }
                ]
            },
            "footer": {
                "type": "box",
                "layout": "vertical",
                "spacing": "sm",
                "backgroundColor": "#DFC9AFFF",
                "contents": [
                {
                    "type": "button",
                    "action": {
                    "type": "message",
                    "label": "แจ้งปัญหาขยะ",
                    "text": "แจ้งปัญหาขยะ"
                    },
                    "flex": 2,
                    "style": "primary"
                }
                ]
            }
            }
        ]
    }
    """
    message = FlexSendMessage(alt_text="problem", contents=json.loads(bubble_string))
    messages = [
        message,
        TextSendMessage(text= TextMessage)
    ]
    line_bot_api.reply_message(Reply_token, messages)

def ReplyMessage(Reply_token, TextMessage, Line_Acees_Token, message_type):
    #LINE_API = 'https://api.line.me/v2/bot/message/reply'

    Authorization = 'Bearer {}'.format(Line_Acees_Token) ##ที่ยาวๆ
    #print(Authorization)

    if (message_type == 'text'):
        headers = {
        'Content-Type': 'application/json; charset=UTF-8',
        'Authorization':Authorization
        }
        data = {
            "replyToken":Reply_token,
            "messages":[
                {
                "type":"text",
                "text":TextMessage
            }]
        }
        data = json.dumps(data, ensure_ascii=False) 
        requests.post(LINE_API, headers=headers, data=data.encode("utf-8"))
       
        
    
    elif (message_type == 'location'):
        headers = {
        'Content-Type': 'application/json; charset=UTF-8',
        'Authorization':Authorization
        }
        data = {
            "replyToken":Reply_token,
            "messages":[
                {
                "type":"text",
                "text":TextMessage
            }]
        }
        data = json.dumps(data, ensure_ascii=False) 
        r = requests.post(LINE_API, headers=headers, data=data.encode("utf-8"))
        print(data)
        print(r)

    return 200 


stemmer = LancasterStemmer()
with open("intents.json", encoding="utf-8") as file:
    data = json.load(file)

with open("data.pickle","rb") as f:
    words, labels, training, output = pickle.load(f)  
words = []
labels = []
docs_x = []
docs_y = []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        #wrds = nltk.word_tokenize(pattern)
        wrds = word_tokenize(pattern,engine='newmm')
        words.extend(wrds)
        #คำ
        docs_x.append(wrds)
        #tag
        docs_y.append(intent["tag"])
        
    if intent["tag"] not in labels:
        labels.append(intent["tag"])
print(docs_x)#คำ
print(docs_y)#หมวดหมู่ 
#print ("%s sentences of training data" % len(training_data))
#print(training_data)
print(labels)# มันคือ tag
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))# คำต่างๆ
labels = sorted(labels)
training = []
output = []
out_empty = [0 for _ in range(len(labels))]
for x, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w.lower()) for w in doc]
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1
    training.append(bag)
    output.append(output_row)
    #print(training)
    

    with open("data.pickle","wb") as f:
        pickle.dump((words, labels, training, output), f)
training = numpy.array(training)
output = numpy.array(output)
print(len(training[0]))
print(training)
print(output)
ops.reset_default_graph()

# Build neural network
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
#ใช้ regression
net = tflearn.regression(net, optimizer='RMSprop')
model = tflearn.DNN(net)



model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")
model.load("model.tflearn")



# Training Step มาจาก จำนวนข้อมูลทั้งหมด/batch_size
# n_epoch จำนวนรอบในการ train


# s คือ inp , words คือ คำของ model 
def bag_of_word(s, words):
    bag = [0 for _ in range(len(words))]
    #print(bag)
    #s_words = word_tokenize(s)
    s_words = word_tokenize(s)
    print("s_words")
    print(s_words)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        # enumerate เป็นคำสั่งสำหรับแจกแจงค่า index 
        #จะได้ (Index,Value)
        for i, w in enumerate(words):
            if w == se:
                bag[i] = (1)
                print(se)
    #ตรงนี้เทียบเอาถ้าต้องกับ bag แล้วใส่ 1           
    print("bag")
    print(len(numpy.array(bag)))
    print("sdsds",numpy.array(bag))
    print(numpy.argmax(bag))  
    return numpy.array(bag)
       


    
