import streamlit as st
import cv2 
import numpy as np
import tempfile
import json
st.title("Automated OMR Evaluator")
upload_OMR_Sheet=input("Upload OMR Sheet Image (jpg, jpeg, png) : ")
uploaded_file = st.file_uploader(Upload_OMR_Sheet, type=["jpg", "jpeg", "png"])
answer_key_file = st.file_uploader("Upload Answer Key (JSON)", type=["json"])
if uploaded_file is not None and answer_key_file is not None:
    answer_key = json.load(answer_key_file)
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    image = cv2.imread(temp_file.name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubble_contours = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        aspectRatio = w / float(h)
        if 20 < w < 50 and 20 < h < 50 and 0.8 < aspectRatio < 1.2:
            bubble_contours.append(c)
    bubble_contours = sorted(bubble_contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
    answers = {1: "A", 2: "C", 3: "C", 4: "C",5: "C",6: "A",7: "C",8: "C",9: "B",10 : "C",11:"A",12:"A" ,13:"D",14 : "A",15: "B",16 : "ABCD",17: "C",18:"D",19:"A",20:"B",21:"A",22:"D",23:"B",24:"A",25:"C",26:"B",27:"A",28:"B",29:"D",30:"C",31:"C",32:"A",33:"B",34:"C",35:"A",36:"B",37:"D",38:"B",39:"A",40:"B",
    41:"C",42:"C",43:"C",44:"B",45:"B",46:"A",47:"C",48:"B",49:"D",50:"A",51:"C",52:"B",53:"C",54:"C",55:"A",56:"B",57:"B",58:"A",59:"AB",60:"B",61:"B",62:"C",63:"A",64:"B",65:"C",66:"B",67:"B",68:"C",69:"C",70:"B",71:"B",72:"B",73:"D",74:"B",75:"A",76:"B",77:"B",78:"B",79:"B",80:"B",81:"A",82:"B",
    83:"C",84:"B",85:"C",86:"B",87:"B",88:"B",89:"A",90:"B",91:"C",92:"B",93:"C",94:"B",95:"B",96:"B",97:"C",98:"A",99:"B",100:"C"
}
    question_number = 1
    for (i, c) in enumerate(bubble_contours):
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        total = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
        choice_index = i % 4
        if choice_index == 0:
            max_val = (total, "A")
        elif choice_index == 1:
            if total > max_val[0]: max_val = (total, "B")
        elif choice_index == 2:
            if total > max_val[0]: max_val = (total, "C")
        elif choice_index == 3:
            if total > max_val[0]: max_val = (total, "D")
            answers[question_number] = max_val[1]
            question_number += 1
    score = 0
    for q, ans in answer_key.items():
        if answers.get(q) == ans:
            score += 1
    st.subheader(" Evaluation Results")
    st.write("Detected Answers:", answers)
    st.write("Final Score:", score)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="OMR Sheet", use_column_width=True)

