import cv2
import numpy as np
import pandas as pd
from keras.preprocessing.image import img_to_array
import calendar;
import time;
from keras.models import load_model
from keras.models import model_from_json


import os
import cv2
import numpy as np
import pandas as pd
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.models import model_from_json



# load the pre-trained Keras model (here we are using a model
# pre-trained on ImageNet and provided by Keras, but you can
# substitute in your own networks just as easily)
global model
json_file = open("model_final.json","r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model_final.h5")

model = loaded_model
model._make_predict_function()

def ocr_prediction(image):
	
	
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (3, 3), 0)

   
	# apply Canny Edge Detection
	edged = cv2.Canny(blurred, 0, 50)

	
	(_,contours,_) = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)
	

	m = list()
	targetvec = list()
	for c in contours:
		p = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * p, True)
		
		if len(approx) == 4 and cv2.contourArea(approx)>1000 and cv2.contourArea(approx) <2000: #parameter which needs to be tuned for separate area size

			#print("Area:",cv2.contourArea(approx))
			targetvec.append(approx)
			
	m = list()
	
   


	point_list = []
	for c in targetvec:
		x1, y1, width1, height1 = cv2.boundingRect(c)
		point_list.append([x1,y1,width1,height1])


	#filter necessary so that the big outer contour is not detected
	point_array = [point for point in point_list]
	duplicate_array = []
	same_pt = []
	point_array = sorted(point_array,key=lambda x: (x[1]))
	
	for i in range(len(point_array)):
		for j in range(i+1,len(point_array)):
			#nearby contour points to remove
			if point_array[i][1]+ 10 > point_array[j][1]:
				point_array[j][1] = point_array[i][1]


	point_array = sorted(point_array,key=lambda x: (x[1],x[0]))

	for i in range(len(point_array)):
		for j in range(i+1,len(point_array)):
			if point_array[i][0]+ 10 > point_array[j][0] and  point_array[i][1]+ 10 > point_array[j][1] and point_array[i][2]+ 10 > point_array[j][2] and point_array[i][3]+ 10 > point_array[j][3] :
				duplicate_array.append(j)

	#deleting from reverse based on index to avoid out of index issue 
	duplicate_array = sorted(list(set(duplicate_array)),reverse=True)
	# print("Points detected:",len(point_array),"Duplicate Points to be removed:",len(list(set(duplicate_array))))
	# #print(duplicate_array)
	# for i in duplicate_array:
	#     print ("Deleted",i)

	for i in duplicate_array:
		del point_array[i]

	
	roilist = []
	  
	for i  in range(0,len(point_array)):

			x, y, width, height = point_array[i][0],point_array[i][1],point_array[i][2],point_array[i][3]
			#if y < 720:
			#cropping some padding which contains box lines
			roi = image[y+3:y+height-3, x+5:x+width-5]

			
			cv2.rectangle(image,(x,y),(x+width,y+height),(0,255,0),1)
		
			roilist.append(roi)
			
			
	characters = ['0','1','2','3','4','5','6','7','8','9']

	responselist = ""
	for roi in roilist:
		thresh = 170    
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))

		gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		
	
		im_bw = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
			cv2.THRESH_BINARY,51,12)


		
		im_bw = cv2.erode(im_bw, kernel, iterations=1)
		
		_,im_bw = cv2.threshold(im_bw, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	
		cv2.imwrite("im_bw.jpg",im_bw)
		height,width = im_bw.shape
		im_bw = cv2.resize(im_bw,dsize = (width*5,height*4),interpolation = cv2.INTER_CUBIC)

		
		ret,thresh = cv2.threshold(im_bw,127,255,cv2.THRESH_BINARY_INV)

		im2,ctrs,hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		
		m = list()
		sorted_ctrs = sorted(ctrs, key = lambda ctr: cv2.boundingRect(ctr)[0])
		
		
		pchl = list()

		dp = im_bw.copy()


		for i,ctr in enumerate(sorted_ctrs):
			x,y,w,h = cv2.boundingRect(ctr)
			
			#print("Height, Weight, W/h, X , Y ->",h,w,float(w)/h,x,y)
		
			
			if float (w/h) < 3 and x>5 and y>10:
				roi = im_bw[y-10:y+h+10, x-5:x+w+10]
			else:
				roi = im_bw 
			
			roi = cv2.resize(roi,dsize = (28,28), interpolation = cv2.INTER_AREA)
			kernel = np.ones((2,2),np.uint8)

			
			roi = np.array(roi)
			t = np.copy(roi)
			t = t /255.0
			t = 1 - t
			t = t.reshape(1,784)
			

			prob = model.predict_proba(t)
			prob_list = prob[0].argsort()[-3:][::-1]
			#top_response = {
			#"t1":[characters[prob_list[0]],prob[0][prob_list[0]]],
			#"t2":[characters[prob_list[1]],prob[0][prob_list[1]]],
			#"t3":[characters[prob_list[2]],prob[0][prob_list[2]]]}
			#print(top_response)
			print(characters[prob_list[0]],characters[prob_list[1]]," probs:: ",prob[0][prob_list[0]],prob[0][prob_list[1]])
			responselist += characters[prob_list[0]]
	return(responselist)

def rectify(h):
		h = h.reshape((4,2))
		hnew = np.zeros((4,2),dtype = np.float32)

		add = h.sum(1)
		hnew[0] = h[np.argmin(add)]
		hnew[2] = h[np.argmax(add)]

		diff = np.diff(h,axis = 1) 
		hnew[1] = h[np.argmin(diff)]
		hnew[3] = h[np.argmax(diff)]

		return hnew


def outerRectangle(image):


    height, width, channels = image.shape
    if width > height:
            image = cv2.transpose(image) 
            image = cv2.flip(image,1)

    # resize image so it can be processed
    image = cv2.resize(image, (1600, 1200))  

    # creating copy of original image
    orig = image.copy()
    # convert to grayscale and blur to smooth
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #blurred = cv2.medianBlur(gray, 5)


    edged = cv2.Canny(blurred, 0,50)
    
    orig_edged = edged.copy()

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    (_,contours, _) = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # get approximate contour
    for c in contours:
            p = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * p, True)

            if len(approx) == 4:
                    target = approx
                    break


    # mapping target points to 800x800 quadrilateral
    approx = rectify(target)
    (tl, tr, br, bl) = approx

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")

    #pts2 = np.float32([[0,0],[800,0],[800,800],[0,800]])

    M = cv2.getPerspectiveTransform(approx,dst)

    dst = cv2.warpPerspective(orig,M,(maxWidth, maxHeight))

    mask = np.ones(orig.shape, np.uint8)
    mask = cv2.bitwise_not(mask)
    x_offset=y_offset=50
    mask[y_offset:y_offset+dst.shape[0], x_offset:x_offset+dst.shape[1]] = dst
    edged = cv2.Canny(blurred, 0,50)

    orig_edged = edged.copy()


    return dst

def correctprespective(image):


    #result2 = cv2.add(orig,result)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()#
    # mapping target points to 800x800 quadrilateral
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #blurred = cv2.medianBlur(gray, 5)

    # apply Canny Edge Detection

    edged = cv2.Canny(blurred, 0,50)

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    (_,contours, _) = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # get approximate contour
    pt = []
    largestctr=""
    for c in contours:
            p = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * p, True)

            if len(approx) == 4:
                    target = approx
                    #largestctr = ctr
                    break
    orig = image.copy()

    approx = rectify(target)

    #cv2.drawContours(orig,[target],-1,(0,255,0),1)		
    x, y, w, h = cv2.boundingRect(approx)
    dst = orig[y:y+h,x:x+w]
    return dst

def innerRectangles(dst):
    names = []
    answers= []
    questions = []

    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # apply Canny Edge Detection
    edged = cv2.Canny(blurred, 0, 50)
    
    dst2 = dst.copy()

    (_,contours,_) = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    targetvec = list()
    for c in contours:
            p = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * p, True)

            if len(approx) == 4 and cv2.contourArea(approx) >4000: #parameter which needs to be tuned for separate area size

                    #print("Area:",cv2.contourArea(approx))

                    targetvec.append(approx)

    point_list = []
    for c in targetvec:
            x1, y1, width1, height1 = cv2.boundingRect                                                                                                                                                                                                                                                  (c)
            point_list.append([x1,y1,width1,height1])

    #filter necessary so that the big outer contour is not detected
    point_array = [point for point in point_list if point[0] > 15]
    duplicate_array = []
    same_pt = []
    point_array = sorted(point_array,key=lambda x: (x[1]))
    # for i in point_array:
    # 		print ("Point Array :",i)

    for i in range(len(point_array)):
            for j in range(i+1,len(point_array)):
                    #nearby contour points to remove
                    if point_array[i][1]+ 10 > point_array[j][1]:
                            point_array[j][1] = point_array[i][1]


    point_array = sorted(point_array,key=lambda x: (x[1],x[0]))

    for i in range(len(point_array)):
            for j in range(i+1,len(point_array)):
                    if point_array[i][0]+ 10 > point_array[j][0] and  point_array[i][1]+ 10 > point_array[j][1] and point_array[i][2]+ 10 > point_array[j][2] and point_array[i][3]+ 10 > point_array[j][3] :
                            duplicate_array.append(j)


    # print("final size is : ", len(point_array))

    # print("duplicate_array size is : ", len(duplicate_array))

    #deleting from reverse based on index to avoid out of index issue 
    duplicate_array = sorted(list(set(duplicate_array)),reverse=True)
    # print("Points detected:",len(point_array),"Duplicate Points to be removed:",len(list(set(duplicate_array))))
    # #print(duplicate_array)
    # for i in duplicate_array:
    # 		print ("Deleted",i)

    for i in duplicate_array:
            del point_array[i]

    for i in point_array:
            x, y, width, height = i[0],i[1],i[2],[3]



    for i  in range(0,len(point_array)):

            x, y, width, height = point_array[i][0],point_array[i][1],point_array[i][2],point_array[i][3]
            #if y < 720:
            #cropping some padding which contains box lines
            roi = dst[y-3:y+height+3, x-5:x+width+5]
            #cv2.rectangle(dst,(x,y),(x+width,y+height),(0,255,0),1)
            #print(roi.shape)
            #print("height - width {}".format(abs(height-width)))      

            area = height * width 
            if height+30  >=width:
                    continue
            #print("final area :: ", area)      


            os.path.join('.')       

            if i==0 or i==1:
                    names.append(roi)

            elif i>1:
                if (area>9000 and area<20000) or area>200000:
                                continue
                elif (area >4500 and area<9000):
                    answers.append(roi)

                elif (area >50000 and area <200000):
                     questions.append(roi)

    print(len(answers))
    print(len(questions))
    for i in range(len(names)):
            if not os.path.isdir('name'):
                    os.makedirs('name')            
            cv2.imwrite(os.path.join("name","name" + str(i+1)+".png"), names[i]) 

    for i in range(len(answers)):
            if not os.path.isdir('answers'):
                    os.makedirs('answers')             
            cv2.imwrite(os.path.join("answers","answers" + str(i+1)+".png"), answers[i]) 
    question_array_names = []
    for i in range(len(questions)):     
            if not os.path.isdir('questions'):
                    os.makedirs('questions')
            file_name = str(int(calendar.timegm(time.gmtime()))) + "_question" + str(i+1)+".png"
            cv2.imwrite(os.path.join("questions", file_name), questions[i])
            question_array_names.append(file_name);

    return len(point_array), answers, question_array_names 




def getResponseFromImage(input_image):
		success = False
		image = cv2.imread("static/" + input_image)

		image = outerRectangle(image)
		dst = correctprespective(image)

		#dst = correctprespective(image)

		#qpts_data = pd.read_csv("question_data.csv")

		regions_detected, answers, question_array_names = innerRectangles(dst)
		print("Detected regions :",regions_detected)
		
		responses = []
		q_types = ["ocr","ocr", "ocr", "omr","omr"]
		idx_char_omr = { 1 : "A", 2 : "B", 3 : "C", 4: "D"}

		if not len(answers) == len(q_types):
			 print("Not able to detect properly")
			 return False, answers, question_array_names

		return True, answers, question_array_names





def getBlob(im):

		params = cv2.SimpleBlobDetector_Params()

		# Change thresholds
		params.minThreshold = 0
		params.maxThreshold = 100

		# Filter by Area.
		params.filterByArea = True
		params.minArea = 100

		# Filter by Circularity
		params.filterByCircularity = True
		params.minCircularity = 0.3
		# Filter by Convexity
		params.filterByConvexity = True
		params.minConvexity = 0.5
				
		# Filter by Inertia
		params.filterByInertia = True
		params.minInertiaRatio = 0.1

		# Create a detector with the parameters
		ver = (cv2.__version__).split('.')
		if int(ver[0]) < 3 :
				detector = cv2.SimpleBlobDetector(params)
		else : 
				detector = cv2.SimpleBlobDetector_create(params)


		# Detect blobs.
		keypoints = detector.detect(im)

		def getKey(item):
				return item[1]

		im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		
		li = []
		for i in range(len(keypoints)):
				print(i,"x:",keypoints[i].pt)
				li.append(keypoints[i].pt)


		keypoint_sorted = sorted(li, key=getKey)
		

		return keypoint_sorted


def getCircles(image):

		output = image.copy()
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT, 1, 20,
									param1=45,
									param2=22,
									minRadius=0,
									maxRadius=55)


		point_list = []
		if circles is not None:
				# convert the (x, y) coordinates and radius of the circles to integers
				circles = np.round(circles[0, :]).astype("int")

				# loop over the (x, y) coordinates and radius of the circles
				for (x, y, r) in circles:
						# draw the circle in the output image, then draw a rectangle
						#print(x,y)
						point_list.append([x,y])
						# corresponding to the center of the circle
						cv2.circle(output, (x, y), r, (0, 255, 0), 1)
						#cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

		#sort by x coord because we will get row_count in metadata
		point_list = sorted(point_list,key=lambda x: x[0])

		return point_list

def evaluateOmrQuestion(image,row_count=2,x_response = ["A","B","C","D"],y_response= ["Vertebrate","Invertebrate"]):

#if __name__ == "__main__":

		#load image
		#image = cv2.imread("roi5.png")

		#get all circles
		# print("shape of OMR: ", image.shape)
		# point_list = getCircles(image)

		# #setting the x-y range based on circles
		# x_range = []
		# y_range = sorted([point_list[i][1] for i in range(row_count)])

		# print(y_range)
		# for i in range(0,len(point_list),row_count):
		#   row_group = point_list[i:i+row_count]

		#   x_range.append(min([row[0] for row in row_group ]))
		# print(x_range)

		x_range = [60,110,160,210]

		#Detecting blob points
		blob_points = getBlob(image)
		print ("blob_points ", blob_points)

		# #final response list
		responses = []

		for point in blob_points: 
				print(point) 
				found = False 
				for i in range(len(x_range)): 
						if int(point[0]) < x_range[i]: 

								responses.append(i + 1)
								#print(responses) 
								found = True 
						if found:break
				
		# print("Final responses")
		# print(responses)
		return responses


success , answers, question_array_names = getResponseFromImage("test.jpg")
q_types = ["ocr","ocr", "ocr", "omr","omr"]
idx_char_omr = { 1 : "A", 2 : "B", 3 : "C", 4: "D"}

responses=[]
for i in range(len(answers)):
                    q_img = "answers"+str(i+1)+".png"
                    if q_types[i] == "omr":
                        img = cv2.imread(os.path.join('./answers',q_img)) 
                        detected_omr_ans = evaluateOmrQuestion(img);
                        print("detected",detected_omr_ans)
                        #responses.append(idx_char_omr[detected_omr_ans[0] ] )

                    if q_types[i] =="ocr":
                        img = cv2.imread(os.path.join('./answers',q_img))
                        responses.append(ocr_prediction(img))

