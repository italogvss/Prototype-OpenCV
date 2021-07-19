# import the necessary packages
from features import quantify_image
from features import isAnomaly
from features import writeMeta
from cv2 import cv2
import argparse
import pickle
import os

#Pegando todas as imagens da pasta exemplo
pathin  =  '/home/italo/Documentos/OpenCV/Prototipo/anomalia/'
pathout = '/home/italo/Documentos/OpenCV/Prototipo/saida/'
folder  =   os.listdir(pathin)
# Contruindo o passador de argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained anomaly detection model")
args = vars(ap.parse_args())

# Fazendo load do model
print("[INFO] loading anomaly detection model...")
model = pickle.loads(open(args["model"], "rb").read())

# Para cada arquivo no diretorio
for file in folder:

	#pega imagem	
	image = cv2.imread(pathin + file)
	
	# Quantifica Imagens
	features = quantify_image(image, bins=(3, 3, 3))

	# Usa o modelo para analisar se a imagem é uma anomalia
	#se for, procura contornos
	preds = model.predict([features])[0]
	if(preds == -1):
		image = isAnomaly(image, pathout)
		cv2.imwrite(pathout + "/{}".format(file), image)
		writeMeta(pathin + file, pathout + file)


	
