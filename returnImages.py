from PIL import Image, ImageDraw, ImageFont
import os
import random


def returnImages(c1, c2, c3, path):
    inputPath = "./temporal"
    im2 = os.path.join(inputPath, "output.jpg")
    im1 = os.path.join(inputPath, "outputSI1.jpg")
    im3 = os.path.join(inputPath, "outputSI2.jpg")
    imagen1 = Image.open(im1)
    imagen2 = Image.open(im2)
    imagen3 = Image.open(im3)
    dimensions = imagen1.size
    final = Image.new("RGB",(dimensions[0]*3,100+dimensions[1]),"black")
    final.paste(imagen1, (0,100))
    final.paste(imagen2, (dimensions[0],100))
    final.paste(imagen3, (dimensions[0]*2,100))
    draw = ImageDraw.Draw(final)
    font = ImageFont.truetype('./arial.ttf', 60)
    #input(dimensions[0])
    long1 = int(len(c1)/2)
    #input(long1)
    long2 = int(len(c2)/2)
    #input(long2)
    long3 = int(len(c3)/2)
    #input(long3)
    centerImage = int(dimensions[0]/2)
    startWord1 = centerImage-long1*50
    #input(startWord1)
    startWord2 = dimensions[0]*3/2 - long2*50
    #input(startWord2)
    startWord3 = dimensions[0]*5/2 - long3*50
    #input(startWord3)
    dimText1 = (startWord1, 20)
    dimText2 = (startWord2, 20)
    dimText3 = (startWord3, 20)
    draw.text(dimText1, c1, font=font, fill="white")
    draw.text(dimText2, c2, font=font, fill="white")
    draw.text(dimText3, c3, font=font, fill="white")
    final.save(os.path.join(path, "output.jpg"))

def returnImages2(c1, c2, path):
    inputPath = "./temporal"
    im2 = os.path.join(inputPath, "output.jpg")
    im1 = os.path.join(inputPath, "outputSI1.jpg")
    imagen1 = Image.open(im1)
    imagen2 = Image.open(im2)
    dimensions = imagen1.size
    final = Image.new("RGB",(dimensions[0]*2,100+dimensions[1]),"black")
    final.paste(imagen1, (0,100))
    final.paste(imagen2, (dimensions[0],100))
    draw = ImageDraw.Draw(final)
    font = ImageFont.truetype('./arial.ttf', 60)
    #input(dimensions[0])
    long1 = int(len(c1)/2)
    #input(long1)
    long2 = int(len(c2)/2)
    #input(long2)
    centerImage = int(dimensions[0]/2)
    startWord1 = centerImage-long1*20
    #input(startWord1)
    startWord2 = dimensions[0]*3/2 - long2*40
    #input(startWord2)
    dimText1 = (startWord1, 20)
    dimText2 = (startWord2, 20)
    draw.text(dimText1, c1, font=font, fill="white")
    draw.text(dimText2, c2, font=font, fill="white")
    final.save(os.path.join(path, "output.jpg"))
