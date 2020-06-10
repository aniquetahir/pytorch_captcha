from PIL import Image, ImageFilter, ImageEnhance
import pytesseract


pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
if __name__ == "__main__":
    im = Image.open('Capture2.PNG')
    im = ImageEnhance.Contrast(im).enhance(7)
    # im = im.filter(ImageFilter.GaussianBlur(3))
    im.convert('1').save('converted.png')
    print(pytesseract.image_to_string(im.convert('1'), config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'))
    print('hello world')
