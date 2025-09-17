from PyPDF2 import PdfReader
from PIL import Image
from io import BytesIO
from docling.document_converter import DocumentConverter
import spaces

EXTENSIONS_IMG_FILES = ['jpeg', 'jpg', 'png']
EXTENSIONS_FILES = ['pdf']
EXTENSIONS_ALLOWED = EXTENSIONS_IMG_FILES + EXTENSIONS_FILES

MSG_NENHUM_ARQUIVO_ENVIADO = 'Nenhum arquivo enviado.'
MSG_TEXTO_NAO_EXTRAIDO = "Não foi possível extrair o texto."

# Max dimensions for processing
MAX_IMAGE_SIZE = 2000  # pixels

def fix_type(file_upload):
    type_file = file_upload.split('/')[-1].split('.')[-1]
    if type_file in EXTENSIONS_IMG_FILES:
        return None, type_file
        # return read_file_img(file_upload), type_file
    elif type_file in EXTENSIONS_FILES:
        return read_file_pdf(file_upload), type_file

@spaces.GPU
def doc_converter(file_path):
    try:
        converter = DocumentConverter()
        result = converter.convert(file_path)
        return result.document.export_to_markdown()
    except Exception as ex:
        print(ex)
        return None

# Resize image while maintaining aspect ratio
def resize_image(image, max_size):
    width, height = image.size
    if width <= max_size and height <= max_size:
        return image
    
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    return image.resize((new_width, new_height), Image.LANCZOS)

def process_image(image_bytes):
    try:
        image = Image.open(BytesIO(image_bytes))
        # Resize large images to prevent memory issues
        # resized = resize_image(image, MAX_IMAGE_SIZE)
        return image
    except Exception as e:
        # st.error(f"Error processing image: {str(e)}")
        return None

def read_file_img(file_img):
    image_bytes = file_img.getvalue()
    img_pil = process_image(image_bytes)
    return img_pil

def read_file_pdf(file_pdf):
    # image_bytes = file_pdf.getvalue()
    reader = PdfReader(file_pdf)
    return reader

def extract_content_in_pdf(reader):
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text + "\\n"
    
    return raw_text

if __name__ == '__main__':
    pass