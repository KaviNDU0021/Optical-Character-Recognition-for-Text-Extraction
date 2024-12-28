import cv2
import easyocr
import numpy as np
from tkinter import Tk, filedialog, Label, Text, Scrollbar, Toplevel, END, messagebox
from PIL import Image, ImageTk

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Increase image resolution (resize)
    height, width = gray_image.shape
    new_size = (width * 2, height * 2)  # Resize to double the original size
    resized_image = cv2.resize(gray_image, new_size, interpolation=cv2.INTER_CUBIC)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)
    
    # Use adaptive thresholding to handle different lighting conditions
    threshold_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Invert the image to have black text on white background
    inverted_image = cv2.bitwise_not(threshold_image)
    
    # Optional: Apply morphological operations to improve text clarity
    kernel = np.ones((2, 2), np.uint8)
    morph_image = cv2.morphologyEx(inverted_image, cv2.MORPH_CLOSE, kernel)
    
    return morph_image

# Function to perform OCR using EasyOCR
def perform_ocr(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert gray image to RGB
    reader = easyocr.Reader(['en'])
    results = reader.readtext(rgb_image)
    return results

# Function to draw bounding boxes around detected text
def draw_boxes(image, results):
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert gray image to BGR for drawing
    for (bbox, text, prob) in results:
        if prob > 0.5:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
            cv2.rectangle(color_image, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(color_image, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return color_image

# Function to select an image file
def select_image_file():
    Tk().withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
    return file_path

# Function to group text lines into sentences
def group_text_into_sentences(ocr_results):
    # Group results based on their vertical positions
    sorted_results = sorted(ocr_results, key=lambda x: x[0][0][1])  # Sort by y-coordinate

    sentences = []
    current_sentence = ""
    last_y = None
    line_spacing_threshold = 20  # Adjust this threshold based on your image

    for (bbox, text, prob) in sorted_results:
        if prob > 0.5:
            top_left = bbox[0]
            if last_y is None or (top_left[1] - last_y) > line_spacing_threshold:
                if current_sentence:
                    sentences.append(current_sentence.strip())
                current_sentence = text
            else:
                current_sentence += " " + text
            last_y = top_left[1]

    if current_sentence:
        sentences.append(current_sentence.strip())

    return sentences

# Function to display the OCR results in a window and save to a text file
def display_results(ocr_results, result_image, save_path):
    result_window = Toplevel()
    result_window.title("OCR Results")

    # Display the image with bounding boxes
    img = Image.fromarray(result_image)
    img = ImageTk.PhotoImage(img)
    img_label = Label(result_window, image=img)
    img_label.image = img
    img_label.pack()

    # Display the OCR text results on a white background with black text
    text_widget = Text(result_window, wrap='word', height=10, bg='white', fg='black')
    text_widget.pack(side='left', fill='both', expand=True)

    scrollbar = Scrollbar(result_window, command=text_widget.yview)
    scrollbar.pack(side='right', fill='y')

    text_widget['yscrollcommand'] = scrollbar.set

    sentences = group_text_into_sentences(ocr_results)
    text_output = "\n\n".join(sentences)

    text_widget.insert(END, text_output)

    # Save the OCR text results to a text file
    with open(save_path, 'w') as text_file:
        text_file.write(text_output)

    # Show a message indicating that the text file has been saved
    messagebox.showinfo("Saved", f"Text file is saved at {save_path}")

# Main function
def main():
    image_path = select_image_file()
    if not image_path:
        return

    preprocessed_image = preprocess_image(image_path)
    ocr_results = perform_ocr(preprocessed_image)
    result_image = draw_boxes(preprocessed_image, ocr_results)
    
    # Save path for the OCR results text file
    save_path = 'ocr_results.txt'
    display_results(ocr_results, result_image, save_path)

# Run the main function
if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    main()
    root.mainloop()