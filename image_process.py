import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import copy

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Converter")
        # self.master.geometry("1280x720")

        # Create the GUI elements
        self.browse_button = tk.Button(self.master, text="Browse", command=self.browse_image)
        self.convert_button = tk.Button(self.master, text="Convert to Grayscale", command=self.convert_to_grayscale)
        self.draw_line_button = tk.Button(self.master, text="Draw a line", command=self.draw_a_line)
        self.detect_faces_button = tk.Button(self.master, text="Detect faces", command=self.detect_faces)
        self.image_src = tk.Label(self.master)
        self.image_out = tk.Label(self.master)

        # Set the layout of the GUI elements
        self.browse_button.grid(row=0, column=0, padx=5, pady=5)
        self.convert_button.grid(row=0, column=1, padx=5, pady=5)
        self.draw_line_button.grid(row=0, column=2, padx=5, pady=5)
        self.detect_faces_button.grid(row=0, column=3, padx=5, pady=5)
        self.image_src.grid(row=1, column=0, columnspan=2)
        self.image_out.grid(row=1, column=10, columnspan=2)

        self.original_img = None
        self.processed_img = None

    def browse_image(self):
        # Allow the user to select an image file
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*")])

        if file_path:
            # Load the selected image and display it
            self.original_img = cv2.imread(file_path)
            self.original_img = cv2.resize(self.original_img, (500, 500))
            img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(image=img)
            self.image_src.configure(image=img)
            self.image_src.image = img
            self.image_out.image = None
        
    def write_to_image_out(self, image):
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image=image)
        self.image_out.configure(image=image)
        self.image_out.image = image

    def convert_to_grayscale(self):
        if self.original_img is not None:
            # Convert the image to grayscale and display it
            gray = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
            self.processed_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            self.write_to_image_out(self.processed_img)

    def draw_a_line(self):
        if self.original_img is not None:
            # Draw a line on the image
            self.processed_img = copy.deepcopy(self.original_img)
            cv2.line(self.processed_img, (0, 0), (self.processed_img.shape[1], self.processed_img.shape[0]), (0, 0, 0), 3)
            img = cv2.cvtColor(self.processed_img, cv2.COLOR_BGR2RGB)
            self.write_to_image_out(img)

    def detect_faces(self):
        if self.original_img is not None:
            # Draw a line on the image
            self.processed_img = copy.deepcopy(self.original_img)
            # Load the face cascade
            face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            # Detect faces in the image
            faces = face_cascade.detectMultiScale(self.processed_img, scaleFactor=1.1, minNeighbors=5)

            # Draw rectangles around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(self.processed_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            img = cv2.cvtColor(self.processed_img, cv2.COLOR_BGR2RGB)
            self.write_to_image_out(img)

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
