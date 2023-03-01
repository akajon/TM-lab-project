import cv2
import tkinter as tk
from tkinter import filedialog, Frame
from PIL import Image, ImageTk
import copy
import numpy as np

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Converter")
        # self.master.maxsize(1000, 1000)
        self.master.config(bg="#fafff4")
        self.master.geometry("1050x650")

        left_frame = Frame(self.master, bg='grey')
        left_frame.grid(row=0, column=0, padx=10, pady=5)

        right_frame = Frame(self.master, bg='grey')
        right_frame.grid(row=0, column=1, padx=10, pady=5)

        # Create the GUI elements
        self.browse_button = tk.Button(left_frame, text="Browse for image", command=self.browse_image,
            background='#64c47d', width=36)
        self.save_button = tk.Button(left_frame, text="Save image", command=self.save_image,
            background='#64c47d', width=36)
        self.convert_button = tk.Button(left_frame, text="Convert to Grayscale", command=self.convert_to_grayscale, width=15)
        self.draw_line_button = tk.Button(left_frame, text="Draw a line", command=self.draw_a_line, width=15)
        self.detect_faces_button = tk.Button(left_frame, text="Detect faces", command=self.detect_faces, width=15)
        self.resize_up_button = tk.Button(left_frame, text="Resize to 600x600", command=self.resize_up, width=15)
        self.resize_down_button = tk.Button(left_frame, text="Resize to 300x300", command=self.resize_down, width=15)
        self.draw_an_elipse_button = tk.Button(left_frame, text="Draw an elipse", command=self.draw_an_elipse, width=15)
        self.add_blur_button = tk.Button(left_frame, text="Blur image", command=self.add_blur, width=15)
        self.canny_edge_button = tk.Button(left_frame, text="Canny edge detection", command=self.canny_edge_detection, width=15)
        self.warp_button = tk.Button(left_frame, text="Warp Perspective", command=self.warp_perspective, width=15)
        self.image_src = tk.Label(left_frame)
        self.image_out = tk.Label(right_frame)

        # Set the layout of the GUI elements
        self.browse_button.grid(row=1, column=0, columnspan=2 , padx=5, pady=5)
        self.convert_button.grid(row=2, column=0, padx=5, pady=5)
        self.draw_line_button.grid(row=2, column=1, padx=5, pady=5)
        self.detect_faces_button.grid(row=3, column=0, padx=5, pady=5)
        self.resize_up_button.grid(row=3, column=1, padx=5, pady=5)
        self.resize_down_button.grid(row=4, column=0, padx=5, pady=5)
        self.draw_an_elipse_button.grid(row=4, column=1, padx=5, pady=5)
        self.add_blur_button.grid(row=5, column=0, padx=5, pady=5)
        self.canny_edge_button.grid(row=5, column=1, padx=5, pady=5)
        self.warp_button.grid(row=6, column=0, padx=5, pady=5)
        self.save_button.grid(row=7, column=0, columnspan=2 , padx=5, pady=5)
        self.image_src.grid(row=0, column=0, columnspan=2)
        self.image_out.grid(row=0, column=0)

        self.original_img = None
        self.processed_img = None

    def browse_image(self):
        # Allow the user to select an image file
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*")])

        if file_path:
            # Load the selected image and display it
            self.original_img = cv2.imread(file_path)
            self.original_img = cv2.resize(self.original_img, (400, 400))
            img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(image=img)
            self.image_src.configure(image=img)
            self.image_src.image = img
            self.image_out.image = None

    def save_image(self):
        # Allow the user to select save path
        file = filedialog.asksaveasfile(mode='w', defaultextension=".png")
        print(str(file.name))

        if file:
            # Save image
            img = cv2.cvtColor(self.processed_img, cv2.COLOR_RGB2BGR)
            status = cv2.imwrite(file.name, img)
            print(status)

    def write_to_image_out(self, image):
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image=image)
        self.image_out.configure(image=image)
        self.image_out.image = image

    def convert_to_grayscale(self):
        if self.original_img is not None:
            # Convert the image to grayscale and display it
            gray = cv2.cvtColor(self.original_img, cv2.COLOR_RGB2GRAY)
            self.processed_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            self.write_to_image_out(self.processed_img)

    def draw_a_line(self):
        if self.original_img is not None:
            self.processed_img = copy.deepcopy(self.original_img)
            # Draw a line on the image
            cv2.line(self.processed_img, (0, 0), (self.processed_img.shape[1], self.processed_img.shape[0]), (0, 0, 0), 3)
            self.processed_img = cv2.cvtColor(self.processed_img, cv2.COLOR_BGR2RGB)
            self.write_to_image_out(self.processed_img)

    def detect_faces(self):
        if self.original_img is not None:
            self.processed_img = copy.deepcopy(self.original_img)
            # Load the face cascade
            face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            # Detect faces in the image
            faces = face_cascade.detectMultiScale(self.processed_img, scaleFactor=1.1, minNeighbors=5)

            # Draw rectangles around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(self.processed_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            self.processed_img = cv2.cvtColor(self.processed_img, cv2.COLOR_BGR2RGB)
            self.write_to_image_out(self.processed_img)

    def resize_up(self):
        if self.original_img is not None:
            self.processed_img = cv2.resize(self.processed_img, (600, 600))
            # self.processed_img = cv2.cvtColor(self.processed_img, cv2.COLOR_BGR2RGB)
            self.write_to_image_out(self.processed_img)

    def resize_down(self):
        if self.original_img is not None:
            self.processed_img = cv2.resize(self.processed_img, (300, 300))
            # self.processed_img = cv2.cvtColor(self.processed_img, cv2.COLOR_BGR2RGB)
            self.write_to_image_out(self.processed_img)

    def draw_an_elipse(self):
        if self.original_img is not None:
            self.processed_img = copy.deepcopy(self.original_img)
            cv2.ellipse(self.processed_img, (250, 250), (100, 50), 0, 0, 360, (0, 0, 0), 2)
            self.processed_img = cv2.cvtColor(self.processed_img, cv2.COLOR_BGR2RGB)
            self.write_to_image_out(self.processed_img)

    def add_blur(self):
        if self.original_img is not None:
            self.processed_img = copy.deepcopy(self.original_img)
            self.processed_img = cv2.GaussianBlur(self.processed_img, (5, 5), 0)
            self.processed_img = cv2.cvtColor(self.processed_img, cv2.COLOR_BGR2RGB)
            self.write_to_image_out(self.processed_img)

    def canny_edge_detection(self):
        if self.original_img is not None:
            self.processed_img = copy.deepcopy(self.original_img)
            gray = cv2.cvtColor(self.processed_img, cv2.COLOR_BGR2GRAY)
            self.processed_img = cv2.Canny(gray, 100, 200)
            self.processed_img = cv2.cvtColor(self.processed_img, cv2.COLOR_GRAY2RGB)
            self.write_to_image_out(self.processed_img)

    def warp_perspective(self):
        if self.original_img is not None:
            self.processed_img = copy.deepcopy(self.original_img)
            src = np.float32([[0, 0], [400, 0], [0, 400], [400, 400]])
            dst = np.float32([[40, 0], [350, 10], [0, 450], [450, 300]])

            # Apply the perspective transform to the image
            M = cv2.getPerspectiveTransform(src, dst)
            self.processed_img = cv2.warpPerspective(self.processed_img , M, (400, 400))
            self.processed_img = cv2.cvtColor(self.processed_img, cv2.COLOR_BGR2RGB)
            self.write_to_image_out(self.processed_img)

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
