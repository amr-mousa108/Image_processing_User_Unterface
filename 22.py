import cv2
import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import random

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pattern Recognition and Image Processing")
        self.root.geometry("1200x800")  # حجم مناسب للواجهة
        
        # تحسين الخطوط والألوان
        self.font_style = ("Arial", 10)
        self.button_style = {"font": ("Arial", 10, "bold"), "bg": "#4CAF50", "fg": "white"}
        self.frame_style = {"bg": "#f0f0f0", "padx": 10, "pady": 10}
        
        # Initialize variables
        self.original_image = None
        self.processed_image = None
        self.image_path = None
        self.is_gray = False
        self.kernel_type = "rectangle"
        self.kernel_size = 3
        self.brightness_value = 0
        self.contrast_value = 1.0
        self.noise_type = None
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        # Main frames
        self.image_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.image_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # إطار التحكم مع شريط التمرير
        self.control_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.control_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)
        
        # إضافة Canvas و Scrollbar
        self.canvas = tk.Canvas(self.control_frame, bg="#f0f0f0", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.control_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#f0f0f0")
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Image display
        self.original_label = tk.Label(self.image_frame, text="Original Image", font=("Arial", 12, "bold"), bg="#f0f0f0")
        self.original_label.pack(pady=5)
        
        self.original_panel = tk.Label(self.image_frame, bg="black", bd=2, relief=tk.SOLID)
        self.original_panel.pack(pady=5)
        
        self.processed_label = tk.Label(self.image_frame, text="Processed Image", font=("Arial", 12, "bold"), bg="#f0f0f0")
        self.processed_label.pack(pady=5)
        
        self.processed_panel = tk.Label(self.image_frame, bg="black", bd=2, relief=tk.SOLID)
        self.processed_panel.pack(pady=5)
        
        # Image operations
        self.create_image_controls()
        self.create_point_transform_controls()
        self.create_local_transform_controls()
        self.create_edge_detection_controls()
        self.create_global_transform_controls()
        self.create_morphological_controls()
        self.create_save_exit_controls()
        
    def create_image_controls(self):
        frame = tk.LabelFrame(self.scrollable_frame, text="Image Operations", font=("Arial", 10, "bold"), 
                            bg="#f0f0f0", padx=10, pady=10)
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(frame, text="Open Image", command=self.open_image, **self.button_style).pack(fill=tk.X, pady=5)
        
        self.color_var = tk.StringVar(value="default")
        color_frame = tk.Frame(frame, bg="#f0f0f0")
        color_frame.pack(fill=tk.X, pady=5)
        tk.Radiobutton(color_frame, text="Default Color", variable=self.color_var, value="default", 
                      command=self.toggle_color, bg="#f0f0f0", font=self.font_style).pack(anchor=tk.W)
        tk.Radiobutton(color_frame, text="Gray Color", variable=self.color_var, value="gray", 
                      command=self.toggle_color, bg="#f0f0f0", font=self.font_style).pack(anchor=tk.W)
        
        tk.Label(frame, text="Add Noise:", bg="#f0f0f0", font=self.font_style).pack(anchor=tk.W, pady=5)
        self.noise_var = tk.StringVar()
        noise_combo = ttk.Combobox(frame, textvariable=self.noise_var, 
                                 values=["None", "Salt & Pepper", "Gaussian", "Poisson"],
                                 font=self.font_style)
        noise_combo.pack(fill=tk.X, pady=5)
        noise_combo.set("None")
        
        tk.Button(frame, text="Apply Noise", command=self.apply_noise, **self.button_style).pack(fill=tk.X, pady=5)
    
    def create_point_transform_controls(self):
        frame = tk.LabelFrame(self.scrollable_frame, text="Point Transform", font=("Arial", 10, "bold"), 
                            bg="#f0f0f0", padx=10, pady=10)
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Brightness
        tk.Label(frame, text="Brightness:", bg="#f0f0f0", font=self.font_style).pack(anchor=tk.W, pady=5)
        self.brightness_scale = tk.Scale(frame, from_=-100, to=100, orient=tk.HORIZONTAL, 
                                        command=self.adjust_brightness, bg="#f0f0f0", font=self.font_style)
        self.brightness_scale.pack(fill=tk.X, pady=5)
        self.brightness_scale.set(0)
        
        # Contrast
        tk.Label(frame, text="Contrast:", bg="#f0f0f0", font=self.font_style).pack(anchor=tk.W, pady=5)
        self.contrast_scale = tk.Scale(frame, from_=0.1, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, 
                                      command=self.adjust_contrast, bg="#f0f0f0", font=self.font_style)
        self.contrast_scale.pack(fill=tk.X, pady=5)
        self.contrast_scale.set(1.0)
        
        # Histogram
        tk.Button(frame, text="Show Histogram", command=self.show_histogram, **self.button_style).pack(fill=tk.X, pady=5)
        tk.Button(frame, text="Histogram Equalization", command=self.histogram_equalization, **self.button_style).pack(fill=tk.X, pady=5)
    
    def create_local_transform_controls(self):
        frame = tk.LabelFrame(self.scrollable_frame, text="Local Transform", font=("Arial", 10, "bold"), 
                            bg="#f0f0f0", padx=10, pady=10)
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        buttons = [
            ("Low Pass Filter", lambda: self.apply_filter("low_pass")),
            ("High Pass Filter", lambda: self.apply_filter("high_pass")),
            ("Median Filter", lambda: self.apply_filter("median")),
            ("Averaging Filter", lambda: self.apply_filter("average"))
        ]
        
        for text, command in buttons:
            tk.Button(frame, text=text, command=command, **self.button_style).pack(fill=tk.X, pady=2)
    
    def create_edge_detection_controls(self):
        frame = tk.LabelFrame(self.scrollable_frame, text="Edge Detection", font=("Arial", 10, "bold"), 
                            bg="#f0f0f0", padx=10, pady=10)
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        buttons = [
            ("Laplacian", lambda: self.edge_detection("laplacian")),
            ("Gaussian", lambda: self.edge_detection("gaussian")),
            ("Sobel Vertical", lambda: self.edge_detection("sobel_v")),
            ("Sobel Horizontal", lambda: self.edge_detection("sobel_h")),
            ("Prewitt Vertical", lambda: self.edge_detection("prewitt_v")),
            ("Prewitt Horizontal", lambda: self.edge_detection("prewitt_h")),
            ("Canny", lambda: self.edge_detection("canny")),
            ("Zero Cross", lambda: self.edge_detection("zero_cross"))
        ]
        
        for text, command in buttons:
            tk.Button(frame, text=text, command=command, **self.button_style).pack(fill=tk.X, pady=2)
    
    def create_global_transform_controls(self):
        frame = tk.LabelFrame(self.scrollable_frame, text="Global Transform", font=("Arial", 10, "bold"), 
                            bg="#f0f0f0", padx=10, pady=10)
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(frame, text="Line Detection (Hough)", command=self.hough_line_transform, **self.button_style).pack(fill=tk.X, pady=5)
        tk.Button(frame, text="Circle Detection (Hough)", command=self.hough_circle_transform, **self.button_style).pack(fill=tk.X, pady=5)
    
    def create_morphological_controls(self):
        frame = tk.LabelFrame(self.scrollable_frame, text="Morphological Operations", font=("Arial", 10, "bold"), 
                            bg="#f0f0f0", padx=10, pady=10)
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Kernel type
        tk.Label(frame, text="Kernel Type:", bg="#f0f0f0", font=self.font_style).pack(anchor=tk.W, pady=5)
        self.kernel_var = tk.StringVar(value="rectangle")
        ttk.Combobox(frame, textvariable=self.kernel_var, 
                    values=["rectangle", "cross", "ellipse", "diamond"],
                    font=self.font_style).pack(fill=tk.X, pady=5)
        
        # Kernel size
        tk.Label(frame, text="Kernel Size:", bg="#f0f0f0", font=self.font_style).pack(anchor=tk.W, pady=5)
        self.kernel_size_scale = tk.Scale(frame, from_=3, to=15, orient=tk.HORIZONTAL, 
                                        bg="#f0f0f0", font=self.font_style)
        self.kernel_size_scale.pack(fill=tk.X, pady=5)
        self.kernel_size_scale.set(3)
        
        # Operations
        buttons = [
            ("Dilation", lambda: self.morphological_operation("dilation")),
            ("Erosion", lambda: self.morphological_operation("erosion")),
            ("Opening", lambda: self.morphological_operation("opening")),
            ("Closing", lambda: self.morphological_operation("closing")),
            ("Thinning", lambda: self.morphological_operation("thinning")),
            ("Thicken", lambda: self.morphological_operation("thicken")),
            ("Skeleton", lambda: self.morphological_operation("skeleton"))
        ]
        
        for text, command in buttons:
            tk.Button(frame, text=text, command=command, **self.button_style).pack(fill=tk.X, pady=2)
    
    def create_save_exit_controls(self):
        frame = tk.Frame(self.scrollable_frame, bg="#f0f0f0")
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(frame, text="Save Result", command=self.save_image, **self.button_style).pack(side=tk.LEFT, expand=True, padx=5)
        tk.Button(frame, text="Exit", command=self.root.quit, **self.button_style).pack(side=tk.RIGHT, expand=True, padx=5)
    
    # بقية الدوال تبقى كما هي بدون تغيير
    def open_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.original_image = cv2.imread(self.image_path)
            self.processed_image = self.original_image.copy()
            self.display_images()
    
    def display_images(self):
        if self.original_image is not None:
            # Convert BGR to RGB for display
            original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            processed_rgb = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
            
            # Resize if too large
            max_size = 500
            h, w = original_rgb.shape[:2]
            if h > max_size or w > max_size:
                scale = max_size / max(h, w)
                original_rgb = cv2.resize(original_rgb, (int(w*scale), int(h*scale)))
                processed_rgb = cv2.resize(processed_rgb, (int(w*scale), int(h*scale)))
            
            # Convert to PhotoImage
            original_img = Image.fromarray(original_rgb)
            processed_img = Image.fromarray(processed_rgb)
            
            self.original_tkimg = ImageTk.PhotoImage(image=original_img)
            self.processed_tkimg = ImageTk.PhotoImage(image=processed_img)
            
            self.original_panel.config(image=self.original_tkimg)
            self.processed_panel.config(image=self.processed_tkimg)
    
    def toggle_color(self):
        if self.original_image is not None:
            if self.color_var.get() == "gray":
                self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR)
                self.is_gray = True
            else:
                self.processed_image = self.original_image.copy()
                self.is_gray = False
            self.display_images()
    
    def apply_noise(self):
        if self.processed_image is None:
            return
            
        noise_type = self.noise_var.get()
        if noise_type == "None":
            return
            
        img = self.processed_image.copy()
        
        if noise_type == "Salt & Pepper":
            row, col, ch = img.shape
            s_vs_p = 0.5
            amount = 0.04
            out = np.copy(img)
            
            # Salt mode
            num_salt = np.ceil(amount * img.size * s_vs_p)
            coords = [np.random.randint(0, i-1, int(num_salt)) for i in img.shape]
            out[coords] = 255
            
            # Pepper mode
            num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i-1, int(num_pepper)) for i in img.shape]
            out[coords] = 0
            self.processed_image = out
            
        elif noise_type == "Gaussian":
            row, col, ch = img.shape
            mean = 0
            var = 0.1
            sigma = var**0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = img + gauss * 50
            self.processed_image = np.clip(noisy, 0, 255).astype(np.uint8)
            
        elif noise_type == "Poisson":
            vals = len(np.unique(img))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(img * vals) / float(vals)
            self.processed_image = np.clip(noisy, 0, 255).astype(np.uint8)
            
        self.display_images()
    
    def adjust_brightness(self, val):
        if self.processed_image is None:
            return
            
        self.brightness_value = int(val)
        self.update_brightness_contrast()
    
    def adjust_contrast(self, val):
        if self.processed_image is None:
            return
            
        self.contrast_value = float(val)
        self.update_brightness_contrast()
    
    def update_brightness_contrast(self):
        if self.original_image is None:
            return
            
        img = self.original_image.copy()
        if self.is_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
        self.processed_image = cv2.convertScaleAbs(img, alpha=self.contrast_value, beta=self.brightness_value)
        self.display_images()
    
    def show_histogram(self):
        if self.processed_image is None:
            return
            
        img = self.processed_image.copy()
        if len(img.shape) == 3:  # Color image
            color = ('b', 'g', 'r')
            plt.figure()
            plt.title("Color Histogram")
            plt.xlabel("Bins")
            plt.ylabel("# of Pixels")
            for i, col in enumerate(color):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                plt.plot(hist, color=col)
                plt.xlim([0, 256])
        else:  # Grayscale image
            plt.figure()
            plt.title("Grayscale Histogram")
            plt.xlabel("Bins")
            plt.ylabel("# of Pixels")
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            plt.plot(hist, color='gray')
            plt.xlim([0, 256])
        plt.show()
    
    def histogram_equalization(self):
        if self.processed_image is None:
            return
            
        img = self.processed_image.copy()
        if len(img.shape) == 3:  # Color image
            # Convert to YCrCb and equalize Y channel
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
            self.processed_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:  # Grayscale image
            self.processed_image = cv2.equalizeHist(img)
        self.display_images()
    
    def apply_filter(self, filter_type):
        if self.processed_image is None:
            return
            
        img = self.processed_image.copy()
        if len(img.shape) == 3 and not self.is_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if filter_type == "low_pass":
            kernel = np.ones((5,5), np.float32)/25
            filtered = cv2.filter2D(img, -1, kernel)
        elif filter_type == "high_pass":
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            filtered = cv2.filter2D(img, -1, kernel)
        elif filter_type == "median":
            filtered = cv2.medianBlur(img, 5)
        elif filter_type == "average":
            filtered = cv2.blur(img, (5,5))
        
        if len(self.processed_image.shape) == 3:
            filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
        self.processed_image = filtered
        self.display_images()
    
    def edge_detection(self, method):
        if self.processed_image is None:
            return
            
        img = self.processed_image.copy()
        if len(img.shape) == 3 and not self.is_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if method == "laplacian":
            edges = cv2.Laplacian(img, cv2.CV_64F)
        elif method == "gaussian":
            blur = cv2.GaussianBlur(img, (5,5), 0)
            edges = cv2.Laplacian(blur, cv2.CV_64F)
        elif method == "sobel_v":
            edges = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        elif method == "sobel_h":
            edges = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        elif method == "prewitt_v":
            kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
            edges = cv2.filter2D(img, -1, kernelx)
        elif method == "prewitt_h":
            kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
            edges = cv2.filter2D(img, -1, kernely)
        elif method == "canny":
            edges = cv2.Canny(img, 100, 200)
        elif method == "zero_cross":
            # Simple zero-crossing implementation
            laplacian = cv2.Laplacian(img, cv2.CV_64F)
            edges = np.zeros_like(laplacian)
            rows, cols = laplacian.shape
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    neighbors = [laplacian[i-1,j], laplacian[i+1,j], 
                               laplacian[i,j-1], laplacian[i,j+1]]
                    if (laplacian[i,j] > 0 and min(neighbors) < 0) or (laplacian[i,j] < 0 and max(neighbors) > 0):
                        edges[i,j] = 255
        
        edges = cv2.convertScaleAbs(edges)
        if len(self.processed_image.shape) == 3:
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        self.processed_image = edges
        self.display_images()
    
    def hough_line_transform(self):
        if self.processed_image is None:
            return
            
        img = self.processed_image.copy()
        if len(img.shape) == 3 and not self.is_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(img, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is not None:
            result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv2.line(result, (x1,y1), (x2,y2), (0,0,255), 2)
            self.processed_image = result
            self.display_images()
    
    def hough_circle_transform(self):
        if self.processed_image is None:
            return
            
        img = self.processed_image.copy()
        if len(img.shape) == 3 and not self.is_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        img = cv2.medianBlur(img, 5)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                                 param1=50, param2=30, minRadius=0, maxRadius=0)
        
        if circles is not None:
            result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                cv2.circle(result, (i[0],i[1]), i[2], (0,255,0), 2)
                cv2.circle(result, (i[0],i[1]), 2, (0,0,255), 3)
            self.processed_image = result
            self.display_images()
    
    def morphological_operation(self, operation):
        if self.processed_image is None:
            return
            
        img = self.processed_image.copy()
        if len(img.shape) == 3 and not self.is_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        kernel_type = self.kernel_var.get()
        kernel_size = self.kernel_size_scale.get()
        
        if kernel_type == "rectangle":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        elif kernel_type == "cross":
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        elif kernel_type == "ellipse":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        elif kernel_type == "diamond":
            # Diamond shape kernel (custom)
            kernel = np.zeros((kernel_size, kernel_size), np.uint8)
            center = kernel_size // 2
            for i in range(kernel_size):
                for j in range(kernel_size):
                    if abs(i - center) + abs(j - center) <= center:
                        kernel[i,j] = 1
        
        if operation == "dilation":
            result = cv2.dilate(img, kernel)
        elif operation == "erosion":
            result = cv2.erode(img, kernel)
        elif operation == "opening":
            result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        elif operation == "closing":
            result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        elif operation == "thinning":
            # Zhang-Suen thinning algorithm
            result = self.zhang_suen_thinning(img)
        elif operation == "thicken":
            # Thickening is dilation with special conditions
            result = cv2.dilate(img, kernel)
        elif operation == "skeleton":
            # Skeletonization using morphological operations
            skeleton = np.zeros(img.shape, np.uint8)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
            done = False
            
            while not done:
                eroded = cv2.erode(img, element)
                temp = cv2.dilate(eroded, element)
                temp = cv2.subtract(img, temp)
                skeleton = cv2.bitwise_or(skeleton, temp)
                img = eroded.copy()
                
                zeros = cv2.countNonZero(img)
                done = (zeros == 0)
            result = skeleton
        
        if len(self.processed_image.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        self.processed_image = result
        self.display_images()
    
    def zhang_suen_thinning(self, img):
        # Convert to binary
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        binary = binary // 255  # Convert to 0 and 1
        
        # Zhang-Suen thinning algorithm
        changing1 = changing2 = 1
        while changing1 or changing2:
            # Step 1
            changing1 = []
            rows, cols = binary.shape
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    if binary[i,j] == 1:
                        p2, p3, p4, p5, p6, p7, p8, p9 = self.get_neighbors(binary, i, j)
                        # Condition A: 2 <= B(p1) <= 6
                        bp = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                        if 2 <= bp <= 6:
                            # Condition B: A(p1) == 1
                            ap = 0
                            if p2 == 0 and p3 == 1: ap += 1
                            if p3 == 0 and p4 == 1: ap += 1
                            if p4 == 0 and p5 == 1: ap += 1
                            if p5 == 0 and p6 == 1: ap += 1
                            if p6 == 0 and p7 == 1: ap += 1
                            if p7 == 0 and p8 == 1: ap += 1
                            if p8 == 0 and p9 == 1: ap += 1
                            if p9 == 0 and p2 == 1: ap += 1
                            
                            if ap == 1:
                                # Condition C and D
                                if p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0:
                                    changing1.append((i,j))
            
            for i, j in changing1:
                binary[i,j] = 0
                
            # Step 2
            changing2 = []
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    if binary[i,j] == 1:
                        p2, p3, p4, p5, p6, p7, p8, p9 = self.get_neighbors(binary, i, j)
                        # Condition A: 2 <= B(p1) <= 6
                        bp = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                        if 2 <= bp <= 6:
                            # Condition B: A(p1) == 1
                            ap = 0
                            if p2 == 0 and p3 == 1: ap += 1
                            if p3 == 0 and p4 == 1: ap += 1
                            if p4 == 0 and p5 == 1: ap += 1
                            if p5 == 0 and p6 == 1: ap += 1
                            if p6 == 0 and p7 == 1: ap += 1
                            if p7 == 0 and p8 == 1: ap += 1
                            if p8 == 0 and p9 == 1: ap += 1
                            if p9 == 0 and p2 == 1: ap += 1
                            
                            if ap == 1:
                                # Condition C and D
                                if p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0:
                                    changing2.append((i,j))
            
            for i, j in changing2:
                binary[i,j] = 0
        
        return binary * 255
    
    def get_neighbors(self, img, i, j):
        p2 = img[i-1, j]
        p3 = img[i-1, j+1]
        p4 = img[i, j+1]
        p5 = img[i+1, j+1]
        p6 = img[i+1, j]
        p7 = img[i+1, j-1]
        p8 = img[i, j-1]
        p9 = img[i-1, j-1]
        return p2, p3, p4, p5, p6, p7, p8, p9
    
    def save_image(self):
        if self.processed_image is None:
            return
            
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", 
                                                filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if file_path:
            cv2.imwrite(file_path, self.processed_image)
            messagebox.showinfo("Success", "Image saved successfully!")

def main():
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()