"""
================================================================================
OpenCV Image Editing & Computer Vision Suite
Assignment Submission: Practical Image Editing Skills with OpenCV (Python)
Reference: OpenCV Documentation (https://docs.opencv.org/5.0/)
================================================================================
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

class OpenCVImageEditor:
    """
    A comprehensive image editing toolkit showcasing core and advanced
    computer vision manipulation techniques using OpenCV (cv2).
    """

    def __init__(self, image_path: str):
        self.image_path = image_path
        self.original = cv2.imread(image_path)
        
        if self.original is None:
            raise FileNotFoundError(f"Error: Could not load image from '{image_path}'")
        
        self.height, self.width, self.channels = self.original.shape
        self.output_dir = "edited_outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("=" * 65)
        print(" OpenCV Image Editing Studio Initialized")
        print(f" Loaded Image : {os.path.basename(image_path)}")
        print(f" Resolution   : {self.width} x {self.height} pixels | Channels: {self.channels}")
        print(f" Output Folder: {self.output_dir}/")
        print("=" * 65)

    # --------------------------------------------------------------------------
    # 1. Geometric Transformations
    # --------------------------------------------------------------------------
    def resize_image(self, target_width=600, target_height=400):
        """Resizes the image using bilinear interpolation."""
        resized = cv2.resize(self.original, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        return resized

    def crop_roi(self, start_x_ratio=0.1, start_y_ratio=0.1, width_ratio=0.8, height_ratio=0.8):
        """Crops a central Region of Interest (ROI) using NumPy slicing."""
        y1 = int(self.height * start_y_ratio)
        y2 = int(self.height * (start_y_ratio + height_ratio))
        x1 = int(self.width * start_x_ratio)
        x2 = int(self.width * (start_x_ratio + width_ratio))
        cropped = self.original[y1:y2, x1:x2]
        return cropped

    def flip_image(self, mode="horizontal"):
        """
        Flips the image:
        0 = Vertical flip, 1 = Horizontal flip, -1 = Both axes
        """
        flip_code = 1 if mode == "horizontal" else (0 if mode == "vertical" else -1)
        flipped = cv2.flip(self.original, flip_code)
        return flipped

    def rotate_image(self, angle=45, scale=1.0):
        """Rotates the image around its center by a specified angle using an affine matrix."""
        center = (self.width // 2, self.height // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(self.original, rot_matrix, (self.width, self.height), borderMode=cv2.BORDER_REFLECT)
        return rotated

    # --------------------------------------------------------------------------
    # 2. Color Space & Tone Manipulations
    # --------------------------------------------------------------------------
    def to_grayscale(self):
        """Converts BGR image to single-channel Grayscale."""
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        return gray

    def invert_colors(self):
        """Inverts pixel intensities (Photographic Negative)."""
        negative = cv2.bitwise_not(self.original)
        return negative

    def adjust_brightness_contrast(self, contrast=1.4, brightness=30):
        """
        Adjusts contrast (alpha) and brightness (beta).
        Formula: Output(x,y) = alpha * Input(x,y) + beta
        """
        adjusted = cv2.convertScaleAbs(self.original, alpha=contrast, beta=brightness)
        return adjusted

    def apply_sepia_filter(self):
        """Applies a vintage warm sepia tone using a 3x3 color transformation matrix."""
        kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        sepia = cv2.transform(self.original, kernel)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)
        return sepia

    def boost_saturation(self, factor=1.5):
        """Converts to HSV color space and boosts the Saturation channel."""
        hsv = cv2.cvtColor(self.original, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return boosted

    # --------------------------------------------------------------------------
    # 3. Filtering & Smoothing
    # --------------------------------------------------------------------------
    def apply_gaussian_blur(self, kernel_size=(15, 15), sigma=0):
        """Applies Gaussian Blur smoothing to remove high-frequency noise."""
        blurred = cv2.GaussianBlur(self.original, kernel_size, sigma)
        return blurred

    def apply_bilateral_filter(self, diameter=9, sigma_color=75, sigma_space=75):
        """
        Bilateral Filter: Smooths textures and surfaces while preserving sharp edges
        (frequently used for portrait skin smoothing and photographic stylization).
        """
        filtered = cv2.bilateralFilter(self.original, d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        return filtered

    # --------------------------------------------------------------------------
    # 4. Edge Detection & Sharpening
    # --------------------------------------------------------------------------
    def detect_edges_canny(self, threshold1=100, threshold2=200):
        """Detects structural boundaries using the Canny Edge Detection algorithm."""
        gray = self.to_grayscale()
        edges = cv2.Canny(gray, threshold1, threshold2)
        return edges

    def sharpen_image(self):
        """Enhances edges and fine details using an unsharp masking kernel."""
        sharpen_kernel = np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ])
        sharpened = cv2.filter2D(self.original, -1, sharpen_kernel)
        return sharpened

    # --------------------------------------------------------------------------
    # 5. Thresholding (Binarization)
    # --------------------------------------------------------------------------
    def apply_otsu_threshold(self):
        """Converts to grayscale and computes the optimal global threshold automatically (Otsu's method)."""
        gray = self.to_grayscale()
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    # --------------------------------------------------------------------------
    # 6. Annotations, Borders & Watermarking
    # --------------------------------------------------------------------------
    def add_watermark_and_border(self, watermark_text="OpenCV 5.0 Image Editing"):
        """Adds an aesthetic framed border and a branded watermark with drop shadow."""
        # Add border
        border_size = 30
        framed = cv2.copyMakeBorder(
            self.original,
            top=border_size, bottom=border_size + 40, left=border_size, right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=(20, 20, 20)
        )
        
        # Add drop shadow and primary watermark text
        text_origin = (border_size + 10, framed.shape[0] - 20)
        shadow_origin = (text_origin[0] + 2, text_origin[1] + 2)
        
        cv2.putText(framed, watermark_text, shadow_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(framed, watermark_text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2, cv2.LINE_AA)
        return framed

    # --------------------------------------------------------------------------
    # Batch Processing & Master Showcase Generation
    # --------------------------------------------------------------------------
    def run_all_edits(self):
        """Executes all image editing transformations and exports results."""
        print("\nExecuting Image Editing Pipeline...")
        
        edits = {
            "01_original": self.original,
            "02_grayscale": self.to_grayscale(),
            "03_contrast_brightness": self.adjust_brightness_contrast(),
            "04_sepia_vintage": self.apply_sepia_filter(),
            "05_saturated_vibrant": self.boost_saturation(),
            "06_gaussian_blur": self.apply_gaussian_blur(),
            "07_bilateral_smooth": self.apply_bilateral_filter(),
            "08_sharpened": self.sharpen_image(),
            "09_canny_edges": self.detect_edges_canny(),
            "10_otsu_threshold": self.apply_otsu_threshold(),
            "11_color_inversion": self.invert_colors(),
            "12_framed_watermarked": self.add_watermark_and_border()
        }

        # Save individual edited images
        for name, img in edits.items():
            filepath = os.path.join(self.output_dir, f"{name}.jpg")
            cv2.imwrite(filepath, img)
            print(f" [OK] Saved: {filepath}")

        # Create multi-panel showcase comparison grid
        self.create_showcase_grid(edits)

    def create_showcase_grid(self, edits: dict):
        """Generates a professional 3x4 comparison collage using Matplotlib."""
        print("\nGenerating Side-by-Side Comparison Showcase Grid...")
        titles = [
            "1. Original Image",
            "2. Grayscale Conversion",
            "3. High Contrast & Brightness",
            "4. Sepia Vintage Tone",
            "5. Vibrance & Saturation Boost",
            "6. Gaussian Smoothing (Blur)",
            "7. Bilateral Edge-Preserving Filter",
            "8. Detail Sharpening",
            "9. Canny Edge Detection",
            "10. Otsu Automatic Thresholding",
            "11. Color Inversion (Negative)",
            "12. Framed with Watermark"
        ]

        fig, axes = plt.subplots(3, 4, figsize=(18, 12))
        axes = axes.flatten()

        for idx, (name, img) in enumerate(edits.items()):
            ax = axes[idx]
            if len(img.shape) == 2:  # Grayscale / Binary
                ax.imshow(img, cmap="gray")
            else:  # Color BGR -> RGB
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(titles[idx], fontsize=11, fontweight="bold", pad=8)
            ax.axis("off")

        plt.suptitle("OpenCV Image Editing & Computer Vision Portfolio", fontsize=18, fontweight="bold", y=0.98)
        plt.tight_layout()
        
        showcase_path = "image_editing_showcase.jpg"
        plt.savefig(showcase_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f" [OK] Master Showcase Collage Saved to: {showcase_path}")
        print("\n" + "=" * 65)
        print(" All image editing tasks completed successfully!")
        print("=" * 65)

# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Choose input image: prefer local sample_input.png or Desktop/Gemini.png
    input_file = "sample_input.png"
    if not os.path.exists(input_file):
        fallback = r"C:\Users\dd\Desktop\Gemini.png"
        if os.path.exists(fallback):
            input_file = fallback
        else:
            # Create a rich default canvas if no file exists
            print("[!] Generating synthetic demonstration image...")
            demo = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.rectangle(demo, (50, 50), (750, 550), (220, 220, 220), -1)
            cv2.circle(demo, (400, 300), 150, (30, 144, 255), -1)
            cv2.putText(demo, "OpenCV 5.0", (220, 320), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
            cv2.imwrite("sample_input.png", demo)
            input_file = "sample_input.png"

    editor = OpenCVImageEditor(input_file)
    editor.run_all_edits()
