import cv2
import os

def load_image(path):
    """Load image from given path"""
    image = cv2.imread(path)
    
    if image is None:
        print("Error: Could not load image.")
        return None
    
    return image


def resize_image(image, width=300, height=300):
    """Resize image"""
    resized = cv2.resize(image, (width, height))
    return resized


def convert_to_gray(image):
    """Convert image to grayscale"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def detect_edges(image):
    """Apply Canny edge detection"""
    edges = cv2.Canny(image, 100, 200)
    return edges


def blur_image(image):
    """Apply Gaussian blur"""
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred


def save_image(image, filename):
    """Save processed image"""
    cv2.imwrite(filename, image)
    print(f"Saved: {filename}")


def main():
    print("📷 Image Processing Toolkit Started")

    image_path = input("Enter image path: ")

    image = load_image(image_path)

    if image is None:
        return

    resized = resize_image(image)
    gray = convert_to_gray(resized)
    edges = detect_edges(gray)
    blurred = blur_image(resized)

    # Create output folder
    if not os.path.exists("output"):
        os.makedirs("output")

    save_image(resized, "output/resized.jpg")
    save_image(gray, "output/grayscale.jpg")
    save_image(edges, "output/edges.jpg")
    save_image(blurred, "output/blurred.jpg")

    print("✅ Processing Completed")


if __name__ == "__main__":
    main()
