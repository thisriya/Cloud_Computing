# Locally reduce resolution of images using docker
from PIL import Image
import os
import argparse

def reduce_resolution(input_path, output_path, scale_factor):
    """
    Reduce image resolution by a given scale factor
    :param input_path: Path to input image
    :param output_path: Path to save reduced image
    :param scale_factor: Factor by which to reduce resolution (0.0-1.0)
    """
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Files in /app/images: {os.listdir('/app/images')}")
    try:
        with Image.open(input_path) as img:
            # Calculate new dimensions
            width, height = img.size
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Resize image
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Save the reduced image
            resized_img.save(output_path)
            print(f"Image resolution reduced and saved to {output_path}")
            
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reduce image resolution')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('output', help='Output image path')
    parser.add_argument('--scale', type=float, default=0.5, 
                        help='Scale factor for resolution reduction (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Validate scale factor
    if args.scale <= 0 or args.scale > 1:
        raise ValueError("Scale factor must be between 0 and 1")
    
    reduce_resolution(args.input, args.output, args.scale)