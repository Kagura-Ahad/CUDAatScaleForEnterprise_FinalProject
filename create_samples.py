import os
import random

#Below function writes a grayscale image in PGM P5 format
def write_pgm(filename: str, image_data: list[int], width: int, height: int):
    with open(filename, 'wb') as f:
        f.write(f"P5\n{width} {height}\n255\n".encode()) #Because PGM P5 is binary
        f.write(bytes(image_data))

#Below function creates a gradient image
def create_gradient_image(width: int = 512, height: int = 512) -> list[int]:
    image: list[int] = []
    for y in range(height):
        for x in range(width):
            value = int((x / width + y / height) * 255 / 2) #Gradient from black to white
            image.append(value)
    return image

#Below function creates a checkerboard pattern
def create_checkerboard(width: int = 512, height: int = 512, square_size: int = 32) -> list[int]:
    image: list[int] = []
    for y in range(height):
        for x in range(width):
            if ((y // square_size) + (x // square_size)) % 2 == 0:
                image.append(255)
            else:
                image.append(0)
    return image

#Below function creates an image with circles
def create_circle_image(width: int = 512, height: int = 512) -> list[int]:
    image: list[int] = []
    centers: list[tuple[int, int, int]] = [(128, 128, 80), (384, 128, 60), (256, 384, 100)]
    
    for y in range(height):
        for x in range(width):
            value = 0
            for cx, cy, radius in centers:
                if (x - cx)**2 + (y - cy)**2 <= radius**2:
                    value = 200
                    break
            image.append(value)
    return image

#Below function creates a random noise image
def create_noise_image(width: int = 512, height: int = 512) -> list[int]:
    return [random.randint(0, 255) for _ in range(width * height)]

#Below function creates horizontal stripes
def create_stripes_image(width: int = 512, height: int = 512, stripe_width: int = 20) -> list[int]:
    image: list[int] = []
    for y in range(height):
        for _ in range(width):
            if (y % (stripe_width * 2)) < stripe_width:
                image.append(255)
            else:
                image.append(0)
    return image

def main():
    #Now we create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    print("Creating sample PGM images...")
    
    #And now we create various test images with (name, generator_func, width, height)
    images: list[tuple[str, list[int], int, int]] = [
        ('gradient', create_gradient_image(), 512, 512),
        ('checkerboard', create_checkerboard(), 512, 512),
        ('circles', create_circle_image(), 512, 512),
        ('noise', create_noise_image(), 512, 512),
        ('stripes', create_stripes_image(), 512, 512),
        ('gradient_large', create_gradient_image(1024, 768), 1024, 768),
        ('checkerboard_small', create_checkerboard(256, 256, 16), 256, 256),
        ('circles_medium', create_circle_image(640, 480), 640, 480),
    ]
    
    for name, image_data, width, height in images:
        filename = f'data/{name}.pgm'
        write_pgm(filename, image_data, width, height)
        print(f"  Created: {filename} ({width}x{height})")
    
    print(f"\nSuccessfully created {len(images)} sample images in the data/ directory")

if __name__ == '__main__':
    main()