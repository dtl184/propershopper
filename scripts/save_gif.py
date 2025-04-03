import imageio
import os

# Directory where the JPG images are located
images_dir = 'videos/basket/'
# Output GIF file name
output_gif_path = 'videos/basket/record.gif'

# Read images
images = []
for file_name in sorted(os.listdir(images_dir)):
    # print(file_name)
    if file_name.endswith('.png'):
        file_path = os.path.join(images_dir, file_name)
        images.append(imageio.imread(file_path))

# Save as a GIF
imageio.mimsave(output_gif_path, images, fps=144)  # Adjust fps (frames per second) as needed
