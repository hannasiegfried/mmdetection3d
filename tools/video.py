import subprocess
import os
import cv2

# Function to execute the image generation script
def generate_images(params):
    subprocess.run(['python', '/home/hasiegf/thesis/mmdetection3d/tools/test.py', *params])

# Function to compile images into a video
def compile_video(image_list, output_video):
    frame = cv2.imread(image_list[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))

    for image in image_list:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()

# Main script execution
def main():
    image_folder = '/media/hasiegf/data/mmdet3d/out/pictures/supervised'
    test_path = '/media/hasiegf/data/mmdet3d/out/transformer/queries/1/relu/input/zero/ep80/epoch_'
    images = [
        'Town01_Opt-scene13-frame60',
        'Town01_Opt-scene23-frame80',
        'Town01_Opt-scene24-frame80',
        'Town01_Opt-scene34-frame65',
        'Town01_Opt-scene3-frame45',
        'Town01_Opt-scene41-frame25',
        'Town01_Opt-scene45-frame25',
        'Town01_Opt-scene45-frame95',
        'Town01_Opt-scene50-frame30',
        'Town01_Opt-scene5-frame45',
    ]


    # Generate images with different parameters
    for i in range(1,80,5):
        path = test_path + str(i) + '.pth'
        generate_images(['--checkpoint', path])
        # Assuming generate_images.py saves images as 'image_0.png', 'image_1.png', ..., 'image_9.png'
        for j in images:
            img_path = os.path.join(image_folder, f'{j}.png')
            new_img_path = os.path.join(image_folder, f'{j}_{i}.png')
            os.rename(img_path, new_img_path)

    # Compile separate videos for each image index
    for j in images:
        image_list = sorted([os.path.join(image_folder, f'{j}_{i}.png') for i in range(1,80,5)])
        output_video = f'output_video_{j}.mp4'
        compile_video(image_list, output_video)

if __name__ == "__main__":
    main()