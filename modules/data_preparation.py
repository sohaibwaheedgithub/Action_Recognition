from constants import CLASSES
from bootstrap_helper import BootstrapHelper




CLASSES = ['duck_position_a1', 'duck_position_a2', 'duck_position_a3', 'duck_position_a4']

for action in CLASSES:
    print("PROCESSING {}\n".format(action.upper()))
    bootstrap_images_in_folder = f'dataset/frames_new_2/{action}'
    bootstrap_images_out_folder = f'dataset/frames_new_2/{action}/output_imgs'
    bootstrap_csvs_out_path = f'dataset/frames_new_2/{action}/landmarks.csv'


    # Initialize helper.
    bootstrap_helper = BootstrapHelper(
        images_in_folder=bootstrap_images_in_folder,
        images_out_folder=bootstrap_images_out_folder,
        csvs_out_path=bootstrap_csvs_out_path,
    )

    # Check how many pose classes and images for them are available.
    #bootstrap_helper.print_images_in_statistics()


    # Bootstrap all images.
    # Set limit to some small number for debug.
    bootstrap_helper.bootstrap(per_pose_class_limit=None)
    
    

    # Check how many images were bootstrapped.
    #bootstrap_helper.print_images_out_statistics()