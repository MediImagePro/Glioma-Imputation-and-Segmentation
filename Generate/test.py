import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util import html
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import numpy as np

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    # Initialize lists to store PSNR and SSIM values
    psnr_values = []
    ssim_values = []

    # test
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()

        if opt.dataset_mode == 'aligned_mat':
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()
            img_path[0] = img_path[0] + str(i)
        elif opt.dataset_mode == 'unaligned_mat':
            visuals = model.get_current_visuals()
            slice_select = [opt.input_nc // 2, opt.input_nc // 2, opt.input_nc // 2]
            visuals['real_A'] = visuals['real_A'][:, :, slice_select]
            visuals['real_B'] = visuals['real_B'][:, :, slice_select]
            visuals['fake_A'] = visuals['fake_A'][:, :, slice_select]
            visuals['fake_B'] = visuals['fake_B'][:, :, slice_select]
            visuals['rec_A'] = visuals['rec_A'][:, :, slice_select]
            visuals['rec_B'] = visuals['rec_B'][:, :, slice_select]
            img_path = model.get_image_paths()
            img_path[0] = img_path[0] + str(i)
        else:
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()

        # Convert visuals to numpy arrays and ensure they are float32
        real_B = visuals['real_B'].astype(np.float32)  # Ensure float32
        fake_B = visuals['fake_B'].astype(np.float32)  # Ensure float32

        # Normalize images to [0, 1]
        real_B = (real_B + 1) / 2
        fake_B = (fake_B + 1) / 2

        # Ensure the images are in the range [0, 1]
        real_B = np.clip(real_B, 0, 1)
        fake_B = np.clip(fake_B, 0, 1)

        # Calculate PSNR and SSIM
        psnr_value = psnr(real_B, fake_B, data_range=1)
        ssim_value = ssim(real_B, fake_B, win_size=3, channel_axis=-1)

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)

        print('%04d: process image... %s, PSNR: %.3f, SSIM: %.3f' % (i, img_path, psnr_value, ssim_value))
        visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)

    webpage.save()

    # Calculate and print mean and std for PSNR and SSIM
    mean_psnr = np.mean(psnr_values)
    std_psnr = np.std(psnr_values)
    mean_ssim = np.mean(ssim_values)
    std_ssim = np.std(ssim_values)

    print('Mean PSNR: %.3f, Std PSNR: %.3f' % (mean_psnr, std_psnr))
    print('Mean SSIM: %.3f, Std SSIM: %.3f' % (mean_ssim, std_ssim))
