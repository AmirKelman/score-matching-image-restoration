import torch
print(torch.__version__)
print(torch.cuda.is_available()

# Example degradation and restoration
clean_image = torch.from_numpy(x[0]).to(args.device).view(1, 32, 32)
degraded_image = downsample_and_upsample(clean_image, factor=4)
noise_level = estimate_noise_level(degraded_image.cpu().numpy(), clean_image.cpu().numpy())
restored_image = restore_image(model, degraded_image, noise_level, args.device)

plt.subplot(1, 3, 1)
plt.title("Clean Image")
plt.imshow(clean_image.cpu().numpy().reshape(32, 32), cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Degraded Image")
plt.imshow(degraded_image.cpu().numpy().reshape(32, 32), cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Restored Image")
# Ensure the restored image has the correct shape before reshaping
if restored_image.size == 32 * 32:
    plt.imshow(restored_image.reshape(32, 32), cmap='gray')
else:
# Reshape restored_image to the correct dimensions
    restored_image = restored_image.reshape(-1, 32, 32)
plt.imshow(restored_image[0], cmap='gray')

plt.show()

# (i) upscaling after downsampling by factors 2 and 4
degraded_image = downsample_and_upsample(torch.from_numpy(x[0]).to(args.device).view(1, 32, 32), factor=4)
noise_level = estimate_noise_level(degraded_image.cpu().numpy(), x.reshape(-1, 32, 32)[0])
restored_image = restore_image(model, degraded_image, noise_level, args.device)
plt.subplot(1, 3, 1)
plt.title("Clean Image")
plt.imshow(x[0].reshape(32, 32), cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Degraded Image")
plt.imshow(degraded_image.cpu().numpy().reshape(32, 32), cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Restored Image")
if restored_image.size == 32 * 32:
    plt.imshow(restored_image.reshape(32, 32), cmap='gray')
else:
    restored_image = restored_image.reshape(-1, 32, 32)
plt.imshow(restored_image[0], cmap='gray')
plt.show()

degraded_image = downsample_and_upsample(torch.from_numpy(x[0]).to(args.device).view(1, 32, 32), factor=2)
noise_level = estimate_noise_mad(degraded_image.cpu().numpy())
restored_image = restore_image(model, degraded_image, noise_level, args.device)
plt.subplot(1, 3, 1)
plt.title("Clean Image")
plt.imshow(x[0].reshape(32, 32), cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Degraded Image")
plt.imshow(degraded_image.cpu().numpy().reshape(32, 32), cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Restored Image")
if restored_image.size == 32 * 32:
    plt.imshow(restored_image.reshape(32, 32), cmap='gray')
else:
    restored_image = restored_image.reshape(-1, 32, 32)
plt.imshow(restored_image[0], cmap='gray')
plt.show()

# (ii) inpainting (filling in) missing quarter and half of the image
mask = np.ones((32, 32))
mask[:16, :16] = 0
degraded_image = inpaint(torch.from_numpy(x[0]).to(args.device).view(1, 32, 32), mask)
noise_level = estimate_noise_level(degraded_image.cpu().numpy(), x.reshape(-1, 32, 32)[0])
restored_image = restore_image(model, degraded_image, noise_level, args.device)
plt.subplot(1, 3, 1)
plt.title("Clean Image")
plt.imshow(x[0].reshape(32, 32), cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Degraded Image")
plt.imshow(degraded_image.cpu().numpy().reshape(32, 32), cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Restored Image")
if restored_image.size == 32 * 32:
    plt.imshow(restored_image.reshape(32, 32), cmap='gray')
else:
    restored_image = restored_image.reshape(-1, 32, 32)
plt.imshow(restored_image[0], cmap='gray')
plt.show()

mask = np.ones((32, 32))
mask[:16, :] = 0
degraded_image = inpaint(torch.from_numpy(x[0]).to(args.device).view(1, 32, 32), mask)
noise_level = estimate_noise_level(degraded_image.cpu().numpy(), x.reshape(-1, 32, 32)[0])
restored_image = restore_image(model, degraded_image, noise_level, args.device)
plt.subplot(1, 3, 1)
plt.title("Clean Image")
plt.imshow(x[0].reshape(32, 32), cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Degraded Image")
plt.imshow(degraded_image.cpu().numpy().reshape(32, 32), cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Restored Image")
if restored_image.size == 32 * 32:
    plt.imshow(restored_image.reshape(32, 32), cmap='gray')
else:
    restored_image = restored_image.reshape(-1, 32, 32)
plt.imshow(restored_image[0], cmap='gray')
plt.show()
