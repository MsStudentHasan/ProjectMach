# import os
# import cv2
# import numpy as np
# from pathlib import Path
# from tqdm import tqdm
# import torch
# import torch.nn.functional as F
# from multiprocessing import Pool, cpu_count
# import time
# from concurrent.futures import ThreadPoolExecutor, as_completed
#
#
# def create_color_map_tensor():
#     """Create CUDA tensor for color mapping"""
#     # Define color mapping (RGB values)
#     color_map_rgb = torch.tensor([
#         [0, 0, 0],  # Background - Black -> 0
#         [255, 0, 0],  # Tool shaft - Red -> 3
#         [0, 255, 0],  # Tool clasper - Green -> 1
#         [0, 0, 255],  # Tool wrist - Blue -> 2
#         [255, 255, 0],  # Thread - Yellow -> 5
#         [255, 0, 255],  # Clamps - Magenta -> 8
#         [0, 255, 255],  # Suturing needle - Cyan -> 4
#         [128, 128, 128],  # Suction tool - Gray -> 6
#         [255, 165, 0],  # Catheter - Orange -> 9
#         [128, 0, 128]  # Needle Holder - Purple -> 7
#     ], dtype=torch.float32)
#
#     # Define target grayscale values
#     target_values = torch.tensor([0, 3, 1, 2, 5, 8, 4, 6, 9, 7], dtype=torch.uint8)
#
#     return color_map_rgb, target_values
#
#
# def rgb_to_grayscale_cuda_batch(images, color_map_rgb, target_values):
#     """Convert batch of RGB images to grayscale using CUDA"""
#     if not images:
#         return []
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     color_map_rgb = color_map_rgb.to(device)
#     target_values = target_values.to(device)
#
#     batch_results = []
#
#     for image in images:
#         if image is None:
#             batch_results.append(None)
#             continue
#
#         # Convert numpy array to torch tensor and move to device
#         image_tensor = torch.from_numpy(image).float().to(device)
#         h, w, c = image_tensor.shape
#
#         # Reshape for efficient computation
#         pixels = image_tensor.view(-1, 3)  # (h*w, 3)
#
#         # Compute distances to all colors in the map (vectorized)
#         distances = torch.norm(
#             pixels.unsqueeze(1) - color_map_rgb.unsqueeze(0),
#             dim=2,
#             p=2
#         )
#
#         # Find closest color index for each pixel
#         closest_indices = torch.argmin(distances, dim=1)
#
#         # Map to target grayscale values
#         grayscale_flat = target_values[closest_indices]
#
#         # Reshape back to image dimensions
#         grayscale_image = grayscale_flat.view(h, w).cpu().numpy().astype(np.uint8)
#
#         batch_results.append(grayscale_image)
#
#     return batch_results
#
#
# def load_image_batch(image_paths):
#     """Load a batch of images"""
#     images = []
#     valid_paths = []
#
#     for image_path in image_paths:
#         try:
#             image = cv2.imread(str(image_path))
#             if image is None:
#                 images.append(None)
#                 valid_paths.append(image_path)
#                 continue
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             images.append(image)
#             valid_paths.append(image_path)
#         except Exception as e:
#             print(f"Error loading {image_path}: {e}")
#             images.append(None)
#             valid_paths.append(image_path)
#
#     return images, valid_paths
#
#
# def process_batch(image_paths_batch, color_map_rgb, target_values):
#     """Process a single batch of images"""
#     images, valid_paths = load_image_batch(image_paths_batch)
#     grayscale_images = rgb_to_grayscale_cuda_batch(images, color_map_rgb, target_values)
#
#     results = []
#     for image_path, grayscale_image in zip(valid_paths, grayscale_images):
#         if grayscale_image is not None:
#             try:
#                 parent_dir = image_path.parent.parent
#                 output_dir = parent_dir / "segmentation"
#                 output_dir.mkdir(exist_ok=True)
#                 output_path = output_dir / image_path.name
#                 cv2.imwrite(str(output_path), grayscale_image)
#                 results.append((image_path, True, None))
#             except Exception as e:
#                 results.append((image_path, False, str(e)))
#         else:
#             results.append((image_path, False, "Failed to process image"))
#
#     return results
#
#
# def process_with_cuda_batches(image_paths, color_map_rgb, target_values, batch_size=64):
#     """Process images in batches using CUDA with manual batching"""
#     total_images = len(image_paths)
#     processed_count = 0
#     failed_count = 0
#     failed_files = []
#
#     # Create batches
#     batches = [image_paths[i:i + batch_size] for i in range(0, total_images, batch_size)]
#
#     with tqdm(total=total_images, desc="Processing images with CUDA", unit="image") as pbar:
#         for batch_paths in batches:
#             results = process_batch(batch_paths, color_map_rgb, target_values)
#
#             for result in results:
#                 image_path, success, error = result
#                 if success:
#                     processed_count += 1
#                 else:
#                     failed_count += 1
#                     failed_files.append((image_path, error))
#
#             pbar.update(len(batch_paths))
#             pbar.set_postfix({
#                 "Processed": processed_count,
#                 "Failed": failed_count,
#                 "Success Rate": f"{(processed_count / total_images * 100):.1f}%"
#             })
#
#     print(f"\nCUDA batch processing complete: {processed_count} successful, {failed_count} failed")
#     if failed_files:
#         print("First 5 failed files:")
#         for file, error in failed_files[:5]:
#             print(f"  {file}: {error}")
#
#
# def process_with_multiprocessing(image_paths, color_map_rgb, target_values, num_workers=None):
#     """Process images using multiprocessing"""
#     if num_workers is None:
#         num_workers = min(cpu_count(), 8)
#
#     print(f"Using {num_workers} workers for parallel processing")
#
#     # Prepare arguments for multiprocessing
#     args = [(path, color_map_rgb.cpu(), target_values.cpu()) for path in image_paths]
#
#     processed_count = 0
#     failed_count = 0
#     failed_files = []
#
#     with Pool(num_workers) as pool:
#         results = list(tqdm(
#             pool.imap(process_single_image, args),
#             total=len(image_paths),
#             desc="Parallel processing",
#             unit="image"
#         ))
#
#     # Count results
#     for result in results:
#         if result and result[1]:
#             processed_count += 1
#         else:
#             failed_count += 1
#             if result:
#                 failed_files.append((result[0], result[2]))
#
#     print(f"Multiprocessing complete: {processed_count} successful, {failed_count} failed")
#     if failed_files:
#         print("Failed files:")
#         for file, error in failed_files[:5]:
#             print(f"  {file}: {error}")
#         if len(failed_files) > 5:
#             print(f"  ... and {len(failed_files) - 5} more")
#
#
# def process_single_image(args):
#     """Process single image (for multiprocessing)"""
#     image_path, color_map_rgb, target_values = args
#     try:
#         image = cv2.imread(str(image_path))
#         if image is None:
#             return image_path, False, "Could not read image"
#
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#         # Convert to grayscale using CUDA
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         image_tensor = torch.from_numpy(image).float().to(device)
#         h, w, c = image_tensor.shape
#
#         pixels = image_tensor.view(-1, 3)
#         distances = torch.norm(
#             pixels.unsqueeze(1) - color_map_rgb.to(device).unsqueeze(0),
#             dim=2,
#             p=2
#         )
#         closest_indices = torch.argmin(distances, dim=1)
#         grayscale_flat = target_values.to(device)[closest_indices]
#         grayscale_image = grayscale_flat.view(h, w).cpu().numpy().astype(np.uint8)
#
#         # Save the image
#         parent_dir = image_path.parent.parent
#         output_dir = parent_dir / "segmentation"
#         output_dir.mkdir(exist_ok=True)
#         output_path = output_dir / image_path.name
#         cv2.imwrite(str(output_path), grayscale_image)
#
#         return image_path, True, None
#
#     except Exception as e:
#         return image_path, False, str(e)
#
#
# def process_images_parallel(input_path, batch_size=64, num_workers=None):
#     """Process images using parallel processing with CUDA acceleration"""
#     base_path = Path(input_path)
#
#     # Find all image files
#     image_paths = []
#     for root, dirs, files in os.walk(base_path):
#         if "RGB_segmentation" in root.split(os.sep):
#             for file in files:
#                 if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                     image_paths.append(Path(root) / file)
#
#     print(f"Found {len(image_paths)} images to process")
#
#     if not image_paths:
#         print("No images found to process!")
#         return
#
#     # Check CUDA availability
#     cuda_available = torch.cuda.is_available()
#     print(f"CUDA available: {cuda_available}")
#     if cuda_available:
#         print(f"GPU: {torch.cuda.get_device_name(0)}")
#         print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
#
#     # Create color map tensors
#     color_map_rgb, target_values = create_color_map_tensor()
#
#     # Always use CUDA batch processing since we have CUDA
#     print("Using CUDA batch processing...")
#     process_with_cuda_batches(image_paths, color_map_rgb, target_values, batch_size)
#
#
# def main():
#     # Set your input path here
#     input_path = r"D:\ProjectMach\output\output_step_02"
#
#     # Verify the path exists
#     if not os.path.exists(input_path):
#         print(f"Error: Path '{input_path}' does not exist!")
#         return
#
#     print(f"Starting processing of: {input_path}")
#     print(f"Available CPU cores: {cpu_count()}")
#
#     start_time = time.time()
#
#     # Process images with optimal method
#     process_images_parallel(input_path, batch_size=128, num_workers=None)
#
#     end_time = time.time()
#     print(f"Total processing time: {end_time - start_time:.2f} seconds")
#     print(f"Processing speed: {16295 / (end_time - start_time):.2f} images/second")
#
#
# if __name__ == "__main__":
#     # Set better multiprocessing settings for Windows
#     torch.set_num_threads(1)
#     main()

import torch
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
