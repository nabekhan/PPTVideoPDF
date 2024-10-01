import cv2
import os
import img2pdf
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from PIL import Image
import fitz  # PyMuPDF

def is_significant_change(frame1, frame2, threshold=0.99):
    # Convert frames to grayscale for easier comparison
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between the two frames
    score, _ = ssim(gray1, gray2, full=True)

    # If the similarity score is less than the threshold, it's considered a significant change
    return score < threshold


def extract_significant_frames(video_path, output_folder, threshold=0.99, seconds_between_frames=10):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the video
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
    frames_interval = int(fps * seconds_between_frames)  # Convert seconds to frame interval

    count = 0
    saved_count = 0
    last_frame = None

    # Initialize tqdm progress bar with total frame count
    with tqdm(total=frame_count, desc="Processing frames", unit="frame") as pbar:
        while True:
            ret, frame = video.read()

            # If there are no more frames to read, break the loop
            if not ret:
                break

            if count == 0:
                # Save the first frame
                last_frame = frame
                output_file = os.path.join(output_folder, f"slide_{saved_count:04d}.jpg")
                cv2.imwrite(output_file, frame)
                saved_count += 1
            else:
                # Compare the current frame with the previous frame every few seconds
                if count % frames_interval == 0:
                    if is_significant_change(last_frame, frame, threshold):
                        last_frame = frame
                        output_file = os.path.join(output_folder, f"slide_{saved_count:04d}.jpg")
                        cv2.imwrite(output_file, frame)
                        saved_count += 1

            count += 1
            pbar.update(1)  # Update the progress bar

    video.release()
    print(f"Extracted {saved_count} significant frames.")


def create_pdf_from_frames(output_folder, output_pdf):
    # Get a list of image files in the folder
    image_files = [f for f in os.listdir(output_folder) if f.endswith('.jpg') or f.endswith('.png')]
    image_files = sorted(image_files)

    image_paths = [os.path.join(output_folder, img_file) for img_file in image_files]

    # Convert images to a PDF using img2pdf
    with open(output_pdf, "wb") as f:
        f.write(img2pdf.convert(image_paths))

    print(f"PDF created: {output_pdf}")


def delete_frames(output_folder):
    image_files = [f for f in os.listdir(output_folder) if f.endswith('.jpg') or f.endswith('.png')]
    for img_file in image_files:
        os.remove(os.path.join(output_folder, img_file))
    #print(f"Deleted {len(image_files)} frames from {output_folder}.")


def compress_pdf(input_pdf, output_pdf, output_folder, max_size=2 * 1024 * 1024, initial_quality=90, quality_step=10):
    quality = initial_quality
    compressed_pdf = output_pdf
    temp_pdfs = []  # Keep track of all intermediate compressed PDFs

    while os.path.getsize(input_pdf) > max_size and quality > 10:
        print(f"Compressing PDF: {input_pdf}, current size: {os.path.getsize(input_pdf) / (1024 * 1024):.2f}MB")
        doc = fitz.open(input_pdf)

        # Process each page and save as compressed images
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()

            # Save temp image in the output folder
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_path = os.path.join(output_folder, f"temp_page_{page_num}.jpg")
            img.save(img_path, "JPEG", quality=quality)

        # Create a new compressed PDF
        compressed_pdf = output_pdf.replace(".pdf", f"_compressed_{quality}.pdf")
        new_doc = fitz.open()
        for page_num in range(len(doc)):
            img = fitz.open(os.path.join(output_folder, f"temp_page_{page_num}.jpg"))
            rect = img[0].rect
            pdfbytes = img.convert_to_pdf()
            img_pdf = fitz.open("pdf", pdfbytes)
            new_page = new_doc.new_page(width=rect.width, height=rect.height)
            new_page.show_pdf_page(rect, img_pdf, 0)

        new_doc.save(compressed_pdf)
        new_doc.close()
        doc.close()

        # Add to the list of intermediate PDFs for cleanup
        temp_pdfs.append(compressed_pdf)

        # Check the size of the compressed PDF
        if os.path.getsize(compressed_pdf) <= max_size:
            print(f"Successfully compressed PDF to {os.path.getsize(compressed_pdf) / (1024 * 1024):.2f}MB")
            break
        else:
            quality -= quality_step

    # Rename the last compressed file to the original name
    if os.path.exists(compressed_pdf):
        os.rename(compressed_pdf, output_pdf)

    # Clean up temp files and higher-quality PDFs
    for file in os.listdir(output_folder):
        if file.startswith("temp_page_"):
            os.remove(os.path.join(output_folder, file))

    # Remove any intermediate PDF files (with _compressed_{quality} in their name)
    for temp_pdf in temp_pdfs:
        if os.path.exists(temp_pdf):
            os.remove(temp_pdf)

    return output_pdf

def process_video_to_pdf(video_path, output_folder, output_pdf, seconds_between_frames=10, threshold=0.99):
    extract_significant_frames(video_path, output_folder, threshold, seconds_between_frames)
    create_pdf_from_frames(output_folder, output_pdf)
    compress_pdf(output_pdf, output_pdf, output_folder)
    delete_frames(output_folder)



if __name__ == "__main__":
    video_input_folder = "videos"
    if not os.path.exists(video_input_folder):
        os.makedirs(video_input_folder)
    videos = os.listdir(video_input_folder)
    # Ensure the /PDF/ folder exists
    pdf_output_folder = "PDF"
    if not os.path.exists(pdf_output_folder):
        os.makedirs(pdf_output_folder)
    pdfs = os.listdir("PDF")
    for video in videos:
        name = video.split(".")[0]
        print(f"Starting {name}")
        pdfname = f"{name}.pdf"
        if pdfname in pdfs:
            print("Already completed!")
            continue
        process_video_to_pdf(f"videos/{video}", 'frames', f"{pdf_output_folder}/{pdfname}", seconds_between_frames=10)