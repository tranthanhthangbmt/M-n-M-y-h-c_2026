# Goal Description
The goal is to convert the book "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" (provided in MHT/HTML format) into a structured course website using Docsify, similar to the existing `webAIAccounting` project. The website will be located at `d:\DongAUniversity\TÀI LIỆU DẠY HỌC_2024-2025\Môn Máy học_2026\machineLearningWeb`. 
The content must be split into chapters, and all images must be correctly extracted into a `Figures` folder, organized by chapter, and named according to their IDs in the book (e.g., `Hình 1-1.png`).

## User Review Required
> [!IMPORTANT]
> The source material is currently in a large Word-generated HTML file (from the `.mht` file). Parsing this format will involve extracting text and images, and matching images to their captions (e.g., `Hình 1-1`). 
> Please review the plan below. I will write a custom Python script to handle this extraction robustly.

## Open Questions
> [!NOTE]
> 1. Do you want the `index.html` to be exactly identical in style to `webAIAccounting` (e.g., requiring password "UDA") or should I remove the password protection for this new site?
> 2. The images are currently in the `Hands-On Machine Learning with ScikitLearn, Keras, and TensorFlow2_files` directory. The script will map these images to their captions and copy them into `machineLearningWeb/Figures/CH0X/`. Are you okay with the script running in the background to do this?

## Proposed Changes

---

### `machineLearningWeb` Directory Initialization
I will set up the base structure for the new Docsify website.

#### [NEW] [index.html](file:///d:/DongAUniversity/TÀI LIỆU DẠY HỌC_2024-2025/Môn Máy học_2026/machineLearningWeb/index.html)
Base HTML for the Docsify site, customized for the "Máy học 2026" course.

#### [NEW] [_sidebar.md](file:///d:/DongAUniversity/TÀI LIỆU DẠY HỌC_2024-2025/Môn Máy học_2026/machineLearningWeb/docs/_sidebar.md)
The sidebar navigation linking to all the extracted chapters (Chương 1 to Chương 19).

#### [NEW] [README.md](file:///d:/DongAUniversity/TÀI LIỆU DẠY HỌC_2024-2025/Môn Máy học_2026/machineLearningWeb/README.md)
The homepage for the course website.

---

### Content Extraction Script
Since the source is a 126MB MHT file (which I have partially extracted to a 9.4MB HTML file), I will write a Python script to do the heavy lifting of converting it to Markdown and organizing the images.

#### [NEW] [build_ml_web.py](file:///d:/DongAUniversity/TÀI LIỆU DẠY HỌC_2024-2025/Môn Máy học_2026/Tài liệu/build_ml_web.py)
A Python script that will:
1. Parse the extracted HTML using `BeautifulSoup`.
2. Iterate through all paragraphs and headings.
3. Detect chapter boundaries (e.g., `<h2...>` containing "CHƯƠNG X") and create a new markdown file `docs/chuong_X.md` for each.
4. Track images (`v:imagedata` tags) and when a caption like `Hình X-Y. ...` is found, map the last seen image to that ID.
5. Copy the source image from `Hands-On Machine Learning with ScikitLearn, Keras, and TensorFlow2_files` to `machineLearningWeb/Figures/CH0X/Hình X-Y.jpg`.
6. Insert the markdown `![Hình X-Y](../../Figures/CH0X/Hình X-Y.jpg)` into the chapter content.
7. De-duplicate text (because Word-generated HTML often duplicates the same text 2-3 times in fallback shapes).

## Verification Plan
### Automated Tests
- I will run the python script `build_ml_web.py` and monitor its output.
- I will use `list_dir` to check if `machineLearningWeb/Figures/` contains the correctly named images (e.g., `Hình 1-1.png`).
- I will verify the markdown files in `machineLearningWeb/docs/` contain the correct text and image links.

### Manual Verification
- You can open the `index.html` file in your browser to verify the course website looks correct and all images load properly.
