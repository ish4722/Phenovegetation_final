from flask import Flask, request, jsonify, render_template, send_file
import os
import shutil
from datetime import datetime
import backend_script

app = Flask(__name__)

UPLOAD_DIR = "uploaded_images"
OUTPUT_FILE = "output.xlsx"

# Function to delete the uploaded_images directory and output.xlsx file
def cleanup_files():
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

# Serve the frontend and cleanup on refresh
@app.route('/')
def index():
    cleanup_files()  # Delete uploaded files and output.xlsx when refreshing the page
    return render_template('index.html')

# Handle the file upload and parameters
@app.route('/process-images', methods=['POST'])
def process_images():
    try:
        # Get files and form data
<<<<<<< HEAD
        directory = request.form['directory']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        filters = request.form.getlist('filters[]')  # List of filters
        mask_needed = request.form['mask_needed']  # Deciduous or Coniferous

        # Ensure the directory exists
        if not os.path.exists(directory):
            return jsonify({'status': 'error', 'message': 'Provided directory does not exist.'})

=======
        files = request.files.getlist('images')
        start_time = request.form['start_time']  # Time input as HH:mm:ss
        end_time = request.form['end_time']  # Time input as HH:mm:ss
        filters = request.form.getlist('filters[]')  # List of filters
        mask_needed = request.form['mask_needed']  # Deciduous or Coniferous

        # Create a directory for uploaded images
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # Save images to the directory
        for file in files:
            file.save(os.path.join(UPLOAD_DIR, file.filename))
>>>>>>> 34cff043f265f83bd6e03eeffc99c3c0848de148

        # Convert time strings to datetime.time objects (in 24-hour format)
        start_time = datetime.strptime(start_time, '%H:%M:%S').time()
        end_time = datetime.strptime(end_time, '%H:%M:%S').time()

        print("Start Time:", start_time)
        print("End Time:", end_time)

        # Call your backend main function
<<<<<<< HEAD
        backend_script.main(directory, start_time, end_time, filters, mask_needed)

        print("HOGYA BADIYAA")

        # Path to the output Excel file
        output_path = "output.xlsx"
=======
        backend_script.main(UPLOAD_DIR, start_time, end_time, filters, mask_needed)
>>>>>>> 34cff043f265f83bd6e03eeffc99c3c0848de148

        # Return the file as a downloadable response
        return send_file(OUTPUT_FILE, as_attachment=True, download_name="output.xlsx")

    except Exception as e:
        print(e)
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
