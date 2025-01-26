from flask import Flask, request, jsonify, render_template, send_file
import os
from datetime import datetime
import backend_script

app = Flask(__name__)

# Serve the frontend
@app.route('/')
def index():
    return render_template('index.html')

# Handle the file upload and parameters
@app.route('/process-images', methods=['POST'])
def process_images():
    try:
        # Get files and form data
        files = request.files.getlist('images')
        start_time = request.form['start_time']  # Time input as HH:mm:ss
        end_time = request.form['end_time']  # Time input as HH:mm:ss
        filters = request.form.getlist('filters[]')  # List of filters
        mask_needed = request.form['mask_needed']  # Deciduous or Coniferous

        # Create a directory for uploaded images
        upload_dir = "uploaded_images"
        os.makedirs(upload_dir, exist_ok=True)

        # Save images to the directory
        for file in files:
            file.save(os.path.join(upload_dir, file.filename))

        # Convert time strings to datetime.time objects (in 24-hour format)
        start_time = datetime.strptime(start_time, '%H:%M:%S').time()
        end_time = datetime.strptime(end_time, '%H:%M:%S').time()

        print("Start Time:", start_time)
        print("End Time:", end_time)

        upload_dir_path='/Users/ishan/Phenovegetation_final-2/uploaded_images'

        # Call your backend main function
        backend_script.main(upload_dir_path, start_time, end_time, filters, mask_needed)

        # Path to the output Excel file
        output_path = "output.xlsx"

        # Return the file as a downloadable response
        return send_file(output_path, as_attachment=True, download_name="output.xlsx")

    except Exception as e:
        print(e)
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
