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
        directory = request.form['directory']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        filters = request.form.getlist('filters[]')  # List of filters
        mask_needed = request.form['mask_needed']  # Deciduous or Coniferous

        # Ensure the directory exists
        if not os.path.exists(directory):
            return jsonify({'status': 'error', 'message': 'Provided directory does not exist.'})


        # Convert date strings to datetime objects
        start_time = datetime.strptime(start_date, '%Y-%m-%d')
        end_time = datetime.strptime(end_date, '%Y-%m-%d')

        print("CHECKPOINT1")

        # Call your backend main function
        backend_script.main(directory, start_time, end_time, filters, mask_needed)

        print("HOGYA BADIYAA")

        # Path to the output Excel file
        output_path = "output.xlsx"

        # Return the file as a downloadable response
        return send_file(output_path, as_attachment=True, download_name="output.xlsx")

    except Exception as e:
        print(e)
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
