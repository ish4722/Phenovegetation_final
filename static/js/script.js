document.addEventListener("DOMContentLoaded", () => {
    const uploadForm = document.getElementById("upload-form");
    const generateBtn = document.getElementById("generate-btn");
    const responseMessage = document.getElementById("response-message");
    const uploadInput = document.getElementById("upload-input");
    const fileChosen = document.getElementById("file-chosen");

    // Update file-chosen text when files are selected
    uploadInput.addEventListener("change", function () {
        if (this.files.length > 0) {
            fileChosen.textContent = `${this.files.length} file(s) chosen`;
        } else {
            fileChosen.textContent = "No files chosen";
        }
    });

    generateBtn.addEventListener("click", async () => {
        responseMessage.innerHTML = "";
        responseMessage.style.display = "none";

        // Get values from number inputs (if using number fields)
        const startHour = document.getElementById("start-hour").value.padStart(2, "0");
        const startMinute = document.getElementById("start-minute").value.padStart(2, "0");
        const startSecond = document.getElementById("start-second").value.padStart(2, "0");

        const endHour = document.getElementById("end-hour").value.padStart(2, "0");
        const endMinute = document.getElementById("end-minute").value.padStart(2, "0");
        const endSecond = document.getElementById("end-second").value.padStart(2, "0");

        // Format as HH:mm:ss
        const startTime = `${startHour}:${startMinute}:${startSecond}`;
        const endTime = `${endHour}:${endMinute}:${endSecond}`;

        // Validate time format (24-hour HH:mm:ss)
        const timeRegex = /^([01]\d|2[0-3]):([0-5]\d):([0-5]\d)$/;
        if (!timeRegex.test(startTime) || !timeRegex.test(endTime)) {
            responseMessage.style.display = "block";
            responseMessage.style.color = "red";
            responseMessage.innerHTML = "<p>Please enter valid times in HH:mm:ss format (24-hour).</p>";
            return;
        }

        if (startTime >= endTime) {
            responseMessage.style.display = "block";
            responseMessage.style.color = "red";
            responseMessage.innerHTML = "<p>Start time must be earlier than end time.</p>";
            return;
        }

        // Ensure form exists before processing
        if (!uploadForm) {
            console.error("Form not found!");
            return;
        }
        

        // Ensure at least one file is uploaded
        const formData = new FormData(uploadForm);
        const files = formData.getAll("images");
        if (files.length === 0) {
            responseMessage.style.display = "block";
            responseMessage.style.color = "red";
            responseMessage.innerHTML = "<p>Please upload at least one image.</p>";
            return;
        }

        // Add the time values to the form data
        formData.set("start_time", startTime);
        formData.set("end_time", endTime);

        // Make a POST request to the server
        try {
            const response = await fetch("/process-images", {
                method: "POST",
                body: formData,
            });

            if (response.ok) {
                // Trigger file download
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.style.display = "none";
                a.href = url;
                a.download = "output.xlsx";
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);

                responseMessage.style.color = "green";
                responseMessage.innerHTML = "<p>Processing complete! Your file is downloading.</p>";
            } else {
                const data = await response.json();
                responseMessage.style.color = "red";
                responseMessage.innerHTML = `<p>Error: ${data.message}</p>`;
            }
        } catch (error) {
            console.error("Error while sending request:", error);
            responseMessage.style.color = "red";
            responseMessage.innerHTML = "<p>Something went wrong. Please try again later.</p>";
        }

        responseMessage.style.display = "block";
    });
});