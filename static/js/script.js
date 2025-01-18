document.addEventListener("DOMContentLoaded", () => {
    const uploadForm = document.getElementById("upload-form");
    const generateBtn = document.getElementById("generate-btn");
    const responseMessage = document.getElementById("response-message");

    generateBtn.addEventListener("click", async () => {
        // Clear the previous response message
        responseMessage.innerHTML = "";
        responseMessage.style.display = "none";

        // Collect form data
        const formData = new FormData(uploadForm);

        // Ensure at least one file is selected
        const files = formData.getAll("images");
        if (files.length === 0) {
            responseMessage.style.display = "block";
            responseMessage.innerHTML = "<p>Please upload at least one image.</p>";
            return;
        }

        // Make a POST request to the Flask server
        try {
            const response = await fetch("/process-images", {
                method: "POST",
                body: formData,
            });

            // If response is ok, trigger file download
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.style.display = "none";
                a.href = url;
                a.download = "output.xlsx"; // Name of the downloaded file
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);

                responseMessage.style.color = "green";
                responseMessage.innerHTML = `<p>Processing complete! Your file is downloading.</p>`;
            } else {
                const data = await response.json();
                responseMessage.style.color = "red";
                responseMessage.innerHTML = `<p>Error: ${data.message}</p>`;
            }

            responseMessage.style.display = "block";
        } catch (error) {
            console.error("Error while sending request:", error);
            responseMessage.style.color = "red";
            responseMessage.style.display = "block";
            responseMessage.innerHTML = "<p>Something went wrong. Please try again later.</p>";
        }
    });
});
