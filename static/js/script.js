document.getElementById('generate-btn').addEventListener('click', async () => {
    const form = document.getElementById('upload-form');
    const formData = new FormData(form);

    // Displaying loading message
    const responseMessage = document.getElementById('response-message');
    responseMessage.textContent = 'Processing... Please wait!';
    responseMessage.style.color = 'blue';

    try {
        const response = await fetch('/process-images', {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();
        if (result.status === 'success') {
            responseMessage.textContent = result.message;
            responseMessage.style.color = 'green';
        } else {
            responseMessage.textContent = result.message;
            responseMessage.style.color = 'red';
        }
    } catch (error) {
        console.error('Error:', error);
        responseMessage.textContent = 'An error occurred. Please try again.';
        responseMessage.style.color = 'red';
    }
});
