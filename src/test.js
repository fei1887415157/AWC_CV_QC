// A simple Node.js script to send a GET request to the YOLO inference server and print the JSON response.



import fetch from 'node-fetch';



// --- Configuration ---
// Define the URL of your Python Flask server endpoint
// Make sure the host and port match your running inference.py script
const serverUrl = 'http://127.0.0.1:2000/trigger-inference';



// --- Core Function ---
// Function to send the trigger request to the server and handle the response
async function sendTriggerRequest() {
    console.log(`[INFO] Sending GET request to: ${serverUrl}`);

    try {
        // --- Send the GET request using fetch ---
        const response = await fetch(serverUrl, {
            method: 'GET', // Explicitly set method to GET
            headers: {
                'Accept': 'application/json' // Indicate we expect JSON back
            }
        });

        // --- Check if the response status is OK (e.g., 200-299) ---
        if (!response.ok) {
            // If response is not OK, log the status and try to get error text
            const errorText = await response.text(); // Get response body as text
            console.error(`[ERROR] Server responded with status: ${response.status} ${response.statusText}`);
            console.error(`[ERROR] Server response body: ${errorText}`);
            // Throw an error to be caught by the outer catch block
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // --- Parse the JSON response ---
        // .json() parses the response body as JSON
        const jsonData = await response.json();

        // --- Print the received JSON data ---
        console.log('[INFO] Received JSON response from server:');
        // Use JSON.stringify with indentation for pretty printing
        console.log(JSON.stringify(jsonData, null, 2));

    } catch (error) {
        // --- Handle potential errors (network issues, JSON parsing errors, etc.) ---
        console.error('[ERROR] Failed to send trigger request or process response:');
        console.error(error); // Log the full error object
    }
}



// --- Execution ---
// Immediately call the asynchronous function when the script is run
sendTriggerRequest();
