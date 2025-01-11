// client.sqf

// Function to send message to server every 10 milliseconds
private _sendClientMessage = {
    while {true} do {
        // Send message to server
        ["Hello from client"] remoteExec ["clientMessageReceived", 0];
        sleep 0.01; // 10 milliseconds
    };
};

// Call the function
[] spawn _sendClientMessage;

// Handle messages received from the server
clientMessageReceived = {
    params ["_msg"];
    hint format ["Client received: %1", _msg];
};
