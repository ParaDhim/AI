// server.sqf

// Function to send message to all clients every 10 milliseconds
private _sendServerMessage = {
    while {true} do {
        // Send message to all clients
        {[_this] remoteExec ["serverMessageReceived", _x]} forEach allPlayers;
        sleep 0.01; // 10 milliseconds
    };
};

// Call the function
[] spawn _sendServerMessage;

// Handle messages received from clients
serverMessageReceived = {
    params ["_msg"];
    hint format ["Server received: %1", _msg];
};
