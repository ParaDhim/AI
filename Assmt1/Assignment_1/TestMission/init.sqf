// init.sqf

if (isServer) then {
    // Run server-side script
    [] execVM "server.sqf";
} else {
    // Run client-side script
    [] execVM "client.sqf";
};
