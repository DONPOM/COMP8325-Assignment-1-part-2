
        window.onbeforeunload = function () {
            var host = Netdania.General.Config.Host;
            connectionConfig = {
                pollingInterval: 3000, //milliseconds
                host: host,
                behavior: Netdania.JsApi.ConnectionType.LONGPOLLING, //LONG POLLING
                type: 1 //the server response format: JSONP
            }
            connection = new Netdania.JsApi.JSONConnection(connectionConfig);
            connection.Disconnect();
        };
    