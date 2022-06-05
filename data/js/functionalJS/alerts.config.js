Netdania = Netdania || {};
Netdania.Alert = Netdania.Alert || {};
Netdania.Alert.Config = Netdania.Alert.Config || {};

Netdania.Alert.Config.MaxAlerts = 50;
Netdania.Alert.Config.AddAlertUrlParameter = "alert";
Netdania.Alert.Config.SchemeAlerts = ('https:'==document.location.protocol?'https://':'http://');
Netdania.Alert.Config.HostAlerts = Netdania.Alert.Config.SchemeAlerts + "showroom-balancer2.netdania.com/StreamingServer/StreamingServer";
//Netdania.Alert.Config.HostAlerts = Netdania.Alert.Config.SchemeAlerts + "internet.netdania.com/StreamingServer/StreamingServer";
Netdania.Alert.Config.AlertDetail = "/Alerts/AlertDetail.aspx";
