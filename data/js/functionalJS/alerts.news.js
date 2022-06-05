Netdania = Netdania || {};
Netdania.Alert = Netdania.Alert || {};

Netdania.Alert.GetConnection = function () {

    var _connection;

    try {
        var connectionConfig = {
            pollingInterval: Netdania.JsApi.PoolingInterval.AUTO, //milliseconds
            host: Netdania.Alert.Config.HostAlerts,
            behavior: Netdania.JsApi.ConnectionType.AUTO, //LONG POLLING
            type: 1 //the server response format: JSONP
        }
        //create a json connection object
        _connection = new Netdania.JsApi.JSONConnection(connectionConfig);
    } catch
        (err) {
        Netdania.JsApi.Utilities.LogError("Netdania.Alert.GetConnection", err);
    }

    return  _connection;
};

Netdania.Alert.MonitorTriggeredAlerts = function(data){

    var activeAlerts = new Netdania.JsApi.Alert.AlertObject();
    activeAlerts.USER_ID = data.userID;
    activeAlerts.TIMESCALE_SECONDS = true;
    activeAlerts.COOKIE = "NDWebsite";

    var _connection = Netdania.Alert.GetConnection();

    var req = Netdania.JsApi.Request.getReqObjAlertMonitorUser(activeAlerts);
    var _lastRequestID = req.i;
    _connection.AddRequest(req);
    _connection.Flush();

    _connection.addListener(Netdania.JsApi.Events.ONALERTMONITORUSER, function (alertTriggeredResponse) {
        try {
            if (alertTriggeredResponse.type == Netdania.JsApi.Alert.Commands.NOTIFICATION && _lastRequestID == alertTriggeredResponse.id) {
                var alertID = _connection.getFieldValueFromSource(alertTriggeredResponse.data.f, Netdania.JsApi.Alert.Fields.ALERT_ID)[0];

                var _returnGetAlert = function (alertObject) {
                    if (alertObject != undefined) {
                        /*
                         Get the news object
                         */
                        var _alertsNews = Netdania.Alert.AlertStorageToNewsTransform([alertObject]);

                        /*
                         Send alert to the return function
                         */
                        Netdania.Alert.SendAlerts([alertObject], Netdania.News.Monitor.News.Events.ALERTS_UPDATE, data.requestID, null);
                    }
                }

                Netdania.Alert.GetTriggeredAlertFromServer($.extend({_returnFunction:_returnGetAlert, alertID: alertID}, data));
            }
        } catch (err) {
            Netdania.JsApi.Utilities.LogError("ONALERTMONITORUSER", err);
        }
    });

}

Netdania.Alert.GetTriggeredAlertFromServer = function (data) {

    try {

        /*
        Get the latest news date from the storage
         */
        var alerts = Netdania.Alert.TriggeredAlertStorage.GetAllAlerts(data);

        /*
        Get time in milliseconds
         */
        var startDate = alerts.lastAlertTime;
        var endDate = new Date().getTime();

        /*
        Make the request from latest news from current date
         */
        var activeAlerts = new Netdania.JsApi.Alert.AlertObject();
        activeAlerts.USER_ID = data.userID;
        activeAlerts.ALERTS_INCLUDED = true;
        activeAlerts.TIMESCALE_SECONDS = true;
        activeAlerts.STARTDATE = startDate;
        activeAlerts.ENDDATE = endDate;
        activeAlerts.COOKIE = "NDWebsite";

        var req10 = Netdania.JsApi.Request.getReqObjAlertGetTriggered(activeAlerts);
        var _lastReqID = req10.i;
        var _connection = Netdania.Alert.GetConnection();
        _connection.AddRequest(req10);
        _connection.Flush();

        _connection.addListener(Netdania.JsApi.Events.ONALERTGETTRIGGERED, function (alertAddedResponse) {
            try {
                if (alertAddedResponse.type == Netdania.JsApi.Alert.Commands.GET_TRIGGERED_ALERTS && _lastReqID == alertAddedResponse.id) {
                    var _alerts = Netdania.Alert.GetSentDeleteAlertsList(alertAddedResponse);
                    /*
                     * Save alerts to storage
                     */
                    Netdania.Alert.TriggeredAlertStorage.AddAlerts({userID: data.userID, alerts: _alerts, overwrite:true});

                    /*
                    Get the search alert
                     */
                    var searchAlert = Netdania.Alert.TriggeredAlertStorage.GetAlert({userID: data.userID, alertID: data.alertID});

                    /*
                    Return the alert to the function
                     */
                    data._returnFunction(searchAlert);
                }
            }
            catch
                (err) {
                Netdania.JsApi.Utilities.LogError("Netdania.Alert.GetTriggeredAlertFromServer.ONALERTGETTRIGGERED", err);
            }
        });
    } catch
        (err) {
        Netdania.JsApi.Utilities.LogError("Netdania.Alert.GetTriggeredAlertFromServer", err);
    }

};

/*
 Extract the alerts from the response
 */
Netdania.Alert.GetSentDeleteAlertsList = function (alertResponse) {

    var _alertsList = [];
    try {
        if (alertResponse != undefined && alertResponse.data != undefined && alertResponse.data.f != undefined) {
            for (var i = 0; i < alertResponse.data.f.length; i++) {
                if (alertResponse.data.f[i].f == Netdania.JsApi.Alert.Fields.ALERTS_HOLDER) {
                    for (var j = 0; j < alertResponse.data.f[i].v.length; j++) {
                        if (alertResponse.data.f[i].v[j].f != undefined) {

                            var config = {};

                            for (var k = 0; k < alertResponse.data.f[i].v[j].f.length; k++) {

                                switch (alertResponse.data.f[i].v[j].f[k].f) {

                                    case Netdania.JsApi.Alert.Fields.EMAIL_CONTENT:
                                    {
                                        config.emailContent = alertResponse.data.f[i].v[j].f[k].v[0];
                                        var triggerLevel = config.emailContent.split("Trigger level: ")[1];
                                        if (triggerLevel != undefined){
                                            triggerLevel = triggerLevel.split("\n")[0];
                                            config.triggerLevel = triggerLevel;
                                        }

                                        var alertLevel = config.emailContent.split("Alert level: ")[1];
                                        if (alertLevel != undefined){
                                            alertLevel = alertLevel.split("\n")[0];
                                            config.alertLevel = alertLevel;
                                        }
                                        break;
                                    }
                                    case Netdania.JsApi.Alert.Fields.ALERT_ID:
                                    {
                                        config.alertID = alertResponse.data.f[i].v[j].f[k].v[0];
                                        break;
                                    }
                                    case Netdania.JsApi.Alert.Fields.TRIGGER_DATE:
                                    {
                                        config.triggerDate = alertResponse.data.f[i].v[j].f[k].v[0];
                                        break;
                                    }
                                    case Netdania.JsApi.Alert.Fields.ALERT_CONDITION:
                                    {
                                        config.alertCondition = alertResponse.data.f[i].v[j].f[k].v[0];
                                        break;
                                    }
                                    case Netdania.JsApi.Alert.Fields.DAYS_TO_LIVE:
                                    {
                                        config.daysToLive = alertResponse.data.f[i].v[j].f[k].v[0];
                                        break;
                                    }
                                    case Netdania.JsApi.Alert.Fields.EMAIL_ADDRESS:
                                    {
                                        config.emailAddress = alertResponse.data.f[i].v[j].f[k].v[0];
                                        break;
                                    }
                                    case Netdania.JsApi.Alert.Fields.EMAIL_FROM:
                                    {
                                        config.emailFrom = alertResponse.data.f[i].v[j].f[k].v[0];
                                        break;
                                    }
                                    case Netdania.JsApi.Alert.Fields.EMAIL_SUBJECT:
                                    {
                                        config.emailSubject = alertResponse.data.f[i].v[j].f[k].v[0];
                                        break;
                                    }
                                    case Netdania.JsApi.Alert.Fields.YAHOO_ID:
                                    {
                                        config.yahooIDs = alertResponse.data.f[i].v[j].f[k].v[0];
                                        break;
                                    }
                                    case Netdania.JsApi.Alert.Fields.SKYPE_ID:
                                    {
                                        config.skypeIDs = alertResponse.data.f[i].v[j].f[k].v[0];
                                        break;
                                    }
                                    case Netdania.JsApi.Alert.Fields.MSN_ID:
                                    {
                                        config.msnIDs = alertResponse.data.f[i].v[j].f[k].v[0];
                                        break;
                                    }
                                    case Netdania.JsApi.Alert.Fields.PUSH_DEVICES:
                                    {
                                        config.pushDevices = alertResponse.data.f[i].v[j].f[k].v[0];
                                        break;
                                    }

                                    case Netdania.JsApi.Alert.Fields.DELETION_DATE:
                                    {
                                        config.deletionDate = alertResponse.data.f[i].v[j].f[k].v[0];
                                        break;
                                    }

                                    case Netdania.JsApi.Alert.Fields.DELETION_REASON:
                                    {
                                        config.deletionReason = alertResponse.data.f[i].v[j].f[k].v[0];
                                        break;
                                    }
                                }
                            }

                            var alertWrapper = new Netdania.Alert.AlertWrapper(config);
                            alertWrapper.setConditionToObject(config.alertCondition);
                            _alertsList.push(alertWrapper.GetStorageObject());
                        }
                    }
                }
            }
        }
    }
    catch
        (err) {
        Netdania.JsApi.Utilities.LogError("Netdania.Alert.GetAllSentAlerts", err);
    }

    return _alertsList;
}

Netdania.Alert.GetAllSentAlerts = function (data) {

    var _connection = Netdania.Alert.GetConnection();

    var activeAlerts = new Netdania.JsApi.Alert.AlertObject();
    activeAlerts.USER_ID = data.userID;
    activeAlerts.ALERTS_INCLUDED = true;
    activeAlerts.TIMESCALE_SECONDS = true;
    activeAlerts.STARTDATE = data.dateStart;
    activeAlerts.ENDDATE = data.dateEnd;
    activeAlerts.COOKIE = "NDWebsite";


    var req01 = Netdania.JsApi.Request.getReqObjAlertAddUser(activeAlerts);
    _connection.AddRequest(req01);
    _connection.Flush();


    var req10 = Netdania.JsApi.Request.getReqObjAlertGetTriggered(activeAlerts);
    var _lastReqID = req10.i ;

    /*Add a constant number so the news requests will not be the same */
    var _alertReqID = _lastReqID + 9000;
    _connection.AddRequest(req10);
    _connection.Flush();

    _connection.addListener(Netdania.JsApi.Events.ONALERTGETTRIGGERED, function (alertAddedResponse) {
        try {
            if (alertAddedResponse.type == Netdania.JsApi.Alert.Commands.GET_TRIGGERED_ALERTS && _lastReqID == alertAddedResponse.id) {
                var _alerts = Netdania.Alert.GetSentDeleteAlertsList(alertAddedResponse);
                /*
                * Save alerts to storage
                */
                Netdania.Alert.TriggeredAlertStorage.AddAlerts({userID:data.userID, alerts:_alerts});

                /*
                 * Combine with alerts from storage
                 */

                if (data.storage != null && data.storage != undefined && data.storage.data != null && data.storage.data != undefined)
                {
                    var combinedAlerts = _alerts.concat(data.storage.data);
                    for(var i=0; i<combinedAlerts.length; ++i) {
                        for(var j=i+1; j<combinedAlerts.length; ++j) {
                            if(combinedAlerts[i].id === combinedAlerts[j].id) {
                                combinedAlerts.splice(j--, 1);
                            }

                            if (combinedAlerts[i].triggerDate > data.dateEnd ||  combinedAlerts[i].triggerDate < data.dateStart){
                                combinedAlerts.splice(j--, 1);
                            }
                        }
                    }

                    _alerts =  combinedAlerts;
                }

                /*
                Filter alerts between dates
                 */

                 Netdania.Alert.SendAlerts(_alerts, Netdania.News.Monitor.News.Events.ALERTS_SEARCH, _alertReqID, data);
            }
        }
        catch
            (err) {
            Netdania.JsApi.Utilities.LogError("Netdania.Alert.GetAllSentAlerts.ONALERTGETTRIGGERED", err);
        }
    });

    _connection.addListener(Netdania.JsApi.Events.ONERROR, function (alertErrorResponse) {
        if (alertErrorResponse && alertErrorResponse.type == Netdania.JsApi.Alert.Commands.ERROR && _lastReqID == alertErrorResponse.id) {
            _connection.removeListener(Netdania.JsApi.Events.ONALERTGETTRIGGERED);
        }
    });



    return _alertReqID;
};

Netdania.Alert.SendAlerts = function (alerts, type, requestID, data) {
    var _alertsNews = [];

    if (alerts.length > 0) {

        /*Transform*/
        var _alertsNews = Netdania.Alert.AlertStorageToNewsTransform(alerts);

        /*Sort and trim*/
        if (data != null && data != undefined && data.headlines != null && data.headlines != undefined) {

            //sort the new data array
            _alertsNews = _alertsNews.sort(function (a, b) {
                var _aT = a.t;
                var _bT = b.t;

                return _bT - _aT;
            });
            _alertsNews = _alertsNews.slice(0, data.headlines);
        }
    }

    /*Send*/
    var _news = Netdania.News.Monitor.News.getInstance();
    _news.fireEvent(type, [
        {id: requestID, data: _alertsNews}
    ]);
};

Netdania.Alert.AlertStorageToNewsTransform = function (alertsList) {
    var news = [], _r;
    $.each(alertsList, function (i) {
        try {
            var item = alertsList[i];
            var triggerDate = new Date(item.triggerDate)/1000;

            _r = {

                data: {
                    id: i,
                    t: triggerDate,
                    s: 1,
                    i: item.id.toString(),
                    h: (item.type == Netdania.Alert.Types.TRENDLINE ? (item.name + ' ' + item.triggerLevel + ' ' + item.alertValue + ' ' + item.alertLevel) : (item.name + ' ' + item.fieldName + ' ' + item.operator + ' ' + item.alertValue)),
                    source: 'Triggered',
                    provider: 'NetdaniaAlerts',
                    alias: 'Netdania Alerts'
                }
            };
            news.push(_r.data);
        }
        catch (err) {
            Netdania.JsApi.Utilities.LogError("Netdania.Alert.AlertStorageToNewsTransform", err);
        }
    });
    return news;
}

Netdania.Alert.HeadlinesSentAlerts = function (data) {

    var headlines = data.headlines;
    var user = data.userID;
    var control = data.control;
    var requestID = {i:Math.floor((Math.random() * 100) + 1)};
    var dateStart, dateEnd;

    try {
        /*
         Get alerts from the storage
         */
        var storageAlerts = Netdania.Alert.TriggeredAlertStorage.GetAllAlerts(data);

        /*
        * Get alerts from the latest alert date to current date
        * */
        if (storageAlerts != null && storageAlerts != undefined && storageAlerts.lastAlertTime != undefined){
            dateStart = new Date(storageAlerts.lastAlertTime).getTime();
            dateEnd = new Date().getTime();
        }
        else
        {
            /*
            No alerts from storage we get alerts from latest 30 days
             */
            var _oneDayMS = 1000 * 24 * 60 * 60;
            var dateStartAlerts = (new Date(new Date() - data.maxNewsDisplayDays * _oneDayMS)).getTime();
            dateStart = dateStartAlerts;
            dateEnd = new Date().getTime();
        }

        requestID = Netdania.Alert.GetAllSentAlerts({dateStart:dateStart, dateEnd:dateEnd, userID:user, control:control, headlines:headlines, storage:storageAlerts, maxNewsDisplayDays:data.maxNewsDisplayDays});

        Netdania.Alert.MonitorTriggeredAlerts({userID:user, requestID:requestID});
    }
    catch (err) {
        Netdania.JsApi.Utilities.LogError("Netdania.Alert.AlertStorageToNewsTransform", err);
    }

    return {i: requestID};
};

Netdania.Alert.SearchSentAlerts = function (data) {

    var headlines = data.headlines;
    var user = data.userID;
    var control = data.control;
    var requestID = Math.floor((Math.random() * 100) + 1);
    /*
    News time is in unix timestamp and alerts are in milliseconds
     */
    data.dateStart = data.dateStart * 1000;
    data.dateEnd = data.dateEnd * 1000;

    var dateStart = data.dateStart;
    var dateEnd = data.dateEnd;

    try {
        /*
         Get alerts from the storage
         */
        var storageAlerts = Netdania.Alert.TriggeredAlertStorage.GetAllAlerts(data);

        var loadFromStorage = false;
        /*
         Get alerts from now to the latest alert date
         */
        if (storageAlerts != null && storageAlerts != undefined && storageAlerts.lastAlertTime != undefined) {
            if (data.dateStart == 0 && dateEnd <= storageAlerts.lastAlertTime && dateEnd > storageAlerts.firstAlertTime){
                loadFromStorage = true;
            }
            else if (dateStart > storageAlerts.firstAlertTime && dateEnd > storageAlerts.lastAlertTime) {
                dateStart = storageAlerts.lastAlertTime;
            }
            else if (dateStart < storageAlerts.firstAlertTime && dateEnd < storageAlerts.lastAlertTime) {
                dateEnd = storageAlerts.firstAlertTime;
            }
            else if (dateStart > storageAlerts.firstAlertTime && dateEnd < storageAlerts.lastAlertTime) {
                loadFromStorage = true;
            }
        }
        if (loadFromStorage) {
            /*
            Get only the alert from the period
             */
            storageAlerts = Netdania.Alert.TriggeredAlertStorage.GetAllAlertsPeriod(data);
            setTimeout(function(){Netdania.Alert.SendAlerts(storageAlerts.data, Netdania.News.Monitor.News.Events.ALERTS_SEARCH, requestID, data);}, 500);

        }
        else {
            dateStart = new Date(dateStart).getTime();
            dateEnd = new Date(dateEnd).getTime();
            requestID = Netdania.Alert.GetAllSentAlerts({dateStart: dateStart, dateEnd: dateEnd, userID: user, control: control, headlines: headlines, storage: storageAlerts});
        }
    }
    catch (err) {
        Netdania.JsApi.Utilities.LogError("Netdania.Alert.SearchSentAlerts", err);
    }

    return {i: requestID};
};

Netdania.Alert.DisplayDetail = function () {

    try {
        var alertID;
        if (queryString('id') !== 'false') {
            alertID = parseInt(queryString('id'));
        }

        var headline = $('#nd-alert-headline');
        var message = $('#nd-alert-message');
        var message1 = $('#nd-alert-message1');
        var time = $('#nd-alert-time');
        var alertSet = $('#nd-alert-set');
        var alertQL = $('#nd-alert-ql');
        var alertAll = $('#nd-alert-all');



        if (alertID != null && alertID != undefined) {
            var _user = new Netdania.Util.User();
            var alertObject = Netdania.Alert.TriggeredAlertStorage.GetAlert({alertID:alertID, userID:_user.getUserId()});
            if (alertObject != undefined) {
                var decimals = Netdania.Util.CountDecimals(alertObject.triggerLevel);

                var ALERT_NOTIFICATION = "<div><b>Instrument name:</b> {0} <br /> <b>Field:</b> {1} <br /> <b>Alert level:</b> {2} <br /> <b>Trigger level:</b> {3} <br /> <b>Received at:</b> {4}  <br /> <b>Message:</b> {5}</div>";
                if (alertObject.type == Netdania.Alert.Types.TRENDLINE) {
                    headline.html(alertObject.name + ' ' + alertObject.triggerLevel + ' ' + alertObject.alertValue + ' ' + alertObject.alertLevel);
                    message.html(ALERT_NOTIFICATION
                        .replace("{0}", alertObject.name)
                        .replace("{1}", alertObject.fieldName)
                        .replace("{2}", alertObject.alertLevel)
                        .replace("{3}", alertObject.triggerLevel)
                        .replace("{4}", new Date(alertObject.triggerDate).toString())
                        .replace("{5}", alertObject.message));
                } else {
                    headline.html(alertObject.name + ' ' + alertObject.fieldName + ' ' + alertObject.operator + ' ' + alertObject.alertValue);
                    message.html(ALERT_NOTIFICATION
                        .replace("{0}", alertObject.name)
                        .replace("{1}", alertObject.fieldName)
                        .replace("{2}", alertObject.alertValue)
                        .replace("{3}", alertObject.triggerLevel)
                        .replace("{4}", new Date(alertObject.triggerDate).toString())
                        .replace("{5}", alertObject.message));
                }


                time.html(new Date(alertObject.triggerDate).format("dddd, MMMM dd yyyy, h:mm:ss"));
                alertSet.html('<div>' + Netdania.Alert.AlertMonitor.AlertImage({s:alertObject.symbol, p:alertObject.provider, d:decimals}) + '<b>Set Alert</b></div>');

                alertQL.html('View FullQuote');
                alertQL.button();
                alertQL.click(function (evt) {
                    window.location.href = "/Products/FullQuote/FullQuote.aspx?symbol=" + alertObject.symbol + "&provider=" + alertObject.provider + "&decimals="+ decimals;
                });

                alertAll.html('Alert Central');
                alertAll.button();
                alertAll.click(function (evt) {
                    window.location.href = "/Alerts/AlertsCentral.aspx";
                });
            }
            else {
                headline.html("Alert not found.");
                message.html("Alert not found.");
            }

        }
    }
    catch
        (err) {
        Netdania.JsApi.Utilities.LogError("Netdania.Alert.DisplayDetail", err);
    }

};

Netdania.Alert.TriggeredAlertStorage = (function () {
    var _storageVersion = '1.4'
    var _storageName = "triggeredAlerts";
    var _alertsObject = {};
    var _maxStorageAlertsDays = 30;

    var GetAlertsFromStorage = function () {
        var defaultObject = { users: [], version: _storageVersion};

        var alertsObject = defaultObject;
        try {
            var alertsString = $.ezstorage.get(_storageName);
            if (alertsString !== undefined) {
                if (typeof alertsString == "string") {
                    alertsObject = JSON.parse(alertsString);
                    if (alertsObject !== undefined && typeof alertsObject == "string") {
                        alertsObject = JSON.parse(alertsObject);
                    }
                }
                else {
                    alertsObject = alertsString;
                }

                /*
                 Check if we have the same version
                 */
                if (alertsObject == null || alertsObject == undefined || alertsObject.version == null || alertsObject.version == undefined || alertsObject.version != _storageVersion) {
                    alertsObject = defaultObject;
                }
            }

        } catch (err) {
            Netdania.JsApi.Utilities.LogError("AlertStorage.GetAlertsFromStorage error", err);
        }
        return  alertsObject;
    }

    var _generateAlertsTime = function (alerts) {
        /*
         Calculate first and the last news time
         */
        var lastAlertTime = 0;
        var firstAlertTime = new Date().getTime();

        if (alerts.data.length > 0) {
            $.each(alerts.data, function (n) {
                if (lastAlertTime < alerts.data[n].triggerDate) {
                    lastAlertTime = alerts.data[n].triggerDate;
                }

                if (firstAlertTime > alerts.data[n].triggerDate) {
                    firstAlertTime = alerts.data[n].triggerDate;
                }
            });

            alerts.lastAlertTime = lastAlertTime;
            alerts.firstAlertTime = firstAlertTime;
        }
    }

    var SetAlertsToStorage = function () {
        try {
            $.ezstorage.set(_storageName, _alertsObject, {persist: true});
            var alertsString = $.ezstorage.get(_storageName);
        } catch
            (err) {
            Netdania.JsApi.Utilities.LogError("AlertStorage.SetAlertsToStorage error", err);
        }
    }

    var DeleteAlertsFromStorage =  function() {
        try {
            $.ezstorage.remove(_storageName);
        } catch
            (err) {
            Netdania.JsApi.Utilities.LogError("MonitorUserAlerts.DeleteAlertsFromStorage error", err);
        }
    }

    var _userAlerts = function (users, userID) {

        var alerts;
        try {
            if (userID != null && userID != undefined) {

                var found = false;
                $.each(users, function (i) {
                    /*Find user match*/
                    if (userID == users[i].id) {
                        users[i].alerts = users[i].alerts || [];
                        alerts = users[i].alerts;
                        found = true;
                        return false;
                    }
                });

                if (!found) {
                    users.push({id: userID, alerts: {data:[]}});
                    alerts = users[0].alerts;
                }
            }
        }
        catch
            (err) {
            Netdania.JsApi.Utilities.LogError("Netdania.Alert.TriggeredAlertStorage._userAlerts", err);
        }

        return alerts;
    };

    var _alertsPeriod = function (alerts, dateStart, dateEnd, headlines) {

        var periodAlerts = {data:[]};
        try {
            if (alerts != null && alerts.data != undefined) {
                var count = 0;
                $.each(alerts.data, function (i) {
                    if (count < headlines) {
                        if (dateStart == 0 || (dateStart != 0 && alerts.data[i].triggerDate >= dateStart)) {
                            if (alerts.data[i].triggerDate <= dateEnd) {
                                periodAlerts.data.push(alerts.data[i]);
                                count++;
                            }
                        }
                    }
                    else {
                        return false;
                    }
                });
                _generateAlertsTime(periodAlerts);
            }
        }
        catch
            (err) {
            Netdania.JsApi.Utilities.LogError("Netdania.Alert.TriggeredAlertStorage._userAlerts", err);
        }

        return periodAlerts;
    };


    var _sortAlerts = function (alerts) {
        try {
            if (alerts.data.length > 0) {
                //sort the new data array
                alerts.data = alerts.data.sort(function (a, b) {
                    var _aT = a.triggerDate;
                    var _bT = b.triggerDate;

                    return _bT - _aT;
                });
            }
        } catch
            (err) {
            Netdania.JsApi.Utilities.LogError("Netdania.Alert.TriggeredAlertStorage._sortAlerts error", err);
        }
    }

    /*
    Remove the older alerts (alerts with date smaller than 30 days ago)
     */
    var _cleanAlerts = function (alerts) {
        try {
            if (alerts.data.length > 0) {
                var _oneDayMS = 1000 * 24 * 60 * 60;
                var dateStartLimit = (new Date(new Date() - _maxStorageAlertsDays * _oneDayMS)).getTime();

                $.each(alerts.data, function (i) {
                    if (alerts.data[i].triggerDate < dateStartLimit) {
                        alerts.data.slice(i, 1);
                    }
                });
            }
        }
        catch
            (err) {
            Netdania.JsApi.Utilities.LogError("Netdania.Alert.TriggeredAlertStorage._cleanAlerts error", err);
        }
    }


    return {
        AddAlerts: function (data) {
            try {
                if (data != null && data != undefined && data.alerts != undefined) {
                    _alertsObject = GetAlertsFromStorage();
                    if (_alertsObject.users != null && _alertsObject.users != undefined) {
                        /*Get alerts for user*/
                        var storageAlerts = _userAlerts(_alertsObject.users, data.userID);

                        $.each(data.alerts, function (i) {
                            var found = false;
                            if (storageAlerts != undefined && storageAlerts.data != undefined) {
                                $.each(storageAlerts.data, function (s) {
                                    if (storageAlerts.data[s].id == data.alerts[i].id) {
                                        found = true;
                                        /*
                                        If we have the option to overwrite
                                         */
                                        if (data.overwrite){
                                            storageAlerts.data[s] = data.alerts[i];
                                        }
                                        return false;
                                    }
                                });
                            }
                            if (!found) {
                                storageAlerts.data.push(data.alerts[i]);
                            }
                        });

                        _cleanAlerts(storageAlerts);

                        _sortAlerts(storageAlerts);

                        _generateAlertsTime(storageAlerts);
                    }
                }
            } catch
                (err) {
                Netdania.JsApi.Utilities.LogError("AlertStorage.AddAllAlerts error", err);
            }
            SetAlertsToStorage();
        },
        SaveAlert: function (data) {
            try {
                _alertsObject = GetAlertsFromStorage();
                if (_alertsObject.users != null && _alertsObject.users != undefined) {
                    /*Get alerts for user*/
                    var storageAlerts = _userAlerts(_alertsObject.users, data.userID);

                    var found = false;
                    $.each(storageAlerts.data, function (s) {
                        if (storageAlerts.data[s].id == data.alert.id) {
                            found = true;
                            return false;
                        }
                    });
                    if (!found) {
                        storageAlerts.data.push(data.alert);
                    }

                    _cleanAlerts(storageAlerts);

                    _sortAlerts(storageAlerts);

                    _generateAlertsTime(storageAlerts);

                }
            } catch
                (err) {
                Netdania.JsApi.Utilities.LogError("AlertStorage.SaveAlert error", err);
            }
            SetAlertsToStorage();
        },
        GetAlert: function (data) {
            _alertsObject = GetAlertsFromStorage();
            var alert = undefined;
            try {
                if (_alertsObject.users != null && _alertsObject.users != undefined) {
                    /*Get alerts for user*/
                    var alerts = _userAlerts(_alertsObject.users, data.userID);
                    $.each(alerts.data, function (i) {
                        if (alerts.data[i].id == data.alertID) {
                            alert = alerts.data[i];
                            return false;
                        }
                    });
                }
            } catch
                (err) {
                Netdania.JsApi.Utilities.LogError("Netdania.Alert.TriggeredAlertStorage.GetAlert error", err);
            }
            return alert;
        },
        GetAllAlerts: function (data) {
            var alerts = undefined;
            try {
                if (data.maxNewsDisplayDays != undefined){
                    _maxStorageAlertsDays = data.maxNewsDisplayDays
                }

                _alertsObject = GetAlertsFromStorage();
                if (_alertsObject.users != null && _alertsObject.users != undefined) {
                    /*Get alerts for user*/
                    alerts = _userAlerts(_alertsObject.users, data.userID);
                }
            } catch
                (err) {
                Netdania.JsApi.Utilities.LogError("Netdania.Alert.TriggeredAlertStorage.GetAllAlerts", err);
            }
            return alerts;
        },
        GetAllAlertsPeriod: function (data){
            var alerts = undefined;
            try {
                _alertsObject = GetAlertsFromStorage();
                if (_alertsObject.users != null && _alertsObject.users != undefined) {
                    /*Get alerts for user*/
                    alerts = _userAlerts(_alertsObject.users, data.userID);
                    alerts = _alertsPeriod(alerts, data.dateStart, data.dateEnd, data.headlines);
                }
            } catch
                (err) {
                Netdania.JsApi.Utilities.LogError("Netdania.Alert.TriggeredAlertStorage.GetAllAlertsPeriod", err);
            }
            return alerts;
        },
        GetAllAlertsCount: function (){
            var count = 0;
            var alerts = this.GetAllAlerts();
            if (alerts != null && alerts != undefined){
                count = alerts.length;
            }
            return count;
        }

    }

})();

